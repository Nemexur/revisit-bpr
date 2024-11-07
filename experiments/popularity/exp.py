# pyright: reportOptionalMemberAccess=false

from typing import Any, Callable, cast
from copy import deepcopy
from pathlib import Path
import re

from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from hydra.utils import instantiate
from ignite.engine import EventEnum, Engine
from rich import print_json
import torch
from torch.utils.data import DataLoader

from experiments import settings
from experiments.base import Experiment
from experiments.options import (
    attach_best_exp_saver,
    attach_checkpoint_loader,
    attach_checkpointer,
    attach_debug_handler,
    attach_log_epoch_metrics,
    attach_metrics,
    attach_output_saver,
    attach_params_watcher,
    attach_preemptible,
    attach_progress_bar,
    attach_user_metric_saver,
)
from experiments.trainer import ModelEvents, Trainer
from experiments.utils import flatten_config
from revisit_bpr.metrics import Metric


class PopularityExperiment(Experiment):
    def __init__(
        self,
        exp_config: dict[str, Any] | Callable[[], dict[str, Any]],
        dir: Path | None = None,
        n_checkpoints: int = 2,
        metrics: dict[str, Metric] | None = None,
        trackers_params: dict[str, Any] | None = None,
        events: dict[str, list[tuple[EventEnum, Callable]]] | None = None,
        seed: int = 13,
        debug: bool = False,
        skip_seen: bool = True,
        save_logits: bool = False,
        save_user_metrics: bool = False,
    ) -> None:
        self._config = exp_config if isinstance(exp_config, dict) else exp_config()
        self._dir = dir
        self._n_checkpoints = n_checkpoints
        self._seed = seed
        self._debug = debug
        self._skip_seen = skip_seen
        self._save_logits = save_logits
        self._save_user_metrics = save_user_metrics
        self._metrics = metrics or {}
        self._trackers_params = trackers_params or {}
        self._events = events or {}

    @property
    def metrics(self) -> dict[str, Any]:
        return self._state.metrics

    def run(self) -> Any:
        self._accelerator = self._get_accelerator()
        if self._accelerator.is_local_main_process:
            print_json(data=self._config)
        self._model = self._accelerator.prepare(instantiate(self._config["model"]))
        max_iters = {k: d.pop("max_iters", None) for k, d in self._config["datasets"].items()}
        self._datasets = {
            key: self._accelerator.prepare_data_loader(
                instantiate(
                    loader,
                    generator=torch.Generator().manual_seed(self._seed),
                    worker_init_fn=seed_worker,
                ),
            )
            for key, loader in self._config["datasets"].items()
        }
        self.trainer = self._get_trainer(self._model, self._datasets)
        self._load_checkpoint_if_needed()
        self._state = self.trainer.run(self._datasets, max_iters=max_iters)
        if self._save_logits and self._dir is not None:
            attach_output_saver(self.trainer, dir=self._dir)
            self.trainer.engines["eval"].run(
                self._datasets.get("eval"), epoch_length=max_iters.get("eval")
            )
        if self._save_user_metrics and self._dir is not None:
            metrics_copy = deepcopy(self._metrics or {})
            for m in metrics_copy.values():
                m.reset()
            attach_user_metric_saver(self.trainer, dir=self._dir, metrics=metrics_copy)
            self.trainer.engines["eval"].run(
                self._datasets.get("eval"), epoch_length=max_iters.get("eval")
            )
        self._accelerator.wait_for_everyone()
        self._accelerator.end_training()

    def interrupt(self) -> None:
        for e in self.trainer.engines.values():
            e.interrupt()

    def clean(self) -> None:
        self._accelerator.free_memory()
        del self._accelerator, self.trainer

    def _get_accelerator(self) -> Accelerator:
        accelerator = Accelerator(
            log_with=list(self._trackers_params) if len(self._trackers_params) > 0 else None,
        )
        if self._dir is not None:
            accelerator.project_configuration = ProjectConfiguration(
                project_dir=str(self._dir),
                automatic_checkpoint_naming=True,
            )
        accelerator.init_trackers(
            project_name=settings.TRACKER_PROJECT,
            config=flatten_config(self._config),
            init_kwargs=self._trackers_params,
        )
        self._seed_everything()
        for m in self._metrics.values():
            m.set_accelerator(accelerator)
        return accelerator

    def _get_trainer(
        self,
        model: torch.nn.Module,
        datasets: dict[str, DataLoader],
    ) -> Trainer:
        trainer = Trainer(
            model,
            optimizer=None,  # Popularity model does not have a loss
            accelerator=self._accelerator,
        )
        if self._skip_seen:
            trainer.add_event("eval", ModelEvents.FORWARD_COMPLETED, self._remove_seen_items)
        attach_preemptible(trainer, self._accelerator)
        attach_params_watcher(trainer, accelerator=self._accelerator)
        if self._debug:
            attach_debug_handler(trainer, num_iters=2000)
        attach_metrics(trainer, self._accelerator, self._metrics)
        if self._accelerator.is_local_main_process:
            attach_progress_bar(trainer, metric_names={"eval": list(self._metrics)[:5]})
        attach_log_epoch_metrics(trainer, self._accelerator)
        if self._dir is not None:
            attach_checkpointer(
                trainer, self._accelerator, checkpoint_objects=self._metrics.values()
            )
            attach_checkpoint_loader(trainer, self._accelerator, datasets)
            if self._accelerator.is_local_main_process:
                attach_best_exp_saver(trainer, self._dir)
        for key, e in self._events.items():
            for event, handler in e:
                trainer.add_event(key, event, handler, accelerator=self._accelerator)
        return trainer

    def _load_checkpoint_if_needed(self) -> None:
        if self._dir is None:
            return
        checkpoints = self._dir / settings.CHECKPOINTS_DIR
        if not checkpoints.exists():
            return
        self._accelerator.load_state()
        # Accelerator requires manually configured save_iteration
        save_iteration = max(
            int(re.findall(r"[\/]?([0-9]+)(?=[^\/]*$)", str(dir))[0])
            for dir in checkpoints.iterdir()
        )
        self._accelerator.project_configuration.iteration = save_iteration + 1

    def _remove_seen_items(self, engine: Engine) -> None:
        batch = cast(dict[str, torch.Tensor], engine.state.batch)
        output = cast(dict[str, torch.Tensor], engine.state.output)
        if (seen_items := batch.get("seen_items")) is not None:
            output["logits"].scatter_(dim=-1, index=seen_items, value=-1e13)
            output["logits"][:, 0] = -1e13

    def _seed_everything(self) -> None:
        import os
        import random

        random.seed(self._seed)
        os.environ["PYTHONHASHSEED"] = str(self._seed)
        set_seed(self._seed)


def seed_worker(*_) -> None:
    import random

    import numpy as np

    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)  # noqa: NPY002
    random.seed(worker_seed)
