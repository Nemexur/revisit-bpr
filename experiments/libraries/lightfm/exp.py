# pyright: reportOptionalMemberAccess=false

from typing import Any, Callable, cast
from copy import deepcopy
import json
from pathlib import Path

from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from hydra.utils import instantiate
from ignite.engine import Engine, EventEnum
import numpy as np
from rich import print_json
from scipy import sparse as sps
import torch
from tqdm import tqdm

from experiments import settings
from experiments.base import Experiment
from experiments.options import (
    attach_best_exp_saver,
    attach_checkpointer,
    attach_debug_handler,
    attach_log_epoch_metrics,
    attach_metrics,
    attach_output_saver,
    attach_params_watcher,
    attach_progress_bar,
    attach_user_metric_saver,
)
from experiments.trainer import ModelEvents, Trainer
from experiments.utils import flatten_config
from revisit_bpr.metrics import Metric


class LightFMExperiment(Experiment):
    def __init__(
        self,
        exp_config: dict[str, Any] | Callable[[], dict[str, Any]],
        dir: Path | None = None,
        n_checkpoints: int = 5,
        datasets_key: str = "datasets",
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
        self._datasets_key = datasets_key
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
        self._neg_gen = torch.Generator(device=self._accelerator.device).manual_seed(self._seed)
        self._model = self._accelerator.prepare(instantiate(self._config["model"]))
        datasets = {
            "train": [self._sparse_dataset(self._config[self._datasets_key]["train"])],
            "eval": self._accelerator.prepare_data_loader(
                instantiate(
                    self._config[self._datasets_key]["eval"],
                    generator=torch.Generator().manual_seed(self._seed),
                    worker_init_fn=seed_worker,
                ),
            ),
        }
        self.trainer = self._get_trainer(self._model)
        self._state = self.trainer.run(datasets)
        if self._save_logits and self._dir is not None:
            attach_output_saver(self.trainer, dir=self._dir)
            self.trainer.engines["eval"].run(datasets.get("eval"))
        if self._save_user_metrics and self._dir is not None:
            metrics_copy = deepcopy(self._metrics or {})
            for m in metrics_copy.values():
                m.reset()
            attach_user_metric_saver(self.trainer, dir=self._dir, metrics=metrics_copy)
            self.trainer.engines["eval"].run(datasets.get("eval"))
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
                total_limit=self._n_checkpoints,
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

    def _get_trainer(self, model: torch.nn.Module) -> Trainer:
        trainer = Trainer(
            model,
            optimizer=None,  # LightFM library is responsible for training
            accelerator=self._accelerator,
        )
        if self._skip_seen:
            trainer.add_event("eval", ModelEvents.FORWARD_COMPLETED, self._remove_seen_items)
        attach_params_watcher(trainer, accelerator=self._accelerator)
        if self._debug:
            attach_debug_handler(trainer, num_iters=2000)
        attach_metrics(trainer, self._accelerator, self._metrics)
        trainer.engines["train"].state.skip_metrics = True
        if self._accelerator.is_local_main_process:
            attach_progress_bar(trainer, metric_names={"eval": list(self._metrics)[:5]})
        attach_log_epoch_metrics(trainer, self._accelerator)
        if self._dir is not None:
            attach_checkpointer(
                trainer,
                self._accelerator,
                checkpoint_objects=self._metrics.values(),
            )
            if self._accelerator.is_local_main_process:
                attach_best_exp_saver(trainer, self._dir)
        for key, e in self._events.items():
            for event, handler in e:
                trainer.add_event(key, event, handler, accelerator=self._accelerator)
        return trainer

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

    def _sparse_dataset(self, path: str | Path) -> sps.coo_matrix:
        user_key, item_key = "user", "item"
        with Path(path).open("r", encoding="utf-8") as file:
            matrix = sps.dok_matrix(
                (self._config["num_users"], self._config["num_items"]), dtype=np.float32
            )
            for sample in tqdm(map(json.loads, file), desc="Building sparse dataset"):
                user_idx, item_idx = sample[user_key], sample[item_key]
                matrix[user_idx, item_idx] = 1.0
            return matrix.tocoo()


def seed_worker(*_) -> None:
    import random

    import numpy as np

    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)  # noqa: NPY002
    random.seed(worker_seed)
