# pyright: reportOptionalMemberAccess=false

from typing import Any, Callable, Literal, cast
from copy import deepcopy
import json
import math
from pathlib import Path
import re
import shutil

from accelerate import Accelerator
from accelerate.optimizer import AcceleratedOptimizer
from accelerate.utils import ProjectConfiguration, set_seed
from einops import repeat
from hydra.utils import instantiate
from ignite.engine import Engine, EventEnum, Events
from loguru import logger
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
    attach_early_stopping,
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
from src.metrics import Metric
from src.models import BPR


class BPRExperiment(Experiment):
    def __init__(
        self,
        exp_config: dict[str, Any] | Callable[[], dict[str, Any]],
        dir: Path | None = None,
        n_checkpoints: int = 2,
        mixed_precision: str | None = None,
        datasets_key: str = "datasets",
        metrics: dict[str, Metric] | None = None,
        trackers_params: dict[str, Any] | None = None,
        events: dict[str, list[tuple[EventEnum, Callable]]] | None = None,
        seed: int = 13,
        debug: bool = False,
        skip_seen: bool = True,
        save_logits: bool = False,
        save_user_metrics: bool = False,
        log_momentum: bool = False,
        early_stopping_metric: str | None = None,
        early_stopping_patience: int = 200,
        early_stopping_direction: Literal["min", "max"] = "max",
        neg_sampling_alpha: float = 0.0,
        adaptive_sampling_prob: float | None = None,
    ) -> None:
        self._config = exp_config if isinstance(exp_config, dict) else exp_config()
        self._dir = dir
        self._n_checkpoints = n_checkpoints
        self._mixed_precision = mixed_precision
        self._seed = seed
        self._debug = debug
        self._skip_seen = skip_seen
        self._save_logits = save_logits
        self._save_user_metrics = save_user_metrics
        self._log_momentum = log_momentum
        self._early_stopping_metric = early_stopping_metric
        self._early_stopping_patience = early_stopping_patience
        self._early_stopping_direction = early_stopping_direction
        self._datasets_key = datasets_key
        self._metrics = metrics or {}
        self._trackers_params = trackers_params or {}
        self._events = events or {}
        self._adaptive_sampling_prob = adaptive_sampling_prob
        self._item_counts = torch.ones(self._config["num_items"], dtype=torch.float32)
        if (path := self._config[datasets_key].pop("item_counts", None)) is not None:
            with open(path, "r", encoding="utf-8") as file:
                for item_count in map(json.loads, file):
                    self._item_counts[item_count["item"]] = (
                        float(item_count["count"]) ** neg_sampling_alpha
                    )

    @property
    def metrics(self) -> dict[str, Any]:
        return self._state.metrics

    def run(self) -> Any:
        self._accelerator = self._get_accelerator()
        if self._accelerator.is_local_main_process:
            print_json(data=self._config)
        self._item_counts = self._item_counts.to(self._accelerator.device)
        self._model = self._accelerator.prepare(instantiate(self._config["model"]))
        self._optimizer = self._accelerator.prepare(
            instantiate(self._config["optimizer"])(self._model.parameters())
        )
        max_iters = {
            k: d.pop("max_iters", None) for k, d in self._config[self._datasets_key].items()
        }
        self._datasets = {
            key: self._accelerator.prepare_data_loader(
                instantiate(
                    loader,
                    generator=torch.Generator().manual_seed(self._seed),
                    worker_init_fn=seed_worker,
                ),
            )
            for key, loader in self._config[self._datasets_key].items()
        }
        for key, loader in self._datasets.items():
            if hasattr(loader.dataset, "collate_fn"):
                self._datasets[key].collate_fn = loader.dataset.collate_fn
        self.trainer = self._get_trainer(self._model, self._optimizer, self._datasets)
        self._load_checkpoint_if_needed()
        self._neg_gen = (
            torch.Generator(device=self._accelerator.device)
            # In case of preemptible tasks neg generator might sample the same data
            .manual_seed(self._seed + self.trainer.engines["train"].state.iteration)
        )
        if self._adaptive_sampling_prob is not None and isinstance(
            self._adaptive_sampling_prob, float
        ):
            self._update_adaptive_stats()
        self._state = self.trainer.run(
            self._datasets, max_iters=max_iters, epochs=self._config["epochs"]
        )
        if self._save_user_metrics and self._dir is not None:
            metrics_copy = deepcopy(self._metrics or {})
            for m in metrics_copy.values():
                m.reset()
            attach_user_metric_saver(self.trainer, dir=self._dir, metrics=metrics_copy)
            self.trainer.engines["eval"].run(
                self._datasets.get("eval"), epoch_length=max_iters.get("eval")
            )
        if self._save_logits and self._dir is not None:
            attach_output_saver(self.trainer, dir=self._dir)
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
            mixed_precision=self._mixed_precision,
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

    def _get_trainer(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        datasets: dict[str, DataLoader],
    ) -> Trainer:
        custom_engines = self._config.get("custom_engines", {})
        trainer = Trainer(
            model,
            optimizer=optimizer,
            accelerator=self._accelerator,
            custom_engines=custom_engines,
        )
        if self._adaptive_sampling_prob is not None and isinstance(
            self._adaptive_sampling_prob, float
        ):
            trainer.add_event(
                "train",
                Events.GET_BATCH_COMPLETED(
                    every=int(
                        self._config["num_items"]
                        * math.log(self._config["num_items"])
                        / self._datasets["train"].total_batch_size
                    )
                ),
                self._update_adaptive_stats,
            )
        trainer.add_event("train", Events.GET_BATCH_COMPLETED, self._train_batch)
        if self._skip_seen:
            trainer.add_event("eval", ModelEvents.FORWARD_COMPLETED, self._remove_seen_items)
        early_stopping = None
        if self._early_stopping_metric is not None:
            early_stopping = attach_early_stopping(
                trainer,
                metric_name=self._early_stopping_metric,
                patience=self._early_stopping_patience,
                direction=self._early_stopping_direction,
            )
        attach_preemptible(trainer, self._accelerator)
        attach_params_watcher(trainer, accelerator=self._accelerator)
        if self._debug:
            attach_debug_handler(trainer, num_iters=2000)
        attach_metrics(trainer, self._accelerator, self._metrics)
        if self._accelerator.is_local_main_process:
            attach_progress_bar(
                trainer, metric_names={"train": ["loss"], "eval": list(self._metrics)[:5]}
            )
        attach_log_epoch_metrics(trainer, self._accelerator)
        if self._dir is not None:
            attach_checkpointer(
                trainer,
                self._accelerator,
                early_stopping=early_stopping,
                checkpoint_objects=self._metrics.values(),
            )
            attach_checkpoint_loader(trainer, self._accelerator, datasets)
            if self._accelerator.is_local_main_process:
                attach_best_exp_saver(trainer, self._dir)
        optim = cast(AcceleratedOptimizer, trainer.optimizer)
        if self._log_momentum and isinstance(optim.optimizer, torch.optim.Adam):
            attach_adam_momentum_saver(trainer, accelerator=self._accelerator)
        for key, e in self._events.items():
            for event, handler in e:
                trainer.add_event(key, event, handler, accelerator=self._accelerator)
        trainer.add_event("train", Events.EPOCH_STARTED, self._reset_metrics)
        trainer.add_event("train", Events.ITERATION_COMPLETED, self._update_metrics)
        return trainer

    def _load_checkpoint_if_needed(self) -> None:
        if self._dir is None:
            return
        checkpoints = self._dir / settings.CHECKPOINTS_DIR
        if not checkpoints.exists():
            return
        sorted_checkpoints = sorted(
            (
                (dir, int(re.findall(r"[\/]?([0-9]+)(?=[^\/]*$)", str(dir))[0]))
                for dir in checkpoints.iterdir()
            ),
            key=lambda d: d[1],
        )
        while checkpoint := sorted_checkpoints.pop():
            dir, num = checkpoint
            try:
                self._accelerator.load_state()
            except Exception as exc:
                logger.error(f"exception: {exc}")
                shutil.rmtree(dir)
                continue
            # Accelerator requires manually configured save_iteration
            self._accelerator.project_configuration.iteration = num + 1
            break

    def _seed_everything(self) -> None:
        import os
        import random

        random.seed(self._seed)
        os.environ["PYTHONHASHSEED"] = str(self._seed)
        set_seed(self._seed)

    def _sampling_weights(self, seen_items: torch.Tensor) -> torch.Tensor:
        weights = repeat(
            self._item_counts, "items -> batch items", batch=seen_items.size(0)
        ).scatter(dim=-1, index=seen_items, value=0.0)
        weights[:, 0] = 0.0  # Discard padding
        weights *= weights.sum(dim=-1, keepdim=True).reciprocal()
        return weights

    def _static_sampling(self, batch: dict[str, torch.Tensor], num: int = 1) -> torch.Tensor:
        return torch.multinomial(
            self._sampling_weights(batch["seen_items"]), num_samples=num, generator=self._neg_gen
        )

    def _adaptive_sampling(self, batch: dict[str, torch.Tensor], num: int = 1) -> torch.Tensor:
        user, seen_items = batch["user"], batch["seen_items"]
        model = self._model
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model = model.module
        model = cast(BPR, model)
        features = model.logits_model.get_features()
        # num_notseen_items ~ (batch size, 1)
        num_notseen_items = self._sampling_weights(seen_items).gt(0).sum(dim=-1, keepdim=True)
        # factor, rank ~ (batch size, num samples)
        factor = torch.multinomial(
            features["user"].abs()[user] * self._factor_std,
            num_samples=num,
            generator=self._neg_gen,
        )
        # In this case rank is not greater than the number of not seen items
        rank = (
            torch.empty_like(factor)
            .geometric_(self._adaptive_sampling_prob, generator=self._neg_gen)
            .clamp_(max=num_notseen_items)
        )
        # Only consider not seen items
        rank = torch.where(
            features["user"][user].gather(dim=-1, index=factor).gt(0),
            rank - 1,
            num_notseen_items - rank,
        )
        if (rank < 0).sum() > 0 or (rank > num_notseen_items.sub(1)).sum() > 0:
            print(
                "Detected out of bounds. Force clamp. "
                f"Lower 0: {(rank < 0).sum().item()}. "
                f"More than not seen: {(rank > num_notseen_items.sub(1)).sum().item()}"
            )
            rank = rank.clamp(min=torch.zeros_like(num_notseen_items), max=num_notseen_items - 1)
        # Just in case manually discard padding item
        seen_items = repeat(
            torch.hstack((seen_items, torch.zeros_like(rank))),
            "batch items -> batch num items",
            num=num,
        )
        return (
            torch.argsort(
                -self._factor_to_items[factor].scatter(-1, index=seen_items, value=-1e13),
                dim=-1,
            )
            .gather(dim=-1, index=rank.unsqueeze(-1))
            .squeeze(-1)
        )

    @torch.no_grad()
    def _update_adaptive_stats(self) -> None:
        model = self._model
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model = model.module
        model = cast(BPR, model)
        features = model.logits_model.get_features()
        # item_by_factor ~ (factors, num items)
        self._factor_to_items = torch.einsum("if->fi", features["item"]).detach().clone()
        # factor_var ~ (1, factors)
        self._factor_std = features["item"][1:].detach().std(dim=0, keepdim=True)

    @torch.no_grad()
    def _train_batch(self, engine: Engine) -> None:
        batch = cast(dict[str, torch.Tensor], engine.state.batch)
        # batch.item, batch.neg ~ (batch size, num items)
        if batch["item"].dim() < 2:
            batch["item"].unsqueeze_(-1)
        batch["neg"] = (
            self._adaptive_sampling(batch, num=batch["item"].size(-1))
            if self._adaptive_sampling_prob is not None
            and isinstance(self._adaptive_sampling_prob, float)
            else self._static_sampling(batch, num=batch["item"].size(-1))
        )

    def _remove_seen_items(self, engine: Engine) -> None:
        batch = cast(dict[str, torch.Tensor], engine.state.batch)
        output = cast(dict[str, torch.Tensor], engine.state.output)
        if (seen_items := batch.get("seen_items")) is not None:
            output["logits"].scatter_(dim=-1, index=seen_items, value=-1e13)
            output["logits"][:, 0] = -1e13

    def _reset_metrics(self, engine: Engine) -> None:
        state = engine.state
        if state.was_interrupted:
            return
        for m in ("bpr_loss", "l2_reg", "logits_diff", "bias_diff", "m_t_sum"):
            state.metrics[f"_{m}"] = torch.tensor(0.0, device=self._accelerator.device)

    @torch.no_grad()
    def _update_metrics(self, engine: Engine) -> None:
        state = engine.state
        out = cast(dict[str, torch.Tensor], state.output)
        batch = cast(dict[str, torch.Tensor], state.batch)
        for m in ("bpr_loss", "l2_reg"):
            state.metrics[f"_{m}"] += out[m]
            state.metrics[m] = state.metrics[f"_{m}"] / state.epoch_iteration
        state.metrics["_logits_diff"] += out["logits"].abs().mean()
        state.metrics["logits_diff"] = state.metrics["_logits_diff"] / state.epoch_iteration
        # Log diffs
        model = self._model
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model = model.module
        model = cast(BPR, model)
        features = model.logits_model.get_features()
        if features.get("item_bias") is None and features.get("bias") is None:
            return
        item_biases = ib if (ib := features.get("item_bias")) is not None else features["bias"]
        state.metrics["_bias_diff"] += (
            (item_biases[batch["item"]] - item_biases[batch["neg"]]).abs().mean()
        )
        state.metrics["bias_diff"] = state.metrics["_bias_diff"] / state.epoch_iteration


def attach_adam_momentum_saver(
    trainer: Trainer, accelerator: Accelerator, log_freq: int = 1000
) -> None:
    def iter_handler(engine: Engine) -> None:
        state, optim = engine.state, trainer.optimizer
        for group in optim.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                if (m_t := optim.state[p].get("exp_avg")) is not None:
                    state.metrics["_m_t_sum"] += m_t.abs().mean()

    def log_handler(engine: Engine) -> None:
        state = engine.state
        accelerator.log(
            {"m_t_sum_epoch/train": state.metrics["_m_t_sum"].item() / log_freq},
            step=state.iteration // log_freq,
        )
        state.metrics["_m_t_sum"] = torch.tensor(0.0, device=accelerator.device)

    trainer.add_event("train", ModelEvents.OPTIMIZER_COMPLETED, iter_handler)
    trainer.add_event("train", Events.ITERATION_COMPLETED(every=log_freq), log_handler)


def seed_worker(*_) -> None:
    import random

    import numpy as np

    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)  # noqa: NPY002
    random.seed(worker_seed)
