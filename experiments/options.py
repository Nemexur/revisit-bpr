# pyright: reportOptionalSubscript=false, reportOptionalMemberAccess=false

from typing import Callable, Iterable, Literal, Type, TypeVar
import contextlib
from functools import partial
import json
from pathlib import Path
import shutil
import tarfile
import time

from accelerate import Accelerator
from accelerate.tracking import ClearMLTracker, WandBTracker
from accelerate.utils import send_to_device
from clearml.binding.frameworks.tensorflow_bind import WeightsGradientHistHelper
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.engine import Engine, Events
from ignite.handlers import EarlyStopping
from loguru import logger
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from experiments import settings
from experiments.trainer import ModelEvents, Trainer
from src.metrics import MaskedMetric, Metric

T = TypeVar("T")


def attach_metrics(
    trainer: Trainer, accelerator: Accelerator, metrics: dict[str, Metric] | None = None
) -> None:
    def reset_handler(engine: Engine) -> None:
        state = engine.state
        if state.was_interrupted:
            return
        for m in metrics.values():
            m.reset()

    @torch.no_grad()
    def update_handler(engine: Engine) -> None:
        state = engine.state
        if state.skip_metrics or "target" not in state.batch:
            return
        for key, m in metrics.items():
            kwargs = {}
            if isinstance(m, MaskedMetric):
                kwargs["mask"] = state.batch.get("mask")
            m(state.output["logits"], state.batch["target"], **kwargs)
            state.metrics[key] = m.get_metric()

    def reduce_metrics_handler() -> Callable[[Engine, Exception | None], None]:
        def handler(engine: Engine, exc: Exception | None = None) -> None:
            if not handler.is_reduced:
                engine.state.metrics = {
                    k: accelerator.reduce(v, reduction="mean")
                    for k, v in engine.state.metrics.items()
                }
            handler.is_reduced = True
            if exc is None:
                return
            raise exc

        def reset() -> None:
            handler.is_reduced = False

        handler.is_reduced = False
        handler.reset = reset
        return handler

    if metrics is None:
        return
    for e in trainer.engines:
        trainer.engines[e].state.skip_metrics = False
        trainer.engines[e].state_dict_user_keys.append("metrics")
        reduce_metrics_h = reduce_metrics_handler()
        trainer.add_event(e, Events.EPOCH_STARTED, partial(lambda h: h.reset(), h=reduce_metrics_h))
        trainer.add_event(e, Events.EPOCH_STARTED, reset_handler)
        trainer.add_event(e, Events.ITERATION_COMPLETED, update_handler)
        trainer.add_event(
            e,
            Events.EPOCH_COMPLETED | Events.INTERRUPT | Events.EXCEPTION_RAISED,
            reduce_metrics_h,
        )


def attach_checkpointer(
    trainer: Trainer,
    accelerator: Accelerator,
    early_stopping: EarlyStopping | None = None,
    checkpoint_objects: Iterable[object] | None = None,
) -> None:
    def save_handler(engine: Engine) -> None:
        engine.state.save_location = _accelerator_save(accelerator)
        if accelerator.is_local_main_process:
            logger.info(f"checkpointer: saved checkpoint in {engine.state.save_location}")

    def save_best_handler(engine: Engine) -> None:
        if early_stopping is not None and early_stopping.counter != 0:
            return
        save_dir = Path(accelerator.project_dir) / settings.BEST_ITERATION_PATH
        if accelerator.is_local_main_process:
            shutil.copytree(engine.state.save_location, save_dir, dirs_exist_ok=True)

    if early_stopping is not None:
        accelerator.register_for_checkpointing(early_stopping)
    for e in trainer.engines.values():
        accelerator.register_for_checkpointing(e)
    for m in checkpoint_objects or []:
        accelerator.register_for_checkpointing(m)
    trainer.add_event("eval", Events.COMPLETED, save_handler)
    trainer.add_event("eval", Events.COMPLETED, save_best_handler)


def attach_checkpoint_loader(
    trainer: Trainer, accelerator: Accelerator, datasets: dict[str, DataLoader]
) -> None:
    def skip_batches_handler(engine: Engine) -> None:
        state = engine.state
        if not state.was_interrupted:
            return
        iterations = state.iteration
        if state.epoch_length is not None:
            iterations %= state.epoch_length
        logger.info("continue {} from iteration {}", state.name, iterations)
        state.dataloader = accelerator.skip_first_batches(datasets[state.name], iterations)

    def ensure_device_handler(engine: Engine) -> None:
        state = engine.state
        for m, v in state.metrics.items():
            if not isinstance(v, torch.Tensor):
                continue
            state.metrics[m] = v.to(accelerator.device)

    def reset_interrupted_handler(engine: Engine) -> None:
        state = engine.state
        if not state.was_interrupted:
            return
        engine.set_data(datasets[state.name])
        state.was_interrupted = False

    for e in trainer.engines:
        trainer.add_event(e, Events.STARTED, skip_batches_handler)
        trainer.add_event(e, Events.STARTED, ensure_device_handler)
        trainer.add_event(e, Events.EPOCH_COMPLETED, reset_interrupted_handler)


def attach_progress_bar(
    trainer: Trainer, metric_names: dict[str, str | list[str]] | None = None
) -> None:
    metric_names = metric_names or {}
    for key, e in trainer.engines.items():
        pbar = ProgressBar(
            persist=True,
            bar_format=(
                "{desc} [{n_fmt}/{total_fmt}] "
                "{percentage:3.0f}%|{bar}|{postfix} "
                "({elapsed}<{remaining}, {rate_fmt})"
            ),
            desc="\033[33m" + key.capitalize() + "\033[00m",
        )
        pbar.attach(e, metric_names=metric_names.get(key))


def attach_early_stopping(
    trainer: Trainer,
    metric_name: str,
    patience: int = 10,
    direction: Literal["min", "max"] = "max",
    min_delta: float = 1e-4,
) -> EarlyStopping:
    def score_function(engine: Engine) -> float:
        metric = engine.state.metrics[metric_name]
        sign = directions[direction]
        return sign * metric

    directions = {"min": -1.0, "max": 1.0}
    if direction not in directions:
        raise ValueError(f"Direction is not valid (expected: {list(directions)}, got: {direction})")
    handler = EarlyStopping(
        patience, score_function, trainer=trainer.engines["train"], min_delta=min_delta
    )
    trainer.add_event("eval", Events.COMPLETED, handler)
    return handler


def attach_preemptible(trainer: Trainer, accelerator: Accelerator) -> None:
    def handler(engine: Engine) -> None:
        for p in trainer.model.parameters():
            p.requires_grad = True
        engine.state.was_interrupted = True
        with contextlib.suppress(Exception):  # Ignore all excpetions
            if (wb := _accelerator_tracker(WandBTracker, accelerator)) is not None:
                wb.tracker.mark_preempting()
            if (clearml := _accelerator_tracker(ClearMLTracker, accelerator)) is not None:
                clearml.tracker.mark_stopped()
                clearml.tracker._at_exit_called = True  # noqa: SLF001
        if accelerator.project_dir is None:
            return
        checkpoints = Path(accelerator.project_dir) / settings.CHECKPOINTS_DIR
        if (
            last_checkpoint := checkpoints / f"checkpoint_{accelerator.save_iteration - 1}"
        ).exists():
            seconds_since_last_save = int(time.time() - last_checkpoint.stat().st_atime)
            if seconds_since_last_save < 10:  # less than 10 sec since last save
                logger.info("preemptible: skipped saving checkpoint")
                return
        save_location = _accelerator_save(accelerator)
        if accelerator.is_local_main_process:
            logger.info(f"preemptible: saved checkpoint in {save_location}")

    def handler_with_exception(engine: Engine, exc: Exception) -> None:
        handler(engine)
        raise exc

    for e in trainer.engines:
        trainer.add_event(e, Events.INTERRUPT, handler)
        trainer.add_event(e, Events.EXCEPTION_RAISED, handler_with_exception)


def attach_params_watcher(
    trainer: Trainer, accelerator: Accelerator, report_freq: int = 1000
) -> None:
    def params_hist_handler(engine: Engine) -> None:
        global_step = engine.state.get_event_attrib_value(Events.ITERATION_COMPLETED)
        for name, p in weights.items():
            if torch.isnan(p.data).sum() > 0:
                logger.info("params hist: found nan, skipping")
                continue
            title_name, _, series_name = name.partition(".")
            hist_helper.add_histogram(
                title=f"parameters/{title_name}",
                series=series_name,
                step=global_step,
                hist_data=p.data.cpu().numpy(),
            )

    def grads_hist_handler(engine: Engine) -> None:
        global_step = engine.state.get_event_attrib_value(ModelEvents.OPTIMIZER_COMPLETED)
        for name, p in weights.items():
            if p.grad is None:
                continue
            if torch.isnan(p.grad).sum() > 0:
                logger.info("gradient hist: found nan, skipping")
                continue
            title_name, _, series_name = name.partition(".")
            hist_helper.add_histogram(
                title=f"gradients/{title_name}",
                series=series_name,
                step=global_step,
                hist_data=p.grad.cpu().numpy(),
            )

    if (wb := _accelerator_tracker(WandBTracker, accelerator)) is not None:
        wb.tracker.watch(trainer.model, log="all", log_freq=report_freq)
    clearml = _accelerator_tracker(ClearMLTracker, accelerator)
    if clearml is None:
        return
    weights = dict(trainer.model.named_parameters())
    hist_helper = WeightsGradientHistHelper(logger=clearml.tracker.get_logger(), report_freq=1)
    trainer.add_event("train", Events.ITERATION_COMPLETED(every=report_freq), params_hist_handler)
    trainer.add_event(
        "train", ModelEvents.OPTIMIZER_COMPLETED(every=report_freq), grads_hist_handler
    )


def attach_debug_handler(trainer: Trainer, num_iters: int = 100) -> None:
    def handler(engine: Engine) -> None:
        if engine.state.epoch_iteration < num_iters:
            return
        engine.terminate_epoch()

    for e in trainer.engines:
        trainer.add_event(e, Events.ITERATION_COMPLETED, handler)


def attach_log_epoch_metrics(trainer: Trainer, accelerator: Accelerator) -> None:
    def get_step() -> int:
        return trainer.engines["train"].state.epoch

    def handler(engine: Engine) -> None:
        metrics = {
            k: v.item() if isinstance(v, torch.Tensor) else v
            for k, v in engine.state.metrics.items()
            if not k.startswith("_")
        }
        if not accelerator.is_local_main_process:
            return
        if len(metrics) == 0:
            return
        run_type = engine.state.name
        logger.info(run_type.capitalize())
        max_length = max(len(x) for x in metrics)
        for metric in sorted(metrics, key=lambda x: (len(x), x)):
            metric_value = metrics.get(metric)
            if isinstance(metric_value, (float, int)):
                logger.info(f"{metric.ljust(max_length)} | {metric_value:.4f}")
        accelerator.log({f"{k}_epoch/{run_type}": v for k, v in metrics.items()}, step=get_step())

    for e in trainer.engines:
        trainer.add_event(e, Events.EPOCH_COMPLETED, handler)


def attach_best_exp_saver(trainer: Trainer, dir: Path) -> None:
    def handler() -> None:
        if not (dir / settings.BEST_ITERATION_PATH).exists():
            return
        exp_archive = dir / "experiment.tar.gz"
        with tarfile.open(exp_archive, "w:gz") as archive:
            archive.add(dir / settings.BEST_ITERATION_PATH, arcname=settings.BEST_ITERATION_PATH)

    if not dir.is_dir():
        logger.error(f"attach_best_exp_saver works with directories only (path={dir})")
        return
    trainer.add_event("train", Events.COMPLETED, handler)


def attach_output_saver(trainer: Trainer, dir: Path, state_attr: str = "result") -> None:
    def iter_handler(engine: Engine) -> None:
        getattr(engine.state, state_attr).append(
            {
                "batch": send_to_device(engine.state.batch, device="cpu"),
                "output": send_to_device(engine.state.output, device="cpu"),
            }
        )

    def save_handler(engine: Engine) -> None:
        path = dir / "preds.jsonl"
        with path.open("a", encoding="utf-8") as file:
            for sample in tqdm(getattr(engine.state, state_attr), desc="Saving output"):
                for user, logits in zip(
                    sample["batch"]["user"], sample["output"]["logits"], strict=True
                ):
                    file.write(
                        json.dumps(
                            {
                                "user": int(user.item()),
                                "items": {
                                    str(item_id): logit.item()
                                    for item_id, logit in enumerate(logits)
                                },
                            },
                            ensure_ascii=False,
                        )
                    )
                    file.write("\n")

    setattr(trainer.engines["eval"].state, state_attr, [])
    trainer.add_event("eval", Events.ITERATION_COMPLETED, iter_handler)
    trainer.add_event("eval", Events.COMPLETED, save_handler)


def attach_user_metric_saver(
    trainer: Trainer,
    dir: Path,
    metrics: dict[str, Metric] | None = None,
    state_attr: str = "user_metrics",
) -> None:
    def iter_handler(engine: Engine) -> None:
        state = engine.state
        batch_metrics = {}
        for key, m in (metrics or {}).items():
            kwargs = {}
            if isinstance(m, MaskedMetric):
                kwargs["mask"] = state.batch.get("mask")
            scores = send_to_device(
                m.compute(state.output["logits"], state.batch["target"], **kwargs), device="cpu"
            )
            batch_metrics[key] = scores
        getattr(state, state_attr).append(
            {"batch": send_to_device(engine.state.batch, device="cpu"), **batch_metrics}
        )

    def save_handler(engine: Engine) -> None:
        path = dir / "user-metrics.jsonl"
        with path.open("a", encoding="utf-8") as file:
            for sample in tqdm(getattr(engine.state, state_attr), desc="Saving output"):
                for idx, user in enumerate(sample["batch"]["user"]):
                    save_sample = {"user": int(user.item())}
                    for m in metrics or {}:
                        save_sample[m] = float(sample[m][idx].item())
                    file.write(json.dumps(save_sample, ensure_ascii=True))
                    file.write("\n")

    setattr(trainer.engines["eval"].state, state_attr, [])
    trainer.add_event("eval", Events.ITERATION_COMPLETED, iter_handler)
    trainer.add_event("eval", Events.COMPLETED, save_handler)


def _accelerator_save(accelerator: Accelerator) -> str:
    # If current checkpoint exists for some reason we save the next one
    # It might happen on a different thread because of forking.
    # For instance, multiple workers for DataLoader might trap a signal
    # and initiate an incomplete save_state.
    checkpoints = Path(accelerator.project_dir) / settings.CHECKPOINTS_DIR
    if (checkpoints / f"checkpoint_{accelerator.save_iteration}").exists():
        logger.error(f"accelerator: checkpoint_{accelerator.save_iteration} already exists")
        accelerator.project_configuration.iteration += 1
    return accelerator.save_state()


def _accelerator_tracker(tracker_type: Type[T], accelerator: Accelerator) -> T | None:
    for t in accelerator.trackers:
        if isinstance(t, tracker_type):
            return t
    return None
