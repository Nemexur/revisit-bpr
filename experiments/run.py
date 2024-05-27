from typing import Callable
from dataclasses import asdict
from pathlib import Path
import signal
import sys

from accelerate import Accelerator
from clearml import Task
import click
from hydra.utils import instantiate
from ignite.engine import Engine, Events
from jinja2 import StrictUndefined, Template
from loguru import logger
import optuna
from optuna.study import MaxTrialsCallback
from optuna.trial import TrialState
import torch
from wandb.util import generate_id
import yaml

from experiments.click_options import (
    State,
    clearml_option,
    debug_option,
    dir_option,
    extra_vars_option,
    name_option,
    pass_state,
    search_hp_options,
    seed_option,
    wandb_option,
)
from experiments.decorator import Distributed, Preemptible, Status
from experiments.hp import create_study
from experiments.settings import TRACKER_PROJECT
from experiments.utils import merge_configs, sample_params


@click.command(
    help="Run experiment.",
    context_settings={"help_option_names": ["-h", "--help"]},
)
@click.argument("config_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@name_option("exp")
@dir_option
@seed_option
@debug_option
@wandb_option
@clearml_option
@extra_vars_option
@search_hp_options(metric="ndcg@100")
@pass_state
def main(state: State, config_path: Path) -> None:
    def objective(trial: optuna.Trial) -> float:
        def trial_prunner_handler() -> Callable[[Engine, Accelerator], None]:
            def handler(engine: Engine, accelerator: Accelerator) -> None:
                if not accelerator.is_local_main_process:
                    return
                step = trial.user_attrs["exp_step"]
                trial.report(engine.state.metrics[state.search_hp.metric], step)
                if trial.should_prune():
                    raise optuna.TrialPruned(f"Trial was pruned at {step} epoch.")
                trial.set_user_attr("exp_step", step + 1)

            if "exp_step" not in trial.user_attrs:
                trial.set_user_attr("exp_step", 0)
            return handler

        if "exp_number" not in trial.user_attrs:
            trial.set_user_attr("exp_number", trial.number)
        trial_name = f"{state.exp_name}-optuna-{trial.user_attrs['exp_number']}"
        trial_config = merge_configs(config, sample_params(trial, config.get("optuna", {})))
        trial_dir = None
        if state.exp_dir is not None:
            trial_dir = state.exp_dir / f"trial-{trial.user_attrs['exp_number']}"
            trial_dir.mkdir(exist_ok=True)
        if trial_dir is not None:
            with (trial_dir / "config.yaml").open("w", encoding="utf-8") as file:
                yaml.safe_dump(trial_config, file, indent=2)
        tracker_params = {}
        if state.use_wandb:
            if "exp_id" not in trial.user_attrs:
                trial.set_user_attr("exp_id", generate_id())
            tracker_params["wandb"] = {
                "name": trial_name,
                "group": state.exp_name,
                "job_type": "optuna",
                "id": trial.user_attrs["exp_id"],
                "resume": True,
            }
        if state.use_clearml:
            continue_task_id = False
            if (t := Task.get_task(project_name=TRACKER_PROJECT, task_name=trial_name)) is not None:
                continue_task_id = t.task_id
            tracker_params["clearml"] = {
                "task_name": trial_name,
                "tags": [state.exp_name, "optuna"],
                "reuse_last_task_id": continue_task_id,
                "continue_last_task": 0 if continue_task_id else False,
                "auto_connect_frameworks": False,
            }
        exp = Preemptible(
            Distributed(
                instantiate(
                    trial_config.pop("experiment"),
                    exp_config=lambda: trial_config,
                    dir=trial_dir,
                    datasets_key="optuna_datasets",
                    debug=state.debug,
                    seed=state.seed,
                    trackers_params=tracker_params,
                    events=(
                        {"eval": [(Events.EPOCH_COMPLETED, trial_prunner_handler())]}
                        if state.search_hp.prune
                        else {}
                    ),
                ),
                launcher=(
                    instantiate(l)
                    if (l := trial_config.pop("launcher", None)) is not None
                    else None
                ),
            )
        )
        match status := exp.run():
            case Status.PREEMPTED:
                sys.exit(status.value)
            case Status.EXCEPTION:
                exc, trace = status.value
                if isinstance(exc, optuna.TrialPruned):
                    raise exc
                print(trace)
                logger.error(f"exception: {exc}")
                sys.exit(signal.SIGINT)
        exp.clean()
        for metric, value in exp.metrics.items():
            if metric.startswith("_"):
                continue
            trial.set_user_attr(metric, value.item() if isinstance(value, torch.Tensor) else value)
        return exp.metrics[state.search_hp.metric]

    with config_path.open("r", encoding="utf-8") as file:
        tmpl = Template(file.read(), undefined=StrictUndefined, autoescape=True)
        config = yaml.safe_load(tmpl.render(asdict(state) | (state.extra_vars or {})))
    if state.exp_dir is not None:
        state.exp_dir.mkdir(exist_ok=True)
    if not state.search_hp.run:
        if state.exp_dir is not None:
            with (state.exp_dir / "config.yaml").open("w", encoding="utf-8") as file:
                yaml.safe_dump(config, file, indent=2)
        tracker_params = {}
        if state.use_wandb:
            tracker_params["wandb"] = {"name": state.exp_name, "resume": True}
        if state.use_clearml:
            continue_task_id = False
            if (
                t := Task.get_task(project_name=TRACKER_PROJECT, task_name=state.exp_name)
            ) is not None:
                continue_task_id = t.task_id
            tracker_params["clearml"] = {
                "task_name": state.exp_name,
                "reuse_last_task_id": continue_task_id,
                "continue_last_task": 0 if continue_task_id else False,
                "auto_connect_frameworks": False,
            }
        exp = Preemptible(
            Distributed(
                instantiate(
                    config.pop("experiment"),
                    exp_config=lambda: config,
                    dir=state.exp_dir,
                    debug=state.debug,
                    seed=state.seed,
                    trackers_params=tracker_params,
                ),
                launcher=(
                    instantiate(l) if (l := config.pop("launcher", None)) is not None else None
                ),
            )
        )
        match status := exp.run():
            case Status.PREEMPTED:
                sys.exit(status.value)
            case Status.EXCEPTION:
                exc, _ = status.value
                raise exc
        return

    study = create_study(
        state.exp_name,
        dir=state.exp_dir,
        storage_url=state.search_hp.storage,
        seed=state.search_hp.seed,
    )
    if len(study.get_trials(deepcopy=True, states=[TrialState.COMPLETE])) < state.search_hp.trials:
        study.optimize(
            objective,
            gc_after_trial=True,
            callbacks=[MaxTrialsCallback(state.search_hp.trials, states=[TrialState.COMPLETE])],
        )
    if not state.search_hp.train_best:
        return
    logger.info("train best model")
    best_name = f"{state.exp_name}-optuna-best"
    best_config = merge_configs(config, study.best_params)
    best_dir = None
    if state.exp_dir is not None:
        best_dir = state.exp_dir / "trial-best"
        best_dir.mkdir(exist_ok=True)
    if best_dir is not None:
        with (best_dir / "config.yaml").open("w", encoding="utf-8") as file:
            yaml.safe_dump(best_config, file, indent=2)
    tracker_params = {}
    if state.use_wandb:
        tracker_params["wandb"] = {
            "name": best_name,
            "group": state.exp_name,
            "job_type": "best",
            "resume": True,
        }
    if state.use_clearml:
        continue_task_id = False
        if (t := Task.get_task(project_name=TRACKER_PROJECT, task_name=best_name)) is not None:
            continue_task_id = t.task_id
        tracker_params["clearml"] = {
            "task_name": best_name,
            "tags": [state.exp_name, "best"],
            "reuse_last_task_id": continue_task_id,
            "continue_last_task": 0 if continue_task_id else False,
            "auto_connect_frameworks": False,
        }
    exp = Preemptible(
        Distributed(
            instantiate(
                best_config.pop("experiment"),
                exp_config=lambda: best_config,
                dir=best_dir,
                debug=state.debug,
                seed=state.seed,
                trackers_params=tracker_params,
            ),
            launcher=(
                instantiate(l) if (l := best_config.pop("launcher", None)) is not None else None
            ),
        )
    )
    match status := exp.run():
        case Status.PREEMPTED:
            sys.exit(status.value)
        case Status.EXCEPTION:
            exc, _ = status.value
            raise exc
    exp.clean()
    for metric, value in exp.metrics.items():
        if metric.startswith("_"):
            continue
        study.set_user_attr(metric, value.item() if isinstance(value, torch.Tensor) else value)


if __name__ == "__main__":
    main()
