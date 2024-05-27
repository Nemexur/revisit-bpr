from dataclasses import asdict
from pathlib import Path
import sys

from clearml import Task
import click
from hydra.utils import instantiate
from jinja2 import StrictUndefined, Template
import torch
import yaml

from experiments.click_options import (
    State,
    clearml_option,
    dir_option,
    extra_vars_option,
    name_option,
    pass_state,
    search_hp_seed_option,
    search_hp_storage_option,
    seed_option,
    wandb_option,
)
from experiments.decorator import Distributed, Preemptible, Status
from experiments.hp import create_study
from experiments.settings import TRACKER_PROJECT
from experiments.utils import merge_configs


@click.command(
    help="Run infer.",
    context_settings={"help_option_names": ["-h", "--help"]},
)
@click.argument("config_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@name_option("exp")
@dir_option
@seed_option
@clearml_option
@wandb_option
@extra_vars_option
@search_hp_seed_option
@search_hp_storage_option
@pass_state
def main(state: State, config_path: Path) -> None:
    with config_path.open("r", encoding="utf-8") as file:
        tmpl = Template(file.read(), undefined=StrictUndefined, autoescape=True)
        config = yaml.safe_load(tmpl.render(asdict(state) | (state.extra_vars or {})))
    if state.exp_dir is not None:
        state.exp_dir.mkdir(exist_ok=True)
    study = create_study(
        state.exp_name,
        dir=state.exp_dir,
        storage_url=state.search_hp.storage,
        seed=state.search_hp.seed,
    )
    infer_name = f"{state.exp_name}-best-infer"
    infer_config = merge_configs(config, study.best_params)
    infer_dir = None
    if state.exp_dir is not None:
        infer_dir = state.exp_dir / "infer"
        infer_dir.mkdir(exist_ok=True)
    if infer_dir is not None:
        with (infer_dir / "config.yaml").open("w", encoding="utf-8") as file:
            yaml.safe_dump(infer_config, file, indent=2)
    tracker_params = {}
    if state.use_wandb:
        tracker_params["wandb"] = {
            "name": infer_name,
            "group": state.exp_name,
            "job_type": "infer",
            "resume": True,
        }
    if state.use_clearml:
        continue_task_id = False
        if (t := Task.get_task(project_name=TRACKER_PROJECT, task_name=infer_name)) is not None:
            continue_task_id = t.task_id
        tracker_params["clearml"] = {
            "task_name": infer_name,
            "tags": [state.exp_name, "infer"],
            "reuse_last_task_id": continue_task_id,
            "continue_last_task": 0 if continue_task_id else False,
            "auto_connect_frameworks": False,
        }
    exp = Preemptible(
        Distributed(
            instantiate(
                infer_config.pop("experiment"),
                exp_config=lambda: infer_config,
                dir=infer_dir,
                debug=state.debug,
                seed=state.seed,
                n_checkpoints=2,
                save_user_metrics=True,
                trackers_params=tracker_params,
            ),
            launcher=(
                instantiate(l) if (l := infer_config.pop("launcher", None)) is not None else None
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
        study.set_user_attr(
            f"infer-{metric}", value.item() if isinstance(value, torch.Tensor) else value
        )


if __name__ == "__main__":
    main()
