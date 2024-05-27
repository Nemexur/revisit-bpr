from typing import Any, Callable
from dataclasses import asdict
import os
from pathlib import Path
import sys
import traceback

from accelerate import Accelerator
import boto3
from clearml import Task
import click
from hydra.utils import instantiate
from ignite.engine import Engine, Events
from jinja2 import StrictUndefined, Template
from loguru import logger
import torch
import yaml

from experiments.base import Experiment
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
from experiments.decorator import _PREEMPT_TO_SAVE, Distributed, Preemptible, Status
from experiments.hp import SAMPLER_PATH, create_study
from experiments.s3.fs import S3FS
from experiments.settings import CHECKPOINTS_DIR, TRACKER_PROJECT
from experiments.utils import merge_configs


class S3Saver(Experiment):
    def __init__(self, exp: Experiment, dir: Path | None, s3fs: S3FS, clean: bool = False) -> None:
        self._exp = exp
        self._dir = dir
        self._s3fs = s3fs
        self._clean = clean

    @property
    def metrics(self) -> dict[str, Any]:
        return self._exp.metrics

    def run(self) -> Any:
        res = self._exp.run()
        if self._dir is None:
            return res
        try:
            for path in filter(os.path.exists, _PREEMPT_TO_SAVE):
                self._s3fs.upload(dst=path, src=path, exist_ok=True, overwrite=True)
                logger.info(f"saved file: {path}")
            self._s3fs.upload(dst=self._dir, src=self._dir, exist_ok=True)
            logger.info(f"saved dir: {self._dir}")
        except Exception as exc:
            print(traceback.format_exc())
            logger.error(f"exception: {exc}")
        return res

    def clean(self) -> None:
        self._exp.clean()

    def interrupt(self) -> None:
        self._exp.interrupt()


def s3_save_handler(
    s3fs: S3FS, dir: Path | None, clean: bool = False
) -> Callable[[Engine, Accelerator], None]:
    def handler(engine: Engine, accelerator: Accelerator) -> None:
        if dir is None:
            return
        try:
            upload_dir = dir / CHECKPOINTS_DIR
            if clean:
                s3fs.remove(upload_dir)
            s3fs.upload(dst=upload_dir, src=upload_dir, exist_ok=True, overwrite=True)
            logger.info(f"saved dir: {upload_dir}")
        except Exception as exc:
            print(traceback.format_exc())
            logger.error(f"exception: {exc}")

    return handler


@click.command(
    help="Run infer.",
    context_settings={"help_option_names": ["-h", "--help"]},
)
@click.argument("config_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--s3-bucket",
    type=click.STRING,
    help="S3 bucket to sync artifacts.",
    required=True,
)
@click.option(
    "--s3-access-key",
    type=click.STRING,
    help="S3 bucket access key.",
    required=True,
    envvar="AWS_ACCESS_KEY_ID",
)
@click.option(
    "--s3-secret-key",
    type=click.STRING,
    help="S3 bucket secret key.",
    required=True,
    envvar="AWS_SECRET_ACCESS_KEY",
)
@click.option(
    "--s3-endpoint",
    type=click.STRING,
    help="S3 bucket endpoint.",
    required=True,
    envvar="AWS_ENDPOINT",
)
@name_option("exp")
@dir_option
@seed_option
@clearml_option
@wandb_option
@extra_vars_option
@search_hp_seed_option
@search_hp_storage_option
@pass_state
def main(
    state: State,
    config_path: Path,
    s3_bucket: str,
    s3_access_key: str,
    s3_secret_key: str,
    s3_endpoint: str,
) -> None:
    with config_path.open("r", encoding="utf-8") as file:
        tmpl = Template(file.read(), undefined=StrictUndefined, autoescape=True)
        config = yaml.safe_load(tmpl.render(asdict(state) | (state.extra_vars or {})))
    s3_client = boto3.session.Session().client(
        service_name="s3",
        aws_access_key_id=s3_access_key,
        aws_secret_access_key=s3_secret_key,
        endpoint_url=s3_endpoint,
        config=boto3.session.Config(),
    )
    s3fs = S3FS(bucket=s3_bucket, s3_client=s3_client)
    if state.exp_dir is not None:
        state.exp_dir.mkdir(exist_ok=True)
        if s3fs.exists(
            (sampler_path := state.exp_dir / f"seed-{state.search_hp.seed}-{SAMPLER_PATH}")
        ):
            s3fs.load(dst=sampler_path, src=sampler_path, exist_ok=True)
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
        if s3fs.exists(infer_dir):
            s3fs.load(dst=infer_dir, src=infer_dir, exist_ok=True)
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
                events={
                    "eval": [(Events.COMPLETED, s3_save_handler(s3fs, dir=infer_dir, clean=True))]
                },
            ),
            launcher=(
                instantiate(l) if (l := infer_config.pop("launcher", None)) is not None else None
            ),
        )
    )
    exp = S3Saver(exp, dir=infer_dir, s3fs=s3fs, clean=True)
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
