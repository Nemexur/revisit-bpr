from pathlib import Path
import pickle

import optuna
from optuna.storages import fail_stale_trials
from optuna.study import StudyDirection

from experiments.decorator import preemptible_add_to_save

MAX_CONNS = 1
SAMPLER_PATH = "sampler.pkl"


def create_study(
    name: str, dir: Path | None = None, storage_url: str | None = None, seed: int = 13
) -> optuna.Study:
    def save_sampler(path: Path) -> None:
        with path.open("wb") as file:
            pickle.dump(sampler, file)

    sampler = optuna.samplers.TPESampler(seed=seed)
    if dir is not None and (sampler_path := dir / f"seed-{seed}-{SAMPLER_PATH}").exists():
        with sampler_path.open("rb") as file:
            sampler = pickle.load(file)  # noqa: S301
    storage = None
    if storage_url is not None:
        storage = optuna.storages.RDBStorage(
            url=storage_url,
            heartbeat_interval=30,
            failed_trial_callback=optuna.storages.RetryFailedTrialCallback(),
            engine_kwargs={
                "pool_size": MAX_CONNS,
                "connect_args": {"target_session_attrs": "read-write"},
            },
        )
    if storage is None and dir is not None:
        storage = optuna.storages.RDBStorage(
            url="sqlite:///" + str(dir / "optuna.db"),
            heartbeat_interval=30,
            failed_trial_callback=optuna.storages.RetryFailedTrialCallback(),
        )
    study = optuna.create_study(
        storage=storage,
        load_if_exists=True,
        study_name=name,
        sampler=sampler,
        direction=StudyDirection.MAXIMIZE,
    )
    # Just in case manually mark preempted trials
    # for scheduling before the optimization
    if storage is not None:
        fail_stale_trials(study)
    if dir is not None:
        preemptible_add_to_save(dir / f"seed-{seed}-{SAMPLER_PATH}", save_sampler)
    return study
