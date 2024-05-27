from typing import Any, Callable, Iterable
from copy import deepcopy
import json
from pathlib import Path

from accelerate import Accelerator
from accelerate.utils import set_seed
from hydra.utils import instantiate
from ignite.engine import EventEnum
from loguru import logger
from rich import print_json
import torch
from tqdm import tqdm

from cornac import Experiment as CornacExp
from cornac.data import Dataset
from cornac.eval_methods import BaseMethod
from cornac.metrics import RankingMetric
from cornac.models import Recommender
from experiments import settings
from experiments.base import Experiment
from experiments.utils import flatten_config
from src.metrics import Metric


class CornacExperiment(Experiment):
    def __init__(
        self,
        exp_config: dict[str, Any] | Callable[[], dict[str, Any]],
        dir: Path | None = None,
        n_checkpoints: int = 3,
        datasets_key: str = "datasets",
        metrics: dict[str, Metric] | None = None,
        cornac_metrics: dict[str, RankingMetric] | None = None,
        trackers_params: dict[str, Any] | None = None,
        events: dict[str, list[tuple[EventEnum, Callable]]] | None = None,
        seed: int = 13,
        debug: bool = False,
        save_user_metrics: bool = False,
    ) -> None:
        self._config = exp_config if isinstance(exp_config, dict) else exp_config()
        self._dir = dir
        self._n_checkpoints = n_checkpoints
        self._seed = seed
        self._debug = debug
        self._datasets_key = datasets_key
        self._save_user_metrics = save_user_metrics
        self._metrics = metrics or {}
        self._cornac_metrics = cornac_metrics or {}
        self._trackers_params = trackers_params or {}
        self._events = events or {}

    @property
    def metrics(self) -> dict[str, Any]:
        return {m_name: m.get_metric() for m_name, m in self._metrics.items()}

    def run(self) -> Any:
        self._accelerator = Accelerator(
            log_with=list(self._trackers_params) if len(self._trackers_params) > 0 else None,
        )
        self._accelerator.init_trackers(
            project_name=settings.TRACKER_PROJECT,
            config=flatten_config(self._config),
            init_kwargs=self._trackers_params,
        )
        self._seed_everything()
        print_json(data=self._config)
        eval_method = BaseMethod(seed=self._seed, exclude_unknowns=True, verbose=True)
        eval_method.build(
            train_data=list(prepare_dataset(self._config["datasets"]["train"])),
            test_data=list(prepare_dataset(self._config["datasets"]["eval"])),
        )
        model = instantiate(self._config["model"])
        exp = CornacExp(
            eval_method=eval_method,
            models=[model],
            metrics=list(self._cornac_metrics.values()),
            user_based=True,
            show_validation=True,
            verbose=True,
            save_dir=None,
        )
        exp.run()
        self._calc_metrics(model, train_set=eval_method.train_set, test_set=eval_method.test_set)
        self._log_metrics()
        if self._save_user_metrics and self._dir is not None:
            metrics_copy = deepcopy(self._metrics or {})
            for m in metrics_copy.values():
                m.reset()
            self._user_metrics_saver(
                model,
                dir=self._dir,
                train_set=eval_method.train_set,
                test_set=eval_method.test_set,
                metrics=metrics_copy,
            )
        self._accelerator.end_training()

    def clean(self) -> None:
        self._accelerator.free_memory()

    def _calc_metrics(self, model: Recommender, train_set: Dataset, test_set: Dataset) -> None:
        for m in self._metrics.values():
            m.reset()
        for user in tqdm(set(test_set.uir_tuple[0]), desc="Calc metrics"):
            if len(test_set.csr_matrix.getrow(user).nonzero()[0]) == 0:
                continue
            _, scores = model.rank(user_idx=user)
            target_row = test_set.csr_matrix.getrow(user).toarray().reshape(-1)
            train_row = train_set.csr_matrix.getrow(user).toarray().reshape(-1)
            scores[train_row > 0] = -1e13
            for m in self._metrics.values():
                m(
                    torch.from_numpy(scores).float().view(1, -1),
                    torch.from_numpy(target_row).float().view(1, -1),
                )

    def _log_metrics(self) -> None:
        metrics = {
            m_name: v.item() if isinstance((v := m.get_metric()), torch.Tensor) else v
            for m_name, m in self._metrics.items()
            if not m_name.startswith("_")
        }
        if len(metrics) == 0:
            return
        run_type = "eval"
        logger.info(run_type.capitalize())
        max_length = max(len(x) for x in metrics)
        for metric in sorted(metrics, key=lambda x: (len(x), x)):
            metric_value = metrics.get(metric)
            if isinstance(metric_value, (float, int)):
                logger.info(f"{metric.ljust(max_length)} | {metric_value:.4f}")
        self._accelerator.log({f"{k}_epoch/{run_type}": v for k, v in metrics.items()})

    def _user_metrics_saver(
        self,
        model: Recommender,
        dir: Path,
        train_set: Dataset,
        test_set: Dataset,
        metrics: dict[str, Metric],
    ) -> None:
        id_to_user_map = {id: u for u, id in model.uid_map.items()}
        path = dir / "user-metrics.jsonl"
        with path.open("a", encoding="utf-8") as file:
            for user in tqdm(set(test_set.uir_tuple[0]), desc="Saving output"):
                save_sample = {"user": int(id_to_user_map[user])}
                _, scores = model.rank(user_idx=user)
                target_row = test_set.csr_matrix.getrow(user).toarray().reshape(-1)
                train_row = train_set.csr_matrix.getrow(user).toarray().reshape(-1)
                scores[train_row > 0] = -1e13
                for key, m in metrics.items():
                    save_sample[key] = float(
                        m.compute(
                            torch.from_numpy(scores).float().view(1, -1),
                            torch.from_numpy(target_row).float().view(1, -1),
                        ).item()
                    )
                file.write(json.dumps(save_sample, ensure_ascii=True))
                file.write("\n")

    def _seed_everything(self) -> None:
        import os
        import random

        random.seed(self._seed)
        os.environ["PYTHONHASHSEED"] = str(self._seed)
        set_seed(self._seed)


def prepare_dataset(path: str) -> Iterable[tuple[int, int, float]]:
    with open(path, "r", encoding="utf-8") as file:
        for line in tqdm(map(json.loads, file), desc="Preparing dataset"):
            if isinstance(line["item"], list):
                for item in line["item"]:
                    yield (line["user"], item, 1.0)
                continue
            yield (line["user"], line["item"], 1.0)
