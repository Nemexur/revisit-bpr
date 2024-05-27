from typing import Any, Callable
import json
from copy import deepcopy
from pathlib import Path

from accelerate import Accelerator
from accelerate.utils import send_to_device, set_seed
from hydra.utils import instantiate
from ignite.engine import EventEnum
from loguru import logger
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.data.dataset import Dataset
from recbole.utils import get_model, get_trainer
from rich import print_json
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from experiments import settings
from experiments.base import Experiment
from experiments.utils import flatten_config
from src.metrics import Metric


class RecboleExperiment(Experiment):
    def __init__(
        self,
        exp_config: dict[str, Any] | Callable[[], dict[str, Any]],
        dir: Path | None = None,
        n_checkpoints: int = 2,
        datasets_key: str = "datasets",
        metrics: dict[str, Metric] | None = None,
        trackers_params: dict[str, Any] | None = None,
        events: dict[str, list[tuple[EventEnum, Callable]]] | None = None,
        seed: int = 13,
        debug: bool = False,
        skip_seen: bool = True,
        save_user_metrics: bool = False,
    ) -> None:
        self._config = exp_config if isinstance(exp_config, dict) else exp_config()
        self._dir = dir
        self._n_checkpoints = n_checkpoints
        self._seed = seed
        self._debug = debug
        self._skip_seen = skip_seen
        self._datasets_key = datasets_key
        self._save_user_metrics = save_user_metrics
        self._metrics = metrics or {}
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
        for m in self._metrics.values():
            m.set_accelerator(self._accelerator)
        self._seed_everything()
        print_json(data=self._config)
        config = Config(
            config_dict=self._config["recbole_config"],
            dataset=self._config[self._datasets_key].pop("train"),
        )
        config["device"] = self._accelerator.device
        dataset = create_dataset(config)
        # Set token mapping
        eval_loader = instantiate(
            self._config[self._datasets_key]["eval"],
            generator=torch.Generator().manual_seed(self._seed),
            worker_init_fn=seed_worker,
        )
        eval_loader.dataset.mapping = dataset.field2token_id
        eval_loader.collate_fn.num_items = len(dataset.field2token_id["item_id"])
        train_loader = data_preparation(config, dataset)[0]
        self._model = get_model(config["model"])(config, train_loader.dataset).to(config["device"])
        self._evaluate(eval_loader)
        self._log_metrics()
        for m in self._metrics.values():
            m.reset()
        trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, self._model)
        trainer.fit(train_loader, show_progress=True, saved=True)
        self._evaluate(eval_loader)
        self._log_metrics()
        if self._save_user_metrics and self._dir is not None:
            metrics_copy = deepcopy(self._metrics or {})
            for m in metrics_copy.values():
                m.reset()
            self._user_metrics_saver(eval_loader, recbole_dataset=dataset, metrics=metrics_copy)
        self._accelerator.end_training()

    def clean(self) -> None:
        self._accelerator.free_memory()

    def _evaluate(self, loader: DataLoader) -> None:
        for batch in tqdm(loader, desc="Evaluating"):
            batch = send_to_device(batch, device=self._accelerator.device)
            user_logits, item_logits = (
                self._model.get_user_embedding(batch["user"]),
                self._model.get_item_embedding(batch["item"]),
            )
            output = {"logits": torch.einsum("bh,b...h->b...", user_logits, item_logits)}
            if self._skip_seen and (seen_items := batch.get("seen_items")) is not None:
                output["logits"].scatter_(dim=-1, index=seen_items, value=-1e13)
                output["logits"][:, 0] = -1e13
            for m in self._metrics.values():
                m(output["logits"], batch["target"])

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
        self, loader: DataLoader, recbole_dataset: Dataset, metrics: dict[str, Metric]
    ) -> None:
        path = self._dir / "user-metrics.jsonl"
        with path.open("a", encoding="utf-8") as file:
            for batch in tqdm(loader, desc="Saving output"):
                batch = send_to_device(batch, device=self._accelerator.device)
                user_logits, item_logits = (
                    self._model.get_user_embedding(batch["user"]),
                    self._model.get_item_embedding(batch["item"]),
                )
                output = {"logits": torch.einsum("bh,b...h->b...", user_logits, item_logits)}
                if self._skip_seen and (seen_items := batch.get("seen_items")) is not None:
                    output["logits"].scatter_(dim=-1, index=seen_items, value=-1e13)
                    output["logits"][:, 0] = -1e13
                batch_metrics = {
                    key: m.compute(output["logits"], batch["target"]) for key, m in metrics.items()
                }
                for idx, user in enumerate(batch["user"]):
                    save_sample = {
                        "user": int(recbole_dataset.field2id_token["user_id"][int(user.item())])
                    }
                    for m in metrics:
                        save_sample[m] = float(batch_metrics[m][idx].item())
                    file.write(json.dumps(save_sample, ensure_ascii=True))
                    file.write("\n")

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
