from typing import Any
from collections import defaultdict
from dataclasses import dataclass, field
import json
import math
from pathlib import Path

from accelerate import Accelerator
from accelerate.optimizer import AcceleratedOptimizer
from accelerate.utils import set_seed
import click
from loguru import logger
import numpy as np
from scipy import sparse as sps
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.datasets.jsonl import Collator
from src.metrics import NDCG, Metric, Precision, Recall, RocAucManySlow
from src.models import BPR
from src.models.bpr import MF
from src.modules import AdaptiveSampler, Sampler

PBAR_FORMAT = (
    "{desc} [{n_fmt}/{total_fmt}] "
    "{percentage:3.0f}%|{bar}|{postfix} "
    "({elapsed}<{remaining}, {rate_fmt})"
)


class TrainDatasetInMemory(Dataset):
    def __init__(
        self,
        path: Path | str,
        seen_items_path: Path | str,
        num_users: int,
        num_items: int,
    ) -> None:
        matrix = self._sparse_matrix(path, num_users=num_users, num_items=num_items)
        self._user_ids = np.repeat(
            np.arange(num_users, dtype=matrix.indptr.dtype), np.ediff1d(matrix.indptr)
        )
        self._item_ids = matrix.indices
        with Path(seen_items_path).open("r", encoding="utf-8") as file:
            self._seen_items = {
                user_items["user"]: user_items["seen_items"] for user_items in map(json.loads, file)
            }

    def __len__(self) -> int:
        return len(self._user_ids)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return {
            "user": (user := int(self._user_ids[idx])),
            "item": int(self._item_ids[idx]),
            "seen_items": self._seen_items[user],
        }

    def _sparse_matrix(self, path: Path | str, num_users: int, num_items: int) -> sps.csr_matrix:
        user_key, item_key = "user", "item"
        matrix = sps.dok_matrix((num_users, num_items), dtype=np.float32)
        with Path(path).open("r", encoding="utf-8") as file:
            for sample in tqdm(map(json.loads, file), desc="Building sparse matrix"):
                user_idx, item_idx = sample[user_key], sample[item_key]
                matrix[user_idx, item_idx] = 1.0
        return matrix.tocsr()


class EvalDatasetInMemory(Dataset):
    def __init__(self, path: Path | str, seen_items_path: Path | str) -> None:
        with Path(path).open("r", encoding="utf-8") as file:
            self._samples = [json.loads(line) for line in file]
        with Path(seen_items_path).open("r", encoding="utf-8") as file:
            self._seen_items = {
                user_items["user"]: user_items["seen_items"] for user_items in map(json.loads, file)
            }

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self._samples[idx]
        return {
            **sample,
            "seen_items": self._seen_items[sample["user"]],
        }


class EvalCollator:
    def __init__(self, num_items: int, padding_value: float = 0) -> None:
        self._num_items = num_items
        self._padding_value = padding_value

    def __call__(self, instances: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        batch = self._make_batch(instances)
        for idx, b_item in enumerate(batch["item"]):
            batch["item"][idx] = torch.arange(self._num_items, dtype=torch.long)
            target = torch.zeros_like(batch["item"][idx], dtype=torch.float)
            target[torch.tensor(b_item)] = 1.0
            batch["target"].append(target)
        tensor_batch = {
            "user": torch.as_tensor(batch["user"]),
            "item": torch.stack(batch["item"]),
            "target": torch.stack(batch["target"]),
            "seen_items": pad_sequence(
                [torch.as_tensor(t) for t in batch["seen_items"]],
                batch_first=True,
                padding_value=self._padding_value,
            ),
        }
        return tensor_batch

    @staticmethod
    def _make_batch(instances: list[dict[str, Any]]) -> defaultdict[str, list[Any]]:
        tensor_dict = defaultdict(list)
        for instance in instances:
            for key, tensor in instance.items():
                tensor_dict[key].append(tensor)
        return tensor_dict


@dataclass()
class EpochState:
    name: str
    accelerator: Accelerator
    iteration: int = 0
    metrics: dict[str, torch.Tensor] = field(default_factory=dict)

    def log(self) -> None:
        metrics = {
            k: v.item() if isinstance(v, torch.Tensor) else v
            for k, v in self.metrics.items()
            if not k.startswith("_")
        }
        if len(metrics) == 0:
            return
        logger.info(self.name.capitalize())
        max_length = max(len(x) for x in metrics)
        for metric in sorted(metrics, key=lambda x: (len(x), x)):
            metric_value = metrics.get(metric)
            if isinstance(metric_value, (float, int)):
                logger.info(f"{metric.ljust(max_length)} | {metric_value:.4f}")
        self.accelerator.log({f"{k}_epoch/{self.name}": v for k, v in metrics.items()})


def seed_everything(seed) -> None:
    import os
    import random

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    set_seed(seed)


def train_one_epoch(
    model: BPR,
    accelerator: Accelerator,
    optimizer: AcceleratedOptimizer,
    loader: DataLoader,
    neg_sampler: Sampler,
) -> EpochState:
    model.train()
    state = EpochState(name="train", accelerator=accelerator)
    state.metrics["_loss"] = torch.tensor(0.0, device=accelerator.device)
    with tqdm(
        total=len(loader),
        bar_format=PBAR_FORMAT,
        desc="\033[33m" + state.name.capitalize() + "\033[00m",
    ) as pbar:
        for batch in loader:
            state.iteration += 1
            if batch["item"].dim() < 2:
                batch["item"].unsqueeze_(-1)
            batch["neg"] = neg_sampler.sample(batch)
            output = model(batch)
            accelerator.backward(output["loss"])
            optimizer.step()
            optimizer.zero_grad()
            if "loss" in output:
                state.metrics["_loss"] += output["loss"].detach()
                state.metrics["loss"] = state.metrics["_loss"] / state.iteration
            pbar.set_postfix(
                {
                    k: v.item() if isinstance(v, torch.Tensor) else v
                    for k, v in state.metrics.items()
                    if not k.startswith("_")
                }
            )
            pbar.update()
    return state


@torch.no_grad()
def eval(
    model: BPR,
    accelerator: Accelerator,
    loader: DataLoader,
    metrics: dict[str, Metric],
) -> EpochState:
    model.eval()
    state = EpochState(name="eval", accelerator=accelerator)
    with tqdm(
        total=len(loader),
        bar_format=PBAR_FORMAT,
        desc="\033[33m" + state.name.capitalize() + "\033[00m",
    ) as pbar:
        for batch in loader:
            state.iteration += 1
            if batch["item"].dim() < 2:
                batch["item"].unsqueeze_(-1)
            output = model(batch)
            if (seen_items := batch.get("seen_items")) is not None:
                output["logits"].scatter_(dim=-1, index=seen_items, value=-1e13)
                output["logits"][:, 0] = -1e13
            for key, m in metrics.items():
                if "target" not in batch:
                    continue
                m(output["logits"], batch["target"])
                state.metrics[key] = m.get_metric()
            pbar.set_postfix(
                {
                    k: v.item() if isinstance(v, torch.Tensor) else v
                    for k, v in state.metrics.items()
                    if not k.startswith("_") and k.endswith("@100")
                }
            )
            pbar.update()
    return state


@click.command(
    help="Train BPR on ML-20M with the best hyperparameters.",
    context_settings={"help_option_names": ["-h", "--help"]},
)
@click.argument(
    "dataset_path", type=click.Path(exists=True, dir_okay=True, file_okay=False, path_type=Path)
)
@click.option("--num-users", type=click.INT, required=False, default=136678, show_default=True)
@click.option("--num-items", type=click.INT, required=False, default=20109, show_default=True)
@click.option("--embedding-dim", type=click.INT, required=False, default=1024, show_default=True)
@click.option("--batch-size", type=click.INT, required=False, default=256, show_default=True)
@click.option("--epochs", type=click.INT, required=False, default=72, show_default=True)
@click.option("--seed", type=click.INT, required=False, default=13, show_default=True)
def main(
    dataset_path: Path,
    num_users: int,
    num_items: int,
    embedding_dim: int,
    batch_size: int,
    epochs: int,
    seed: int,
) -> None:
    accelerator = Accelerator()
    metrics: dict[str, Metric] = {
        "ndcg@100": NDCG(topk=100),
        "recall@100": Recall(topk=100),
        "ndcg@10": NDCG(topk=10),
        "recall@10": Recall(topk=10),
        "auc": RocAucManySlow(),
        "ndcg@5": NDCG(topk=5),
        "recall@5": Recall(topk=5),
        "ndcg@50": NDCG(topk=50),
        "recall@50": Recall(topk=50),
        "precision@5": Precision(topk=5),
        "precision@10": Precision(topk=10),
        "precision@50": Precision(topk=50),
        "precision@100": Precision(topk=100),
    }
    seed_everything(seed)
    for m in metrics.values():
        m.set_accelerator(accelerator)
    model = BPR(
        fuse_forward=True,
        logits_model=MF(
            user_emb=torch.nn.Embedding(
                num_embeddings=num_users,
                embedding_dim=embedding_dim,
                padding_idx=0,
            ),
            item_emb=torch.nn.Embedding(
                num_embeddings=num_items,
                embedding_dim=embedding_dim,
                padding_idx=0,
            ),
            item_bias=False,
            user_bias=False,
        ),
        reg_alphas={
            "user": 0.0016,
            "item": 0.0001,
            "neg": 0.00375,
        },
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.00943667980759196)
    model, optimizer = accelerator.prepare(model, optimizer)
    neg_sampler = AdaptiveSampler(
        model,
        num_items=num_items,
        sampling_prob=1 / 700,
        every=int(num_items * math.log(num_items) / batch_size),
        neg_gen=torch.Generator(device=accelerator.device).manual_seed(seed),
    )
    neg_sampler.update_stats()
    datasets = {
        "train": accelerator.prepare_data_loader(
            DataLoader(
                dataset=TrainDatasetInMemory(
                    path=dataset_path / "full-train-with-fold-in.jsonl",
                    seen_items_path=dataset_path / "full-train-with-fold-in-user-seen-items.jsonl",
                    num_users=num_users,
                    num_items=num_items,
                ),
                collate_fn=Collator(pad=["seen_items"]),
                batch_size=batch_size,
                shuffle=True,
                pin_memory=True,
                num_workers=3,
            )
        ),
        "eval": accelerator.prepare_data_loader(
            DataLoader(
                dataset=EvalDatasetInMemory(
                    path=dataset_path / "test-grouped.jsonl",
                    seen_items_path=dataset_path / "full-train-with-fold-in-user-seen-items.jsonl",
                ),
                collate_fn=EvalCollator(num_items),
                batch_size=batch_size,
                shuffle=False,
                pin_memory=True,
                num_workers=3,
            )
        ),
    }
    for e in range(epochs):
        train_state = train_one_epoch(
            model=model,
            accelerator=accelerator,
            optimizer=optimizer,
            loader=datasets["train"],
            neg_sampler=neg_sampler,
        )
        train_state.log()
        eval_state = eval(
            model=model,
            accelerator=accelerator,
            loader=datasets["eval"],
            metrics=metrics,
        )
        eval_state.log()
        for m in metrics.values():
            m.reset()
        logger.info(f"Finished epoch: {e + 1}")
    accelerator.end_training()


if __name__ == "__main__":
    main()
