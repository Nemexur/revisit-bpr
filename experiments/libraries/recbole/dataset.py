from typing import Any, Iterator
from collections import defaultdict
from itertools import islice
import json
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, IterableDataset, get_worker_info


class InMemory(Dataset):
    def __init__(
        self,
        path: Path | str,
        seen_items_path: Path | str,
        mapping: dict[str, dict[str, int]] | None = None,
    ) -> None:
        self.mapping = mapping
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
            "user": self.mapping["user_id"][str(sample["user"])],
            "item": [self.mapping["item_id"][str(i)] for i in sample["item"]],
            "seen_items": [
                self.mapping["item_id"][str(i)] for i in self._seen_items[sample["user"]]
            ],
        }


class Iter(IterableDataset):
    def __init__(
        self,
        path: Path | str,
        seen_items_path: Path | str,
        mapping: dict[str, dict[str, int]] | None = None,
    ) -> None:
        self.mapping = mapping
        self._path = Path(path)
        with Path(seen_items_path).open("r", encoding="utf-8") as file:
            self._seen_items = {
                user_items["user"]: user_items["seen_items"] for user_items in map(json.loads, file)
            }

    def __iter__(self) -> Iterator[dict[str, Any]]:
        worker_info = get_worker_info()
        with self._path.open("r", encoding="utf-8") as file:
            start, step = 0, 1
            if worker_info is not None and worker_info.num_workers > 0:
                start, step = worker_info.id, worker_info.num_workers
            lines = islice(file, start, None, step)
            for line in lines:
                sample = json.loads(line)
                sample = {
                    "user": self.mapping["user_id"][str(sample["user"])],
                    "item": [self.mapping["item_id"][str(i)] for i in sample["item"]],
                    "seen_items": [
                        self.mapping["item_id"][str(i)] for i in self._seen_items[sample["user"]]
                    ],
                }
                yield sample


class NetflixIter(IterableDataset):
    def __init__(
        self,
        path: Path | str,
        mapping: dict[str, dict[str, int]] | None = None,
    ) -> None:
        self.mapping = mapping
        self._path = Path(path)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        worker_info = get_worker_info()
        with self._path.open("r", encoding="utf-8") as file:
            start, step = 0, 1
            if worker_info is not None and worker_info.num_workers > 0:
                start, step = worker_info.id, worker_info.num_workers
            lines = islice(file, start, None, step)
            for line in lines:
                sample = json.loads(line)
                try:
                    sample = {
                        "user": self.mapping["user_id"][str(sample["user"])],
                        "item": self.mapping["item_id"][str(sample["seen_items"][sample["item"]])],
                        "seen_items": [
                            self.mapping["item_id"][str(i)] for i in sample["seen_items"]
                        ],
                    }
                except:  # noqa: E722, S112
                    continue
                yield sample


class OnePosCollator:
    def __init__(self, num_items: int) -> None:
        self._num_items = num_items

    def __call__(self, instances: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        batch = {key: torch.tensor(tensor) for key, tensor in self._make_batch(instances).items()}
        return self._process(batch)

    def _process(self, batch) -> dict[str, torch.Tensor]:
        seen_items = batch["seen_items"].view(-1)
        all_items = torch.ones(self._num_items, dtype=torch.long)
        all_items[0] = 0  # Discard padding
        all_items[seen_items] = 0
        batch["item"] = torch.hstack(
            (
                batch["item"].unsqueeze(0),
                torch.arange(self._num_items)[all_items.ne(0)].unsqueeze(0),
            )
        )
        target = torch.zeros_like(batch["item"], dtype=torch.float)
        target[:, 0] = 1.0
        batch["target"] = target
        return batch

    @staticmethod
    def _make_batch(instances: list[dict[str, Any]]) -> dict[str, list[Any]]:
        tensor_dict = defaultdict(list)
        for instance in instances:
            for field, tensor in instance.items():
                tensor_dict[field].append(tensor)
        return tensor_dict


class AllItemsCollator:
    def __init__(self, num_items: int, padding_value: float = 0) -> None:
        self.num_items = num_items
        self._padding_value = padding_value

    def __call__(self, instances: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        batch = self._make_batch(instances)
        for idx, b_item in enumerate(batch["item"]):
            batch["item"][idx] = torch.arange(self.num_items, dtype=torch.long)
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
            for field, tensor in instance.items():
                tensor_dict[field].append(tensor)
        return tensor_dict
