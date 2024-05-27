from typing import Any, Iterator
from collections import defaultdict
from itertools import islice
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info


class InMemory(Dataset):
    def __init__(self, path: Path | str, num_items: int) -> None:
        self._num_items = num_items
        with Path(path).open("r", encoding="utf-8") as file:
            self._samples = [json.loads(line) for line in file]

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = {
            key: torch.zeros(self._num_items).scatter_(dim=-1, index=torch.tensor(items), value=1.0)
            for key, items in self._samples[idx].items()
            if key != "user"
        }
        if "user" in self._samples[idx]:
            sample["user"] = torch.tensor(self._samples[idx]["user"], dtype=torch.long)
        return sample


class Iter(IterableDataset):
    def __init__(self, path: Path | str, num_items: int) -> None:
        self._path = Path(path)
        self._num_items = num_items

    def __iter__(self) -> Iterator[dict[str, Any]]:
        worker_info = get_worker_info()
        with self._path.open("r", encoding="utf-8") as file:
            start, step = 0, 1
            if worker_info is not None and worker_info.num_workers > 0:
                start, step = worker_info.id, worker_info.num_workers
            lines = islice(file, start, None, step)
            for line in map(json.loads, lines):
                sample = {
                    key: torch.zeros(self._num_items).scatter_(
                        dim=-1, index=torch.tensor(items), value=1.0
                    )
                    for key, items in line.items()
                    if key != "user"
                }
                if "user" in line:
                    sample["user"] = torch.tensor(line["user"], dtype=torch.long)
                yield sample


class Collator:
    def __call__(self, instances: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        batch = {
            key: torch.vstack(tensor) if key != "user" else torch.hstack(tensor)
            for key, tensor in self._make_batch(instances).items()
        }
        return batch

    @staticmethod
    def _make_batch(instances: list[dict[str, Any]]) -> defaultdict[str, list[Any]]:
        tensor_dict = defaultdict(list)
        for instance in instances:
            for field, tensor in instance.items():
                tensor_dict[field].append(tensor)
        return tensor_dict
