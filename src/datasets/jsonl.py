from typing import Any, Iterator
from collections import defaultdict
from itertools import islice
import json
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, IterableDataset, get_worker_info


class InMemory(Dataset):
    def __init__(self, path: Path | str) -> None:
        with Path(path).open("r", encoding="utf-8") as file:
            self._samples = [json.loads(line) for line in file]

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self._samples[idx]


class Iter(IterableDataset):
    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        worker_info = get_worker_info()
        with self._path.open("r", encoding="utf-8") as file:
            start, step = 0, 1
            if worker_info is not None and worker_info.num_workers > 0:
                start, step = worker_info.id, worker_info.num_workers
            lines = islice(file, start, None, step)
            yield from map(json.loads, lines)


class Collator:
    def __init__(self, pad: list[str] | None = None, padding_value: float = 0) -> None:
        self._pad = set(pad or [])
        self._padding_value = padding_value

    def __call__(self, instances: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        batch = {
            key: (
                pad_sequence(
                    [torch.as_tensor(t) for t in tensor],
                    batch_first=True,
                    padding_value=self._padding_value,
                )
                if key in self._pad
                else torch.tensor(tensor)
            )
            for key, tensor in self._make_batch(instances).items()
        }
        for key in self._pad:
            batch[f"{key}_mask"] = batch[key].ne(self._padding_value).float()
        return batch

    @staticmethod
    def _make_batch(instances: list[dict[str, Any]]) -> dict[str, list[Any]]:
        tensor_dict = defaultdict(list)
        for instance in instances:
            for field, tensor in instance.items():
                tensor_dict[field].append(tensor)
        return tensor_dict
