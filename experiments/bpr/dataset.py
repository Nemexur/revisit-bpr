from typing import Any, Iterator
from collections import defaultdict
from itertools import islice
import json
from pathlib import Path
import random

import numpy as np
from scipy import sparse as sps
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, IterableDataset, get_worker_info
from tqdm import tqdm


class InMemory(Dataset):
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


class Iter(IterableDataset):
    def __init__(self, path: Path | str, seen_items_path: Path | str) -> None:
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
                sample["seen_items"] = self._seen_items[sample["user"]]
                yield sample


class SamplingInMemory(Dataset):
    def __init__(self, path: Path | str, seen_items_path: Path | str, seed: int = 13) -> None:
        self._gen = random.Random(seed)  # noqa: S311
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
            "user": sample["user"],
            "item": self._gen.choice(sample["item"]),
            "seen_items": self._seen_items[sample["user"]],
        }


class SamplingIter(IterableDataset):
    def __init__(self, path: Path | str, seen_items_path: Path | str, seed: int = 13) -> None:
        self._path = Path(path)
        self._gen = random.Random(seed)  # noqa: S311
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
                yield {
                    "user": sample["user"],
                    "item": self._gen.choice(sample["item"]),
                    "seen_items": self._seen_items[sample["user"]],
                }


class SparseSamplingInMemory(Dataset):
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


class SparseSamplingInMemoryWithCollator(Dataset):
    def __init__(
        self,
        path: Path | str,
        seen_items_path: Path | str,
        num_users: int,
        num_items: int,
        padding_value: float = 0,
        put_on_cuda: bool = False,
    ) -> None:
        matrix = self._sparse_matrix(path, num_users=num_users, num_items=num_items)
        self._user_ids = torch.from_numpy(
            np.repeat(np.arange(num_users, dtype=matrix.indptr.dtype), np.ediff1d(matrix.indptr))
        )
        self._item_ids = torch.from_numpy(matrix.indices)
        seen_items: list[torch.Tensor] = [torch.tensor([0]) for _ in range(num_users)]
        with Path(seen_items_path).open("r", encoding="utf-8") as file:
            for user_items in map(json.loads, file):
                seen_items[user_items["user"]] = torch.tensor(
                    user_items["seen_items"], dtype=torch.long
                )
        self._seen_items = pad_sequence(seen_items, batch_first=True, padding_value=padding_value)
        if put_on_cuda and torch.cuda.is_available():
            self._user_ids = self._user_ids.cuda()
            self._item_ids = self._item_ids.cuda()
            self._seen_items = self._seen_items.cuda()

    def __len__(self) -> int:
        return len(self._user_ids)

    def __getitem__(self, idx: int) -> int:
        return idx

    def collate_fn(self, indices: list[int]) -> dict[str, torch.Tensor]:
        users, items = self._user_ids[indices], self._item_ids[indices]
        return {
            "user": users,
            "item": items,
            "seen_items": self._seen_items[users],
        }

    def _sparse_matrix(self, path: Path | str, num_users: int, num_items: int) -> sps.csr_matrix:
        user_key, item_key = "user", "item"
        matrix = sps.dok_matrix((num_users, num_items), dtype=np.float32)
        with Path(path).open("r", encoding="utf-8") as file:
            for sample in tqdm(map(json.loads, file), desc="Building sparse matrix"):
                user_idx, item_idx = sample[user_key], sample[item_key]
                matrix[user_idx, item_idx] = 1.0
        return matrix.tocsr()


class OnePosCollator:
    def __init__(self, num_items: int) -> None:
        self._num_items = num_items

    def __call__(self, instances: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        batch = {key: torch.tensor(tensor) for key, tensor in self._make_batch(instances).items()}
        return self._process(batch)

    def _process(self, batch) -> dict[str, torch.Tensor]:
        seen_items = batch["seen_items"].view(-1)
        pos_item = seen_items[batch["item"]]
        all_items = torch.ones(self._num_items, dtype=torch.long)
        # Discard padding
        all_items[0] = 0
        all_items[seen_items] = 0
        batch["item"] = torch.hstack(
            (
                pos_item.unsqueeze(0),
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


class ManyPosCollator:
    def __init__(self, num_items: int, padding_value: float = 0) -> None:
        self._num_items = num_items
        self._padding_value = padding_value

    def __call__(self, instances: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        batch = self._make_batch(instances)
        for idx, (b_item, b_seen) in enumerate(
            zip(batch["item"], batch["seen_items"], strict=True)
        ):
            num_pos = len(b_item)
            all_items = torch.ones(self._num_items, dtype=torch.long)
            all_items[0] = 0
            all_items[torch.tensor(b_seen)] = 0
            batch["item"][idx] = torch.hstack(
                (torch.tensor(b_item), torch.arange(self._num_items)[all_items.gt(0)])
            )
            target = torch.zeros_like(batch["item"][idx], dtype=torch.float)
            target[:num_pos] = 1.0
            batch["target"].append(target)
        tensor_batch = {
            "user": torch.as_tensor(batch["user"]),
            "item": pad_sequence(
                batch["item"], batch_first=True, padding_value=self._padding_value
            ),
            "seen_items": pad_sequence(
                [torch.as_tensor(t) for t in batch["seen_items"]],
                batch_first=True,
                padding_value=self._padding_value,
            ),
            "target": pad_sequence(
                batch["target"], batch_first=True, padding_value=self._padding_value
            ),
        }
        tensor_batch["mask"] = tensor_batch["item"].gt(self._padding_value).float()
        return tensor_batch

    @staticmethod
    def _make_batch(instances: list[dict[str, Any]]) -> defaultdict[str, list[Any]]:
        tensor_dict = defaultdict(list)
        for instance in instances:
            for field, tensor in instance.items():
                tensor_dict[field].append(tensor)
        return tensor_dict


class AllItemsCollator:
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
            for field, tensor in instance.items():
                tensor_dict[field].append(tensor)
        return tensor_dict
