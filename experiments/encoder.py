from typing import Any, Iterable, Iterator, Optional, Union
from copy import deepcopy
from dataclasses import dataclass, field
import json
from pathlib import Path

OOV_TOKEN = "[OOV]"


@dataclass
class AttrEncoder:
    item_to_idx: dict[str, int] = field(default_factory=dict)
    idx_to_item: dict[str, Any] = field(default_factory=dict)
    oov: bool = False

    def __post_init__(self) -> None:
        if self.oov:
            self.add(OOV_TOKEN)

    def __len__(self) -> int:
        return len(self.item_to_idx)

    def __contains__(self, v: str) -> bool:
        return v in self.item_to_idx

    def add(self, item: Any) -> None:
        idx = len(self.item_to_idx)
        self.item_to_idx[str(item)] = idx
        self.idx_to_item[str(idx)] = item

    def encode(self, item: Any) -> Union[int, None]:
        return self.item_to_idx.get(str(item), 0 if self.oov else None)

    def decode(self, id: int) -> Union[Any, None]:
        return self.idx_to_item.get(str(id), OOV_TOKEN if self.oov else None)

    def is_empty(self) -> bool:
        return len(self) <= 1

    def save(self, path: Path) -> None:
        with path.open("w", encoding="utf-8") as file:
            json.dump(
                {"item_to_idx": self.item_to_idx, "idx_to_item": self.idx_to_item},
                file,
                indent=4,
                ensure_ascii=False,
            )

    def load(self, path: Path) -> "AttrEncoder":
        with path.open("r", encoding="utf-8") as file:
            state = json.load(file)
        self.item_to_idx, self.idx_to_item = state["item_to_idx"], state["idx_to_item"]
        return self


class JsonLEncoder:
    def __init__(self, attrs: Optional[Union[list[str], dict[str, AttrEncoder]]] = None) -> None:
        if attrs is None:
            attrs = ["user", "item"]
        self.attrs = {a: AttrEncoder() for a in attrs} if isinstance(attrs, list) else attrs

    def size(self, key: str = "user") -> int:
        return len(self.attrs[key])

    def fit(self, data: Iterable[dict[str, Any]]) -> "JsonLEncoder":
        for sample in data:
            for a, e in self.attrs.items():
                sample_value = sample[a]
                if sample_value in e.item_to_idx:
                    continue
                e.add(sample_value)
        return self

    def transform(
        self,
        data: Iterable[dict[str, Any]],
        decode: bool = False,
        lazy: bool = False,
    ) -> Iterable[dict[str, Any]]:
        def internal() -> Iterator[dict[str, Any]]:
            for sample in data:
                sample_copy = deepcopy(sample)
                for a, e in self.attrs.items():
                    func = e.encode if not decode else e.decode
                    sample_copy[a] = func(sample_copy[a])
                yield sample_copy

        if any(a.is_empty() for a in self.attrs.values()):
            raise ValueError("encoder: one or more attr encoders are empty")
        if not lazy:
            return list(internal())
        return internal()

    def save(self, path: Path) -> None:
        for key, e in self.attrs.items():
            e.save(path / f"{key}.json")

    def load(self, path: Path) -> "JsonLEncoder":
        for key, e in self.attrs.items():
            e.load(path / f"{key}.json")
        return self
