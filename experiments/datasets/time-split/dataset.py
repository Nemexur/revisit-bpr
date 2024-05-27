from typing import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import click
import pandas as pd


@dataclass
class NamespaceEncoder:
    item_to_idx: dict[str, int] = field(default_factory=dict)
    idx_to_item: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.add("@@OOV@@")

    def __len__(self) -> int:
        return len(self.item_to_idx)

    def add(self, item: str) -> None:
        idx = len(self.item_to_idx)
        self.item_to_idx[item] = idx
        self.idx_to_item[str(idx)] = item


class Encoder:
    def __init__(self, user_idx: str = "user", item_idx: str = "item") -> None:
        self._user_idx = user_idx
        self._item_idx = item_idx
        self.namespaces = {user_idx: NamespaceEncoder(), item_idx: NamespaceEncoder()}

    def encode(self, data: Iterable[str], namespace: str) -> list[str]:
        encoder = self.namespaces[namespace].item_to_idx
        return [encoder.get(x, 0) for x in data if x in encoder]

    def decode(self, data: Iterable[str | int], namespace: str) -> list[str]:
        decoder = self.namespaces[namespace].idx_to_item
        return [decoder.get(x, "@@OOV@@") for x in map(str, data) if x in decoder]

    def fit(self, data: pd.DataFrame) -> "Encoder":
        for attr, encoder in self.namespaces.items():
            for item in data[attr].unique():
                encoder.add(item)
        return self

    def transform(self, data: pd.DataFrame, decode: bool = False) -> pd.DataFrame:
        data_copy = data.copy()
        data_copy = data_copy[
            data_copy[self._user_idx].isin(set(self.namespaces[self._user_idx].item_to_idx))
            & data_copy[self._item_idx].isin(set(self.namespaces[self._item_idx].item_to_idx))
        ]
        for attr in self.namespaces:
            func = self.encode if not decode else self.decode
            data_copy[attr] = func(data_copy[attr].values, namespace=attr)
        return data_copy


def get_count(data: pd.DataFrame, column: str) -> pd.DataFrame:
    grouped_by_column = data.groupby(column, as_index=False)
    return grouped_by_column.size()


def _filter_ratings(
    data: pd.DataFrame,
    user_idx: str,
    item_idx: str,
    min_user_count: int = 3,
    min_item_count: int = 3,
) -> pd.DataFrame:
    if min_item_count > 0:
        itemcount = get_count(data, column=item_idx)
        data = data.loc[
            data[item_idx].isin(itemcount[item_idx][itemcount["size"] >= min_item_count])
        ]
    if min_user_count > 0:
        usercount = get_count(data, column=user_idx)
        data = data.loc[
            data[user_idx].isin(usercount[user_idx][usercount["size"] >= min_user_count])
        ]
    return data


def filter_ratings(
    data: pd.DataFrame,
    user_idx: str,
    item_idx: str,
    min_user_count: int = 3,
    min_item_count: int = 3,
) -> pd.DataFrame:
    while True:
        cur_num_rows = data.shape[0]
        data = _filter_ratings(
            data,
            user_idx=user_idx,
            item_idx=item_idx,
            min_user_count=min_user_count,
            min_item_count=min_item_count,
        )
        new_num_rows = data.shape[0]
        if cur_num_rows == new_num_rows:
            break
    return data


@click.command(
    help="Split Yelp dataset",
    context_settings={"help_option_names": ["-h", "--help"]},
)
@click.argument("dataset_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument(
    "dst_dir", type=click.Path(exists=False, file_okay=False, dir_okay=True, path_type=Path)
)
@click.option(
    "--rating-cutoff",
    type=click.FLOAT,
    default=3.5,
    show_default=True,
)
@click.option(
    "--user-idx",
    type=click.STRING,
    default="user",
    show_default=True,
)
@click.option(
    "--item-idx",
    type=click.STRING,
    default="item",
    show_default=True,
)
@click.option(
    "--value-idx",
    type=click.STRING,
    default="value",
    show_default=True,
)
@click.option(
    "--date-idx",
    type=click.STRING,
    default="date",
    show_default=True,
)
@click.option(
    "--test-days",
    type=click.INT,
    default=3 * 365,
    show_default=True,
)
@click.option(
    "--eval-days",
    type=click.INT,
    default=365,
    show_default=True,
)
@click.option(
    "--min-user-count",
    type=click.INT,
    default=3,
    show_default=True,
)
@click.option(
    "--min-item-count",
    type=click.INT,
    default=3,
    show_default=True,
)
@click.option("--drop-duplicates", is_flag=True, default=False, show_default=True)
def main(
    dataset_path: Path,
    dst_dir: Path,
    rating_cutoff: float = 3.5,
    user_idx: str = "user",
    item_idx: str = "item",
    value_idx: str = "value",
    date_idx: str = "date",
    test_days: int = 3 * 365,
    eval_days: int = 365,
    min_user_count: int = 3,
    min_item_count: int = 3,
    drop_duplicates: bool = False,
) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(dataset_path)
    data = data[data[value_idx] > rating_cutoff]
    if drop_duplicates:
        data.drop_duplicates(subset=[user_idx, item_idx], keep="last", inplace=True)
    try:
        data[date_idx] = pd.to_datetime(data[date_idx], format="%Y-%m-%d %H:%M:%S")
    except ValueError:
        data[date_idx] = pd.to_datetime(data[date_idx], unit="s")
    print(f"min date: {data[date_idx].min()}", f"max date: {data[date_idx].max()}")
    data = filter_ratings(
        data,
        user_idx=user_idx,
        item_idx=item_idx,
        min_user_count=min_user_count,
        min_item_count=min_item_count,
    )
    splits = {
        "test": (test_split := data[date_idx].max() - pd.Timedelta(test_days, "days")),
        "eval": test_split - pd.Timedelta(eval_days, "days"),
    }
    datasets = {
        "full_train": (full_train := data.loc[data[date_idx] <= splits["test"]].copy()),
        "train": full_train.loc[full_train[date_idx] <= splits["eval"]].copy(),
        "eval": full_train.loc[full_train[date_idx] > splits["eval"]].copy(),
        "test": data.loc[data[date_idx] > splits["test"]].copy(),
    }
    for d_key in ("full_train", "train"):
        datasets[d_key] = filter_ratings(
            datasets[d_key],
            user_idx=user_idx,
            item_idx=item_idx,
            min_user_count=min_user_count,
            min_item_count=min_item_count,
        )
        print(
            f"users: {datasets[d_key][user_idx].unique().shape}",
            f"items: {datasets[d_key][item_idx].unique().shape}",
        )
    encoder = Encoder(user_idx=user_idx, item_idx=item_idx).fit(datasets["full_train"])
    for d_key, d in datasets.items():
        datasets[d_key] = encoder.transform(d)
    valid_users = {
        "full_train": (
            set(datasets["full_train"][user_idx])
            | (set(datasets["full_train"][user_idx]) & set(datasets["test"][user_idx]))
        ),
        "train": (
            set(datasets["train"][user_idx])
            | (set(datasets["train"][user_idx]) & set(datasets["eval"][user_idx]))
        ),
        "eval": (
            set(datasets["train"][user_idx])
            | (set(datasets["train"][user_idx]) & set(datasets["eval"][user_idx]))
        ),
        "test": (
            set(datasets["full_train"][user_idx])
            | (set(datasets["full_train"][user_idx]) & set(datasets["test"][user_idx]))
        ),
    }
    valid_items = {
        "full_train": (
            set(datasets["full_train"][item_idx])
            | (set(datasets["full_train"][item_idx]) & set(datasets["test"][item_idx]))
        ),
        "train": (
            set(datasets["train"][item_idx])
            | (set(datasets["train"][item_idx]) & set(datasets["eval"][item_idx]))
        ),
        "eval": (
            set(datasets["train"][item_idx])
            | (set(datasets["train"][item_idx]) & set(datasets["eval"][item_idx]))
        ),
        "test": (
            set(datasets["full_train"][item_idx])
            | (set(datasets["full_train"][item_idx]) & set(datasets["test"][item_idx]))
        ),
    }
    for d_key, d in datasets.items():
        datasets[d_key] = d.loc[
            d[user_idx].isin(valid_users[d_key]) & d[item_idx].isin(valid_items[d_key])
        ]
        datasets[d_key].rename(columns={user_idx: "user", item_idx: "item"}).loc[
            :, ["user", "item"]
        ].to_csv(dst_dir / f"{d_key}.csv", index=False)


if __name__ == "__main__":
    main()
