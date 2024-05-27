from io import BytesIO
from pathlib import Path

import click
import polars as pl


def get_valid(dataset: pl.DataFrame, group_col: str, count_col: str, min_count: int) -> set[int]:
    return set(
        dataset.group_by(group_col)
        .agg(pl.col(count_col).len())
        .filter(pl.col(count_col) > min_count)
        .select(pl.col(group_col))
        .to_numpy()
        .reshape(-1)
        .tolist()
    )


@click.command(
    help="Cut off users and items lower than min.",
    context_settings={"help_option_names": ["-h", "--help"]},
)
@click.argument("dataset_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "-o",
    "--out",
    type=click.File("wb", encoding="utf-8"),
    help="Output file. By default prints to stdout.",
    default="-",
)
@click.option("--user-col", type=click.STRING, default="user", show_default=True)
@click.option("--item-col", type=click.STRING, default="item", show_default=True)
@click.option("--min-users", type=click.INT, default=5, show_default=True)
@click.option("--min-items", type=click.INT, default=5, show_default=True)
def main(
    dataset_path: Path,
    out: BytesIO,
    min_users: int = 5,
    min_items: int = 5,
    user_col: str = "user",
    item_col: str = "item",
) -> None:
    dataset = pl.read_csv(dataset_path)
    num_samples = dataset.shape[0]
    while True:
        valid_users = get_valid(
            dataset, group_col=user_col, count_col=item_col, min_count=min_items
        )
        dataset = dataset.filter(pl.col(user_col).is_in(valid_users))
        valid_items = get_valid(
            dataset, group_col=item_col, count_col=user_col, min_count=min_users
        )
        dataset = dataset.filter(pl.col(item_col).is_in(valid_items))
        if num_samples == dataset.shape[0]:
            break
        num_samples = dataset.shape[0]
    dataset.write_csv(out)


if __name__ == "__main__":
    main()
