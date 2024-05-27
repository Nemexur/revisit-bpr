from typing import Iterable
from datetime import datetime
from pathlib import Path

import click
import polars as pl
from tqdm import tqdm

from experiments.encoder import JsonLEncoder
from experiments.types import DictParamType

DTTM_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


def chain_when_then(chain: Iterable[tuple[pl.Expr, pl.Expr]], otherwise: pl.Expr) -> pl.Expr:
    expr = pl.when(pl.lit(False)).then(None)
    for condition, value in iter(chain):
        expr = expr.when(condition).then(value)
    expr = expr.otherwise(otherwise)
    return expr


@click.command(
    help="Build evaluation dataset",
    context_settings={"help_option_names": ["-h", "--help"]},
)
@click.argument("dataset_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument(
    "dst", type=click.Path(exists=False, file_okay=False, dir_okay=True, path_type=Path)
)
@click.option(
    "--encoders-dir",
    type=click.Path(exists=False, dir_okay=True, file_okay=False, path_type=Path),
    default=None,
    show_default=True,
)
@click.option(
    "--dates",
    type=DictParamType(),
    help="Time based split. Format: train=2000-01-01T13:00:00Z,eval=2000-01-01T14:00:00Z,...",
)
def main(
    dataset_path: Path,
    dst: Path,
    encoders_dir: Path | None = None,
    dates: dict[str, str] | None = None,
) -> None:
    dst.mkdir(exist_ok=True)
    if dates is None:
        dates = {"train": datetime.max.strftime(DTTM_FORMAT)}
    q = pl.scan_csv(dataset_path).select(
        pl.col("user"),
        pl.col("item"),
        pl.col("rating"),
        pl.col("date").str.strptime(pl.Datetime, DTTM_FORMAT),
    )
    sorted_dates = sorted(
        {k: datetime.strptime(d, DTTM_FORMAT) for k, d in dates.items()}.items(),
        key=lambda x: x[1],
    )
    q = q.with_columns(
        chain_when_then(
            [(pl.col("date") <= d, pl.lit(key)) for key, d in sorted_dates],
            otherwise=pl.lit("test"),
        ).alias("part"),
    )
    print(
        q.group_by(by="part")
        .agg(pl.col("date").min().alias("min"), pl.col("date").max().alias("max"))
        .collect()
    )
    # Build encoders
    enc = JsonLEncoder()
    for e in enc.attrs.values():
        e.add(0)
    enc.fit(
        tqdm(
            q.with_columns(pl.col("user").cast(pl.Utf8), pl.col("item").cast(pl.Utf8))
            .select("user", "item")
            .collect()
            .to_dicts(),
            desc="Building encoder",
        )
    )
    if encoders_dir is not None:
        encoders_dir.mkdir(exist_ok=True)
        enc.save(encoders_dir)
    q = q.with_columns(
        user=pl.col("user").map_elements(lambda x: enc.attrs["user"].encode(x)),
        item=pl.col("item").map_elements(lambda x: enc.attrs["item"].encode(x)),
    )
    # Save seen_items by user
    q.group_by(by="user").agg(
        pl.col("item").sort_by("date").alias("seen_items")
    ).collect().write_ndjson(dst / "user-seen-items.jsonl")
    dataset = q.collect()
    for part in dataset.select("part").unique().to_numpy().reshape(-1):
        dataset_part = dataset.filter(pl.col("part") == part)
        dataset_part.select("user", "item").write_ndjson(dst / f"{part}.jsonl")
        dataset_part.group_by(by="user").agg(pl.col("item").sort_by(by="date")).write_ndjson(
            dst / f"{part}-grouped.jsonl"
        )


if __name__ == "__main__":
    main()
