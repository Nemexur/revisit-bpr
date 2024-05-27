from pathlib import Path

import click
import numpy as np
import polars as pl

from experiments.encoder import AttrEncoder, JsonLEncoder

NUM_USERS = 10_000
NUM_MOVIES = 5_000


def pick_users_items(
    dataset: pl.LazyFrame, num_users: int, num_items: int, rng: np.random.Generator
) -> dict[str, set[int]]:
    users = dataset.select(pl.col("user")).unique(maintain_order=True).collect().to_dict()["user"]
    movies = (
        dataset.select(pl.col("movie")).unique(maintain_order=True).collect().to_dict()["movie"]
    )
    return {
        "user": set(rng.choice(users, size=num_users, replace=False).tolist()),
        "movie": set(rng.choice(movies, size=num_items, replace=False).tolist()),
    }


@click.command(
    help="Build datasets for BPR.",
    context_settings={"help_option_names": ["-h", "--help"]},
)
@click.argument("dataset_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("train_path", type=click.Path(exists=False, dir_okay=False, path_type=Path))
@click.argument("eval_path", type=click.Path(exists=False, dir_okay=False, path_type=Path))
@click.option(
    "--encoders-dir",
    type=click.Path(exists=False, dir_okay=True, file_okay=False, path_type=Path),
    default=None,
    show_default=True,
)
@click.option("--seed", type=click.INT, default=13, show_default=True)
def main(
    dataset_path: Path,
    train_path: Path,
    eval_path: Path,
    encoders_dir: Path | None = None,
    seed: int = 13,
) -> None:
    rng = np.random.default_rng(seed=seed)
    q = pl.scan_csv(dataset_path).select(
        pl.col("user"),
        pl.col("movie"),
        pl.col("rating"),
        pl.col("date").str.strptime(pl.Date, "%Y-%m-%d"),
    )
    valid_users_movies = pick_users_items(q, num_users=NUM_USERS, num_items=NUM_MOVIES, rng=rng)
    q = q.filter(
        pl.col("user").is_in(valid_users_movies["user"])
        & pl.col("movie").is_in(valid_users_movies["movie"])
    )
    # Build encoders
    enc = JsonLEncoder(attrs={"user": AttrEncoder(), "movie": AttrEncoder()})
    for e in enc.attrs.values():
        e.add(0)
    enc.fit(
        q.with_columns(pl.col("user").cast(pl.Utf8), pl.col("movie").cast(pl.Utf8))
        .collect()
        .to_dicts()
    )
    if encoders_dir is not None:
        encoders_dir.mkdir(exist_ok=True)
        enc.save(encoders_dir)
    q = (
        q.with_columns(
            user=pl.col("user").map_elements(lambda x: enc.attrs["user"].encode(x)),
            movie=pl.col("movie").map_elements(lambda x: enc.attrs["movie"].encode(x)),
        )
        .group_by(by="user")
        .agg(
            pl.col("movie").sort_by("date"),
            pl.col("rating").sort_by("date"),
            pl.col("date").sort_by("date"),
        )
        .with_columns(pl.col("rating").list.len().alias("length"))
        .sort(by="user")
        .with_columns(
            pl.col("movie").alias("seen_movies"),
            pl.col("length").map_elements(lambda x: int(rng.integers(0, x))).alias("test"),
            pl.col("length")
            .map_elements(
                lambda x: np.arange(x).tolist(), return_dtype=pl.List(pl.Int64)  # pyright: ignore
            )
            .alias("index"),
        )
    )
    dataset = q.collect()
    train_dataset = (
        dataset.explode(["movie", "rating", "date", "index"])
        .select(
            pl.col("user"),
            pl.col("movie"),
            pl.col("rating"),
            pl.col("date"),
            pl.col("index"),
            pl.col("length"),
            pl.col("test"),
            pl.col("seen_movies"),
        )
        .filter(pl.col("test") != pl.col("index"))
        .clone()
    )
    train_dataset.select(
        pl.col("user"), pl.col("movie").alias("item"), pl.col("seen_movies").alias("seen_items")
    ).write_ndjson(train_path)
    dataset.select(
        pl.col("user"), pl.col("test").alias("item"), pl.col("movie").alias("seen_items")
    ).write_ndjson(eval_path)


if __name__ == "__main__":
    main()
