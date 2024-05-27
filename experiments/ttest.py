from collections import defaultdict
from pathlib import Path

import click
import pandas as pd
from scipy.stats import ttest_ind
from tqdm import tqdm


@click.command(
    help="Perform ttest compairing study metrics",
    context_settings={"help_option_names": ["-h", "--help"]},
)
@click.option(
    "--first",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    required=True,
)
@click.option(
    "--second",
    type=click.Path(exists=True, dir_okay=True, file_okay=False, path_type=Path),
    required=True,
)
@click.option(
    "-m",
    "--metrics",
    type=click.STRING,
    multiple=True,
    default=["ndcg@5", "ndcg@10", "ndcg@100", "recall@5", "recall@10", "recall@100"],
)
def main(first: Path, second: Path, metrics: list[str]) -> None:
    results = defaultdict(list, {"metrics": metrics})
    first_df, second_df = (
        pd.read_json(first, orient="records"),
        pd.read_json(second, orient="records"),
    )
    alphas = [0.05, 0.01, 0.001]
    for m in tqdm(metrics, desc="Testing"):
        first_metric, second_metric = (
            first_df.loc[:, ["user", m]].copy().drop_duplicates("user", keep="last"),
            second_df.loc[:, ["user", m]].copy().drop_duplicates("user", keep="last"),
        )
        metric_df = pd.merge(
            first_metric, second_metric, on="user", how="inner", suffixes=("_first", "_second")
        )
        first_mean, second_mean = first_metric[m].mean(), second_metric[m].mean()
        t, pval = ttest_ind(metric_df[f"{m}_second"].values, metric_df[f"{m}_first"].values)
        results["t"].append(t)
        results["pval"].append(pval)
        results["pval_bonferoni"].append(pval * len(metrics))
        results["first_mean"].append(first_mean)
        results["second_mean"].append(second_mean)
        results["diff"].append(abs(second_mean - first_mean))
        results["diff_percentage"].append(abs(second_mean - first_mean) * 100 / first_mean)
        for a in alphas:
            results[f"diff_{a}"].append(results["pval"][-1] < a)
            results[f"diff_{a}_bonferoni"].append(results["pval_bonferoni"][-1] < a)
    results_df = pd.DataFrame(results)
    with pd.option_context(
        "display.max_rows", None, "display.max_columns", None, "display.expand_frame_repr", False
    ):
        print(results_df)


if __name__ == "__main__":
    main()
