from pathlib import Path
import argparse
import re
import sys

import pandas as pd
import matplotlib.pyplot as plt


DATASETS = ["adult", "heloc", "fintech"]

STRATEGY_ORDER = [
    "S1_Balanced",
    "S2_SynMaj",
    "S3_Overlap",
]

# Only metrics you currently want
METRICS = [
    "XG_Acc",
    "Min_Acc",
    "Macro_F1",
    "DCR_5th_perc",
    "NNDR_5th_perc",
    "Pairwise_Corr_Error",
]


def repo_root_from_script() -> Path:
    """
    root/scripts/data-analysis/generate_metric_strategy_plots.py
    """
    return Path(__file__).resolve().parents[2]


def find_combined_csv(dataset_dir: Path, pattern: str) -> Path:
    candidates = list(dataset_dir.glob(pattern))
    if not candidates:
        raise FileNotFoundError(
            f"No combined CSV found inside: {dataset_dir}\n"
            f"Expected pattern: {pattern}"
        )
    candidates = sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def extract_gamma(model_name: str):
    match = re.search(r"Gamma_([0-9.]+)", str(model_name))
    if not match:
        return None
    return float(match.group(1))


def sort_strategies(strategies):
    known = [s for s in STRATEGY_ORDER if s in strategies]
    unknown = sorted([s for s in strategies if s not in STRATEGY_ORDER])
    return known + unknown


def metric_precision(metric: str) -> int:
    """
    Controls value-label precision shown on bars.
    """
    if metric in ["XG_Acc", "Min_Acc"]:
        return 3
    if metric in ["Macro_F1", "DCR_5th_perc", "NNDR_5th_perc", "Pairwise_Corr_Error"]:
        return 6
    return 4


def metric_padding(metric: str, min_val: float, max_val: float) -> float:
    """
    Choose y-axis padding based on metric scale.
    This is the key thing that makes the plots readable.
    """
    value_range = max_val - min_val

    # Accuracy metrics in ~70-90 range
    if metric in ["XG_Acc", "Min_Acc"]:
        return max(0.25, value_range * 0.20)

    # Macro_F1 and privacy decimal metrics: very tight range
    if metric in ["Macro_F1", "DCR_5th_perc", "NNDR_5th_perc"]:
        return max(0.0025, value_range * 0.25)

    # Correlation error often has small but meaningful differences
    if metric == "Pairwise_Corr_Error":
        return max(0.003, value_range * 0.20)

    return max(0.001, value_range * 0.20)


def set_metric_axis(ax, metric: str, values: pd.Series):
    values = values.dropna()
    if values.empty:
        return

    min_val = float(values.min())
    max_val = float(values.max())

    if min_val == max_val:
        pad = metric_padding(metric, min_val, max_val)
        lower = min_val - pad
        upper = max_val + pad
    else:
        pad = metric_padding(metric, min_val, max_val)
        lower = min_val - pad
        upper = max_val + pad

    # Some metrics should remain in logical bounds
    if metric in ["Macro_F1", "DCR_5th_perc", "NNDR_5th_perc", "Pairwise_Corr_Error"]:
        lower = max(0, lower)
        upper = min(1, upper) if metric != "Pairwise_Corr_Error" else upper

    ax.set_ylim(lower, upper)


def add_value_labels(ax, bars, metric: str):
    precision = metric_precision(metric)
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:.{precision}f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=90,
        )


def plot_metric_for_dataset(
    dataset_name: str,
    csv_path: Path,
    metric: str,
    output_dir: Path,
    dpi: int,
):
    df = pd.read_csv(csv_path)

    required_cols = {"Model", "Strat", metric}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"{csv_path} is missing required columns: {sorted(missing)}"
        )

    df = df[df["Model"].astype(str).str.startswith("Proposed_CFG")].copy()
    if df.empty:
        print(f"[SKIP] No Proposed_CFG rows found in {csv_path}")
        return None

    df["gamma"] = df["Model"].apply(extract_gamma)
    df[metric] = pd.to_numeric(df[metric], errors="coerce")

    df = df.dropna(subset=["gamma", metric])
    if df.empty:
        print(f"[SKIP] No valid gamma/{metric} rows found in {csv_path}")
        return None

    strategies = sort_strategies(df["Strat"].dropna().unique().tolist())
    strategies = [s for s in STRATEGY_ORDER if s in strategies]

    if not strategies:
        print(f"[SKIP] No expected strategies found in {csv_path}")
        return None

    fig, axes = plt.subplots(
        nrows=3,
        ncols=1,
        figsize=(20, 18),
        sharex=False,
    )

    if len(strategies) == 1:
        axes = [axes]

    for i, strategy in enumerate(strategies):
        ax = axes[i]

        sub = (
            df[df["Strat"] == strategy][["gamma", metric]]
            .groupby("gamma", as_index=False)[metric]
            .mean()
            .sort_values("gamma")
        )

        if sub.empty:
            ax.set_visible(False)
            continue

        x_labels = [f"{g:.2f}" for g in sub["gamma"]]
        bars = ax.bar(x_labels, sub[metric])

        ax.set_title(
            f"{dataset_name.upper()} | {strategy} | {metric} vs Gamma",
            fontsize=15,
            pad=12,
        )
        ax.set_xlabel("Gamma", fontsize=12)
        ax.set_ylabel(metric, fontsize=12)

        ax.tick_params(axis="x", rotation=60, labelsize=10)
        ax.tick_params(axis="y", labelsize=10)
        ax.grid(axis="y", alpha=0.3)

        set_metric_axis(ax, metric, sub[metric])
        add_value_labels(ax, bars, metric)

        best_row = sub.loc[sub[metric].idxmax()]
        worst_row = sub.loc[sub[metric].idxmin()]

        ax.text(
            0.01,
            0.97,
            f"Max: γ={best_row['gamma']:.2f}, value={best_row[metric]:.{metric_precision(metric)}f}\n"
            f"Min: γ={worst_row['gamma']:.2f}, value={worst_row[metric]:.{metric_precision(metric)}f}",
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", alpha=0.15),
        )

    fig.suptitle(
        f"{dataset_name.upper()} — {metric} across Proposed CFG gamma values",
        fontsize=20,
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.98])

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{metric}.png"
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return output_path


def process_dataset(root: Path, dataset_name: str, input_glob: str, output_root: Path, dpi: int):
    dataset_dir = root / "results" / dataset_name
    if not dataset_dir.exists():
        print(f"[SKIP] Dataset folder not found: {dataset_dir}")
        return []

    csv_path = find_combined_csv(dataset_dir, input_glob)
    print(f"[INFO] {dataset_name}: using input file: {csv_path}")

    dataset_output_dir = output_root / dataset_name
    generated = []

    for metric in METRICS:
        try:
            output_path = plot_metric_for_dataset(
                dataset_name=dataset_name,
                csv_path=csv_path,
                metric=metric,
                output_dir=dataset_output_dir,
                dpi=dpi,
            )
            if output_path:
                generated.append(
                    {
                        "dataset": dataset_name,
                        "metric": metric,
                        "plot_path": str(output_path),
                    }
                )
                print(f"[DONE] {dataset_name} | {metric}: saved {output_path}")

        except Exception as exc:
            print(
                f"[ERROR] Failed for dataset '{dataset_name}', metric '{metric}': {exc}",
                file=sys.stderr,
            )

    return generated


def main():
    parser = argparse.ArgumentParser(
        description="Generate per-metric strategy plots for Proposed CFG gamma values."
    )

    parser.add_argument(
        "--datasets",
        nargs="*",
        default=DATASETS,
        help="Dataset folders inside results/. Default: adult heloc fintech",
    )

    parser.add_argument(
        "--input-glob",
        default="*combined*.csv",
        help="Pattern to find combined CSV inside each dataset folder.",
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=220,
        help="Output image DPI.",
    )

    parser.add_argument(
        "--root",
        type=Path,
        default=repo_root_from_script(),
        help="Project root. Default: auto-detected from script location.",
    )

    args = parser.parse_args()

    root = args.root.resolve()
    output_root = root / "plots" / "data-analysis"

    print(f"[INFO] Project root: {root}")
    print(f"[INFO] Output root: {output_root}")

    all_generated = []

    for dataset_name in args.datasets:
        try:
            generated = process_dataset(
                root=root,
                dataset_name=dataset_name,
                input_glob=args.input_glob,
                output_root=output_root,
                dpi=args.dpi,
            )
            all_generated.extend(generated)

        except Exception as exc:
            print(f"[ERROR] Failed dataset '{dataset_name}': {exc}", file=sys.stderr)

    if all_generated:
        manifest = pd.DataFrame(all_generated)
        manifest_path = output_root / "metric_plot_manifest.csv"
        manifest.to_csv(manifest_path, index=False)
        print(f"[DONE] Manifest saved at: {manifest_path}")

    print(f"[DONE] Total plots generated: {len(all_generated)}")


if __name__ == "__main__":
    main()