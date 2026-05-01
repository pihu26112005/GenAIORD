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

METRIC = "AUPRC"


def repo_root_from_script() -> Path:
    """
    Script location:
    root/scripts/data-analysis/generate_auprc_strategy_plots.py

    parents[0] = data-analysis
    parents[1] = scripts
    parents[2] = root
    """
    return Path(__file__).resolve().parents[2]


def find_combined_csv(dataset_dir: Path, pattern: str) -> Path:
    candidates = list(dataset_dir.glob(pattern))

    if not candidates:
        raise FileNotFoundError(
            f"No combined CSV found inside: {dataset_dir}\n"
            f"Expected pattern: {pattern}"
        )

    # If multiple matching files exist, use the latest modified one.
    candidates = sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def extract_gamma(model_name: str):
    match = re.search(r"Gamma_([0-9.]+)", str(model_name))
    if not match:
        return None
    return float(match.group(1))


def sort_strategies(strategies: list[str]) -> list[str]:
    known = [s for s in STRATEGY_ORDER if s in strategies]
    unknown = sorted([s for s in strategies if s not in STRATEGY_ORDER])
    return known + unknown


def set_tight_y_axis(ax, values: pd.Series):
    """
    AUPRC values are often very close, e.g. 0.901 vs 0.896.
    So we zoom the y-axis around the local min/max instead of using 0 to 1.
    """
    values = values.dropna()

    if values.empty:
        return

    min_val = values.min()
    max_val = values.max()
    value_range = max_val - min_val

    # For very close values, force a small but visible padding.
    if value_range == 0:
        padding = max(0.001, abs(max_val) * 0.002)
    else:
        padding = max(value_range * 0.25, 0.001)

    lower = max(0, min_val - padding)
    upper = min(1, max_val + padding)

    # Avoid invalid axis if values are extremely close.
    if upper <= lower:
        upper = lower + 0.002

    ax.set_ylim(lower, upper)


def plot_dataset_auprc(
    dataset_name: str,
    csv_path: Path,
    output_dir: Path,
    dpi: int,
):
    df = pd.read_csv(csv_path)

    required_cols = {"Model", "Strat", METRIC}
    missing = required_cols - set(df.columns)

    if missing:
        raise ValueError(
            f"{csv_path} is missing required columns: {sorted(missing)}"
        )

    # Keep only Proposed CFG rows.
    df = df[df["Model"].astype(str).str.startswith("Proposed_CFG")].copy()

    if df.empty:
        print(f"[SKIP] No Proposed_CFG rows found in {csv_path}")
        return None

    df["gamma"] = df["Model"].apply(extract_gamma)
    df[METRIC] = pd.to_numeric(df[METRIC], errors="coerce")

    df = df.dropna(subset=["gamma", METRIC])

    if df.empty:
        print(f"[SKIP] No valid gamma/{METRIC} rows found in {csv_path}")
        return None

    strategies = sort_strategies(df["Strat"].dropna().unique().tolist())

    # Keep only three expected strategies if present.
    strategies = [s for s in STRATEGY_ORDER if s in strategies]

    if not strategies:
        print(f"[SKIP] No expected strategies found in {csv_path}")
        return None

    # Bigger figure: three large plots in one image.
    fig, axes = plt.subplots(
        nrows=3,
        ncols=1,
        figsize=(18, 18),
        sharex=False,
    )

    if len(strategies) == 1:
        axes = [axes]

    for ax, strategy in zip(axes, strategies):
        sub = (
            df[df["Strat"] == strategy][["gamma", METRIC]]
            .groupby("gamma", as_index=False)[METRIC]
            .mean()
            .sort_values("gamma")
        )

        if sub.empty:
            ax.set_visible(False)
            continue

        x_labels = [f"{g:.2f}" for g in sub["gamma"]]

        ax.bar(x_labels, sub[METRIC])

        ax.set_title(
            f"{dataset_name.upper()} | {strategy} | {METRIC} vs Proposed CFG Gamma",
            fontsize=15,
            pad=12,
        )

        ax.set_xlabel("Gamma", fontsize=12)
        ax.set_ylabel(METRIC, fontsize=12)

        ax.tick_params(axis="x", rotation=60, labelsize=10)
        ax.tick_params(axis="y", labelsize=10)

        ax.grid(axis="y", alpha=0.3)

        # Important: zoom y-axis to make close AUPRC differences visible.
        set_tight_y_axis(ax, sub[METRIC])

        # Add min/max text for quick reading.
        best_row = sub.loc[sub[METRIC].idxmax()]
        worst_row = sub.loc[sub[METRIC].idxmin()]

        ax.text(
            0.01,
            0.95,
            f"Best: γ={best_row['gamma']:.2f}, {METRIC}={best_row[METRIC]:.6f}\n"
            f"Worst: γ={worst_row['gamma']:.2f}, {METRIC}={worst_row[METRIC]:.6f}",
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", alpha=0.15),
        )

    fig.suptitle(
        f"{dataset_name.upper()} — {METRIC} variation across Proposed CFG gamma values",
        fontsize=20,
        y=0.995,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.98])

    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{METRIC}.png"
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return output_path


def process_dataset(
    root: Path,
    dataset_name: str,
    input_glob: str,
    output_root: Path,
    dpi: int,
):
    dataset_dir = root / "results" / dataset_name

    if not dataset_dir.exists():
        print(f"[SKIP] Dataset folder not found: {dataset_dir}")
        return None

    csv_path = find_combined_csv(dataset_dir, input_glob)

    print(f"[INFO] {dataset_name}: using input file: {csv_path}")

    output_dir = output_root / dataset_name

    output_path = plot_dataset_auprc(
        dataset_name=dataset_name,
        csv_path=csv_path,
        output_dir=output_dir,
        dpi=dpi,
    )

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate AUPRC plots for Proposed CFG models across strategies."
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
        help="CSV pattern to search inside each results/<dataset>/ folder.",
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
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

    # Important: folder name is plots, not plot.
    output_root = root / "plots" / "data-analysis"

    print(f"[INFO] Project root: {root}")
    print(f"[INFO] Output root: {output_root}")

    generated = []

    for dataset_name in args.datasets:
        try:
            output_path = process_dataset(
                root=root,
                dataset_name=dataset_name,
                input_glob=args.input_glob,
                output_root=output_root,
                dpi=args.dpi,
            )

            if output_path:
                generated.append(
                    {
                        "dataset": dataset_name,
                        "metric": METRIC,
                        "plot_path": str(output_path),
                    }
                )
                print(f"[DONE] {dataset_name}: saved {output_path}")

        except Exception as exc:
            print(
                f"[ERROR] Failed for dataset '{dataset_name}': {exc}",
                file=sys.stderr,
            )

    if generated:
        manifest = pd.DataFrame(generated)
        manifest_path = output_root / "auprc_plot_manifest.csv"
        manifest.to_csv(manifest_path, index=False)
        print(f"[DONE] Manifest saved at: {manifest_path}")

    print(f"[DONE] Total dataset plots generated: {len(generated)}")


if __name__ == "__main__":
    main()