from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from scipy.stats import binomtest, wilcoxon
except Exception:
    binomtest = None
    wilcoxon = None


# ------------------------------------------------------------
# Loading helpers
# ------------------------------------------------------------
def load_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(Path(path))


def normalize_bool_col(series: pd.Series) -> pd.Series:
    return series.fillna(0).astype(int)


# ------------------------------------------------------------
# Metrics
# ------------------------------------------------------------
def compute_metrics(df: pd.DataFrame) -> dict[str, Any]:
    n = len(df)
    if n == 0:
        return {}

    clean_correct = normalize_bool_col(df["clean_correct"]).sum()
    adv_correct = normalize_bool_col(df["adversarial_correct"]).sum()
    flips = normalize_bool_col(df["decision_flipped"]).sum()

    clean_fallbacks = normalize_bool_col(df["clean_fallback_used"]).sum()
    adv_fallbacks = normalize_bool_col(df["adv_fallback_used"]).sum()

    clean_unsafe = normalize_bool_col(df["clean_unsafe"]).sum() if "clean_unsafe" in df else 0
    adv_unsafe = normalize_bool_col(df["adv_unsafe"]).sum() if "adv_unsafe" in df else 0

    clean_latency_median = float(df["clean_latency_ms"].median()) if "clean_latency_ms" in df else 0.0
    adv_latency_median = float(df["adv_latency_ms"].median()) if "adv_latency_ms" in df else 0.0
    overall_latency_median = float(
        pd.concat([df["clean_latency_ms"], df["adv_latency_ms"]]).median()
    ) if "clean_latency_ms" in df and "adv_latency_ms" in df else 0.0

    return {
        "n": n,
        "clean_accuracy": clean_correct / n,
        "adversarial_accuracy": adv_correct / n,
        "aer": (n - adv_correct) / n,
        "pafr": flips / n,
        "clean_fallback_rate": clean_fallbacks / n,
        "adv_fallback_rate": adv_fallbacks / n,
        "safe_reject_rate": (clean_fallbacks + adv_fallbacks) / (2 * n),
        "clean_htr": clean_unsafe / n,
        "adv_htr": adv_unsafe / n,
        "overall_htr": (clean_unsafe + adv_unsafe) / (2 * n),
        "median_clean_latency_ms": clean_latency_median,
        "median_adv_latency_ms": adv_latency_median,
        "median_latency_ms": overall_latency_median,
    }


# ------------------------------------------------------------
# Bootstrap confidence intervals
# ------------------------------------------------------------
def bootstrap_metric(
    df: pd.DataFrame,
    metric_name: str,
    n_boot: int = 2000,
    seed: int = 0,
) -> tuple[float, float, float]:
    if n_boot < 40:
        raise ValueError(f"n_boot must be at least 40 for reliable CI estimation, got {n_boot}")
    
    rng = random.Random(seed)
    n = len(df)
    if n == 0:
        return (0.0, 0.0, 0.0)

    point = compute_metrics(df)[metric_name]
    samples = []

    rows = list(range(n))
    for _ in range(n_boot):
        sampled_idx = [rng.choice(rows) for _ in range(n)]
        sample_df = df.iloc[sampled_idx]
        samples.append(compute_metrics(sample_df)[metric_name])

    samples.sort()
    lo = samples[int(0.025 * len(samples))]
    hi = samples[int(0.975 * len(samples))]
    return (point, lo, hi)


# ------------------------------------------------------------
# Pairwise significance tests
# ------------------------------------------------------------
def mcnemar_from_frames(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    outcome_col: str,
) -> dict[str, Any]:
    merged = df_a[["id", outcome_col]].merge(
        df_b[["id", outcome_col]],
        on="id",
        suffixes=("_a", "_b"),
    )

    if len(merged) == 0:
        import warnings
        warnings.warn(f"No matching IDs found between dataframes for outcome column '{outcome_col}'")
        return {
            "b01": 0,
            "b10": 0,
            "p_value": None,
            "test": "mcnemar_exact_no_matches",
        }

    a = normalize_bool_col(merged[f"{outcome_col}_a"])
    b = normalize_bool_col(merged[f"{outcome_col}_b"])

    b01 = int(((a == 0) & (b == 1)).sum())
    b10 = int(((a == 1) & (b == 0)).sum())
    discordant = b01 + b10

    if discordant == 0:
        return {
            "b01": b01,
            "b10": b10,
            "p_value": 1.0,
            "test": "mcnemar_exact",
        }

    if binomtest is not None:
        p_value = binomtest(min(b01, b10), n=discordant, p=0.5, alternative="two-sided").pvalue
    else:
        p_value = None

    return {
        "b01": b01,
        "b10": b10,
        "p_value": p_value,
        "test": "mcnemar_exact" if binomtest is not None else "mcnemar_exact_unavailable",
    }


def wilcoxon_latency_test(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    latency_col: str = "adv_latency_ms",
) -> dict[str, Any]:
    merged = df_a[["id", latency_col]].merge(
        df_b[["id", latency_col]],
        on="id",
        suffixes=("_a", "_b"),
    )

    x = merged[f"{latency_col}_a"]
    y = merged[f"{latency_col}_b"]

    if wilcoxon is None:
        return {
            "test": "wilcoxon_unavailable",
            "p_value": None,
        }

    try:
        result = wilcoxon(x, y, alternative="two-sided")
        return {
            "test": "wilcoxon_signed_rank",
            "p_value": float(result.pvalue),
        }
    except Exception:
        return {
            "test": "wilcoxon_failed",
            "p_value": None,
        }


# ------------------------------------------------------------
# Reporting helpers
# ------------------------------------------------------------
def build_system_label(df: pd.DataFrame) -> str:
    system = df["system"].iloc[0]
    model = df["model"].iloc[0]
    return f"{system} | {model}"


def metrics_with_ci(df: pd.DataFrame) -> dict[str, Any]:
    metrics = compute_metrics(df)

    for metric_name in [
        "clean_accuracy",
        "adversarial_accuracy",
        "aer",
        "pafr",
        "safe_reject_rate",
        "clean_htr",
        "adv_htr",
        "overall_htr",
    ]:
        point, lo, hi = bootstrap_metric(df, metric_name)
        metrics[f"{metric_name}_ci_low"] = lo
        metrics[f"{metric_name}_ci_high"] = hi

    return metrics


# ------------------------------------------------------------
# Main analysis
# ------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze robot result CSVs.")
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="One or more per-system CSV files from run_robot_all_systems.py",
    )
    parser.add_argument(
        "--compare-to",
        default=None,
        help='Optional system label to compare all other systems against, e.g. "abci | Qwen/Qwen3-1.7B"',
    )
    parser.add_argument(
        "--output-prefix",
        default="robot_analysis",
        help="Prefix for generated analysis files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    frames: list[pd.DataFrame] = []
    for path in args.inputs:
        df = load_csv(path)
        frames.append(df)

    summary_rows = []
    by_label: dict[str, pd.DataFrame] = {}

    for df in frames:
        label = build_system_label(df)
        by_label[label] = df
        row = {
            "label": label,
            **metrics_with_ci(df),
        }
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows).sort_values("label")
    summary_csv = Path(f"{args.output_prefix}_summary.csv")
    summary_df.to_csv(summary_csv, index=False)

    comparison_rows = []
    if args.compare_to is not None and args.compare_to in by_label:
        reference_df = by_label[args.compare_to]

        for label, df in by_label.items():
            if label == args.compare_to:
                continue

            mc_adv_correct = mcnemar_from_frames(reference_df, df, "adversarial_correct")
            mc_flip = mcnemar_from_frames(reference_df, df, "decision_flipped")
            mc_htr = mcnemar_from_frames(reference_df, df, "adv_unsafe")
            wilcox = wilcoxon_latency_test(reference_df, df, latency_col="adv_latency_ms")

            comparison_rows.append({
                "reference": args.compare_to,
                "other": label,
                "adv_correct_b01": mc_adv_correct["b01"],
                "adv_correct_b10": mc_adv_correct["b10"],
                "adv_correct_p_value": mc_adv_correct["p_value"],
                "flip_b01": mc_flip["b01"],
                "flip_b10": mc_flip["b10"],
                "flip_p_value": mc_flip["p_value"],
                "adv_htr_b01": mc_htr["b01"],
                "adv_htr_b10": mc_htr["b10"],
                "adv_htr_p_value": mc_htr["p_value"],
                "latency_p_value": wilcox["p_value"],
            })

    comparisons_df = pd.DataFrame(comparison_rows)
    comparisons_csv = Path(f"{args.output_prefix}_comparisons.csv")
    comparisons_df.to_csv(comparisons_csv, index=False)

    print(f"Saved summary to: {summary_csv}")
    print(f"Saved comparisons to: {comparisons_csv}")


if __name__ == "__main__":
    main()
