"""Statistical analysis: paired bootstrap confidence intervals and condition comparisons."""

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd


@dataclass
class BootstrapCI:
    """Represents a paired bootstrap confidence interval for a metric difference."""
    mean_diff: float
    ci_lower: float
    ci_upper: float
    confidence_level: float
    iterations: int
    is_significant: bool  # True if 0 is not contained within [ci_lower, ci_upper]

    def to_dict(self) -> Dict[str, float | bool | int]:
        return {
            "mean_diff": round(self.mean_diff, 4),
            "ci_lower": round(self.ci_lower, 4),
            "ci_upper": round(self.ci_upper, 4),
            "confidence_level": self.confidence_level,
            "iterations": self.iterations,
            "is_significant": self.is_significant,
        }


def compute_paired_bootstrap_ci(
    scores_treatment: List[float],
    scores_control: List[float],
    iterations: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42
) -> BootstrapCI:
    """Compute non-parametric paired bootstrap confidence interval for (treatment - control).

    Pairs are matched by question across the identical set of items.
    """
    if len(scores_treatment) != len(scores_control):
        raise ValueError(
            f"Paired bootstrap requires equal sample lengths: "
            f"{len(scores_treatment)} != {len(scores_control)}"
        )
    n = len(scores_treatment)
    if n == 0:
        return BootstrapCI(0.0, 0.0, 0.0, confidence_level, iterations, False)

    diffs = np.array(scores_treatment) - np.array(scores_control)
    mean_diff = float(np.mean(diffs))

    rng = np.random.default_rng(seed)
    boot_means = np.empty(iterations)

    for i in range(iterations):
        resampled_indices = rng.integers(0, n, size=n)
        boot_means[i] = np.mean(diffs[resampled_indices])

    alpha = (1.0 - confidence_level) / 2.0
    ci_lower = float(np.percentile(boot_means, alpha * 100))
    ci_upper = float(np.percentile(boot_means, (1.0 - alpha) * 100))

    # Significant at (1-alpha) level if the confidence interval does not overlap 0
    is_sig = (ci_lower > 0) or (ci_upper < 0)

    return BootstrapCI(
        mean_diff=mean_diff,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        confidence_level=confidence_level,
        iterations=iterations,
        is_significant=is_sig,
    )


def summarize_condition_metrics(df_predictions: pd.DataFrame) -> pd.DataFrame:
    """Aggregate question-level predictions into condition summary statistics."""
    grouped = df_predictions.groupby("condition_id")
    records = []

    for cond_id, grp in grouped:
        n_total = len(grp)
        n_valid = len(grp[grp["generation_status"] == "success"])
        mean_f1 = float(grp["f1"].mean())
        em_pct = float(grp["exact_match"].mean()) * 100.0
        distractor_count = grp["distractor_count"].iloc[0]
        distractor_type = grp["distractor_type"].iloc[0]
        reranking = grp["reranking_method"].iloc[0]
        experiment_block = grp["experiment_block"].iloc[0]

        records.append({
            "condition_id": cond_id,
            "experiment_block": experiment_block,
            "distractor_count": distractor_count,
            "distractor_type": distractor_type,
            "reranking_method": reranking,
            "n_samples": n_total,
            "valid_samples": n_valid,
            "mean_f1": round(mean_f1, 4),
            "exact_match_pct": round(em_pct, 2),
        })

    summary_df = pd.DataFrame(records)
    # Sort for consistent display: Block A/B/C/D then distractor count
    if not summary_df.empty:
        summary_df.sort_values(by=["experiment_block", "distractor_count", "reranking_method"], inplace=True)
    return summary_df
