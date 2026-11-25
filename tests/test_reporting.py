"""Unit tests for reporting, LaTeX formatting, and figures."""

import pandas as pd

from robustrag.reporting import ReportGenerator


def test_generate_latex_table_empty():
    empty_df = pd.DataFrame()
    latex = ReportGenerator.generate_latex_table(empty_df)
    assert "No experimental results available" in latex


def test_generate_latex_table_populated():
    df = pd.DataFrame([
        {
            "condition_id": "E0_gold_only",
            "reranking_method": "none",
            "distractor_count": 0,
            "distractor_type": "none",
            "mean_f1": 0.8523,
            "exact_match_pct": 75.0,
        }
    ])
    latex = ReportGenerator.generate_latex_table(df)
    assert r"\begin{table}" in latex
    assert "E0\\_gold\\_only" in latex
    assert "85.23" in latex
    assert "75.00" in latex


def test_generate_figures(tmp_path):
    df = pd.DataFrame([
        {
            "condition_id": "E0_gold_only",
            "experiment_block": "baseline",
            "reranking_method": "none",
            "distractor_count": 0,
            "distractor_type": "none",
            "mean_f1": 0.90,
            "exact_match_pct": 80.0,
        },
        {
            "condition_id": "E1_1rand_no_rerank",
            "experiment_block": "A",
            "reranking_method": "unranked",
            "distractor_count": 1,
            "distractor_type": "random",
            "mean_f1": 0.80,
            "exact_match_pct": 70.0,
        },
        {
            "condition_id": "E1_1rand_reranked",
            "experiment_block": "C",
            "reranking_method": "cross_encoder",
            "distractor_count": 1,
            "distractor_type": "random",
            "mean_f1": 0.88,
            "exact_match_pct": 78.0,
        },
        {
            "condition_id": "E_hard_3dist_no_rerank",
            "experiment_block": "B",
            "reranking_method": "unranked",
            "distractor_count": 3,
            "distractor_type": "hard",
            "mean_f1": 0.65,
            "exact_match_pct": 55.0,
        },
    ])

    fig_paths = ReportGenerator.generate_figures(df, tmp_path)
    assert "noise_degradation_f1" in fig_paths
    assert fig_paths["noise_degradation_f1"].exists()
    assert "distractor_type_comparison" in fig_paths
    assert fig_paths["distractor_type_comparison"].exists()
