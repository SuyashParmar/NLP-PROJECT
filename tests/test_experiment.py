"""Unit tests for end-to-end experiment orchestration."""

from robustrag.config import AppConfig
from robustrag.experiment import ExperimentRunner


def test_experiment_runner_mock_pipeline(tmp_path):
    config = AppConfig(
        name="test_mock_run",
        random_seed=42,
        is_smoke_test=True,
    )
    config.dataset.sample_size = 4
    config.dataset.fallback_path = "data/squad_sample_fallback.json"
    config.storage.output_dir = str(tmp_path / "outputs")
    config.generator.backend = "mock"
    config.reranker.backend = "mock"
    config.noise.quantities = [0, 1]
    config.noise.fixed_distractor_count = 1
    config.evaluation.bootstrap_iterations = 50

    runner = ExperimentRunner(config)
    results = runner.run()

    assert "run_id" in results
    assert results["run_dir"].exists()
    assert (results["run_dir"] / "predictions.csv").exists()
    assert (results["run_dir"] / "scores.csv").exists()
    assert (results["run_dir"] / "summary.json").exists()
    assert (results["run_dir"] / "report.md").exists()

    df_preds = results["predictions"]
    assert len(df_preds) > 0
    # Baseline condition must be present
    assert "E0_gold_only" in df_preds["condition_id"].values
