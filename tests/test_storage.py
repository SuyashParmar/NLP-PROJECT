"""Unit tests for storage, persistence, and run loading."""

import pandas as pd

from robustrag.config import AppConfig
from robustrag.storage import ExperimentStore


def test_storage_save_and_load(tmp_path):
    store = ExperimentStore(tmp_path)
    run_id = store.create_run_id("test_exp", is_smoke_test=False)
    config = AppConfig(name="test_exp")

    df_preds = pd.DataFrame([{"question_id": "q1", "f1": 1.0, "exact_match": 1.0}])
    df_summary = pd.DataFrame([{"condition_id": "c1", "mean_f1": 1.0, "exact_match_pct": 100.0}])
    df_failures = pd.DataFrame([{"question_id": "q1", "failure_category": "Test"}])
    summary_dict = {"test": 123}

    run_dir = store.save_run(
        run_id=run_id,
        config=config,
        df_predictions=df_preds,
        df_summary=df_summary,
        df_failures=df_failures,
        summary_dict=summary_dict,
    )
    assert run_dir.exists()

    # Load back
    loaded = store.load_run(run_id)
    assert loaded["run_id"] == run_id
    assert len(loaded["predictions"]) == 1
    assert loaded["metadata"]["status"] == "completed"
    assert loaded["metadata"]["is_smoke_test"] is False


def test_update_failure_annotations(tmp_path):
    store = ExperimentStore(tmp_path)
    run_id = store.create_run_id("test_exp")
    config = AppConfig()

    store.save_run(
        run_id=run_id,
        config=config,
        df_predictions=pd.DataFrame(),
        df_summary=pd.DataFrame(),
        df_failures=pd.DataFrame([{"qid": "q1", "researcher_notes": "original"}]),
        summary_dict={},
    )

    df_updated = pd.DataFrame([{"qid": "q1", "researcher_notes": "annotated by human"}])
    store.update_failure_annotations(run_id, df_updated)

    loaded = store.load_run(run_id)
    assert loaded["failures"]["researcher_notes"].iloc[0] == "annotated by human"
