"""File-based experiment run persistence, metadata tracking, and run loader."""

import datetime
import json
import logging
import platform
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from robustrag.config import AppConfig

logger = logging.getLogger("robustrag.storage")


class ExperimentStore:
    """Manages file-based persistence for RobustRAG experimental runs."""

    def __init__(self, base_dir: str | Path = "./outputs/runs"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def create_run_id(self, config_name: str = "run", is_smoke_test: bool = False) -> str:
        """Create timestamped unique run identifier."""
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = "smoke_" if is_smoke_test else "run_"
        clean_name = "".join(c if c.isalnum() else "_" for c in config_name)[:20]
        return f"{prefix}{ts}_{clean_name}"

    def get_run_dir(self, run_id: str) -> Path:
        return self.base_dir / run_id

    def list_runs(self) -> List[Dict[str, Any]]:
        """List all saved experiment runs sorted newest first."""
        runs = []
        if not self.base_dir.exists():
            return []

        for p in self.base_dir.iterdir():
            if p.is_dir():
                meta_file = p / "metadata.json"
                if meta_file.exists():
                    try:
                        with open(meta_file, "r", encoding="utf-8") as f:
                            meta = json.load(f)
                        runs.append(meta)
                    except Exception:
                        pass
        # Sort newest first
        runs.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        return runs

    def save_run(
        self,
        run_id: str,
        config: AppConfig,
        df_predictions: pd.DataFrame,
        df_summary: pd.DataFrame,
        df_failures: pd.DataFrame,
        summary_dict: Dict[str, Any],
        status: str = "completed"
    ) -> Path:
        """Atomically persist all assets for an experiment run."""
        run_dir = self.get_run_dir(run_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "figures").mkdir(exist_ok=True)

        # 1. Save config.yaml
        config.save_yaml(run_dir / "config.yaml")

        # 2. Save metadata.json
        metadata = {
            "run_id": run_id,
            "status": status,
            "is_smoke_test": config.is_smoke_test,
            "experiment_name": config.name,
            "timestamp": datetime.datetime.now().isoformat(),
            "sample_size": config.dataset.sample_size,
            "random_seed": config.random_seed,
            "generator_backend": config.generator.backend,
            "generator_model": config.generator.model_name,
            "reranker_model": config.reranker.model_name,
            "python_version": sys.version,
            "platform": platform.platform(),
        }
        with open(run_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        # 3. Save predictions.csv
        df_predictions.to_csv(run_dir / "predictions.csv", index=False)

        # 4. Save scores.csv
        df_summary.to_csv(run_dir / "scores.csv", index=False)

        # 5. Save summary.json
        with open(run_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary_dict, f, indent=2)

        # 6. Save failure_cases.csv
        df_failures.to_csv(run_dir / "failure_cases.csv", index=False)

        logger.info("Successfully persisted experiment run assets at: %s", run_dir)
        return run_dir

    def load_run(self, run_id: str) -> Dict[str, Any]:
        """Load all files and dataframes for a specific run."""
        run_dir = self.get_run_dir(run_id)
        if not run_dir.exists():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")

        with open(run_dir / "metadata.json", "r", encoding="utf-8") as f:
            metadata = json.load(f)

        config_file = run_dir / "config.yaml"
        config = AppConfig.from_yaml(config_file) if config_file.exists() else AppConfig()

        def safe_read_csv(fpath: Path) -> pd.DataFrame:
            if not fpath.exists():
                return pd.DataFrame()
            try:
                return pd.read_csv(fpath)
            except pd.errors.EmptyDataError:
                return pd.DataFrame()

        df_predictions = safe_read_csv(run_dir / "predictions.csv")
        df_summary = safe_read_csv(run_dir / "scores.csv")
        df_failures = safe_read_csv(run_dir / "failure_cases.csv")

        with open(run_dir / "summary.json", "r", encoding="utf-8") as f:
            summary_dict = json.load(f)

        return {
            "run_id": run_id,
            "metadata": metadata,
            "config": config,
            "predictions": df_predictions,
            "summary": df_summary,
            "failures": df_failures,
            "summary_dict": summary_dict,
            "run_dir": run_dir,
        }

    def update_failure_annotations(self, run_id: str, df_failures: pd.DataFrame) -> None:
        """Update failure_cases.csv with user-provided researcher annotations."""
        run_dir = self.get_run_dir(run_id)
        if run_dir.exists():
            df_failures.to_csv(run_dir / "failure_cases.csv", index=False)
            logger.info("Updated failure annotations for run %s", run_id)
