"""Reproducible question sampling and manifest management."""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from robustrag.data_loader import QAExample

logger = logging.getLogger("robustrag.sampling")


@dataclass
class SamplingManifest:
    """Detailed audit manifest of sampled questions for experimental reproducibility."""
    dataset_name: str
    split: str
    sample_size: int
    random_seed: int
    selected_ids: List[str]
    excluded_count: int
    exclusion_reasons: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return {
            "dataset_name": self.dataset_name,
            "split": self.split,
            "sample_size": self.sample_size,
            "random_seed": self.random_seed,
            "selected_ids": self.selected_ids,
            "excluded_count": self.excluded_count,
            "exclusion_reasons": self.exclusion_reasons,
        }

    def save_json(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([{"question_id": qid} for qid in self.selected_ids])


class QuestionSampler:
    """Selects and locks a reproducible subset of QA examples across all conditions."""

    def __init__(self, sample_size: int = 100, random_seed: int = 42):
        self.sample_size = sample_size
        self.random_seed = random_seed

    def sample(
        self,
        examples: List[QAExample],
        dataset_name: str = "squad",
        split: str = "validation"
    ) -> tuple[List[QAExample], SamplingManifest]:
        """Perform deterministic sampling on valid QA examples and record audit manifest."""
        valid_examples: List[QAExample] = []
        exclusion_reasons: Dict[str, str] = {}

        # Screen for validity
        for ex in examples:
            if not ex.id:
                continue
            if not ex.context or len(ex.context.strip()) < 20:
                exclusion_reasons[ex.id] = "Context too short (< 20 chars)"
                continue
            if not ex.question or len(ex.question.strip()) < 5:
                exclusion_reasons[ex.id] = "Question too short (< 5 chars)"
                continue
            if not ex.answers:
                exclusion_reasons[ex.id] = "No valid reference answers"
                continue
            valid_examples.append(ex)

        total_valid = len(valid_examples)
        if total_valid == 0:
            raise ValueError("No valid examples available to sample from.")

        target_size = min(self.sample_size, total_valid)
        if target_size < self.sample_size:
            logger.warning(
                "Requested sample size %d exceeds valid examples count %d. Using %d.",
                self.sample_size, total_valid, target_size
            )

        # Deterministic sampling using fixed numpy generator
        rng = np.random.default_rng(self.random_seed)
        indices = rng.choice(total_valid, size=target_size, replace=False)
        # Sort indices to preserve reproducible dataset sequence
        indices.sort()

        selected_examples = [valid_examples[i] for i in indices]
        selected_ids = [ex.id for ex in selected_examples]

        manifest = SamplingManifest(
            dataset_name=dataset_name,
            split=split,
            sample_size=target_size,
            random_seed=self.random_seed,
            selected_ids=selected_ids,
            excluded_count=len(exclusion_reasons),
            exclusion_reasons=exclusion_reasons,
        )

        logger.info(
            "Sampled %d questions deterministically (seed=%d, valid_pool=%d, exclusions=%d)",
            target_size, self.random_seed, total_valid, len(exclusion_reasons)
        )
        return selected_examples, manifest
