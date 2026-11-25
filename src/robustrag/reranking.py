"""Cross-Encoder Reranking and Baseline Ordering Interventions."""

import hashlib
import logging
from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np

from robustrag.corpus import Passage

logger = logging.getLogger("robustrag.reranking")


def verify_identical_candidate_sets(passages_a: List[Passage], passages_b: List[Passage]) -> None:
    """Rigorous scientific assertion: verify that two conditions evaluate identical document sets."""
    set_a = {p.passage_id for p in passages_a}
    set_b = {p.passage_id for p in passages_b}
    if set_a != set_b:
        raise ValueError(
            f"Experimental flaw detected! Candidate sets differ between conditions: "
            f"{set_a ^ set_b}"
        )


class BaseReranker(ABC):
    """Abstract base interface for passage reranking models."""

    @abstractmethod
    def score_passages(self, query: str, passages: List[Passage]) -> List[Tuple[Passage, float]]:
        """Score passages for query, returning list of (Passage, score) pairs in descending order."""
        pass

    def rerank(self, query: str, passages: List[Passage]) -> List[Passage]:
        """Return passages ordered in descending score order."""
        scored = self.score_passages(query, passages)
        return [p for p, _ in scored]


class CrossEncoderReranker(BaseReranker):
    """Hugging Face SentenceTransformers CrossEncoder wrapper.

    Default model: cross-encoder/ms-marco-MiniLM-L-6-v2
    Note: Cross-encoder scores represent uncalibrated pairwise relevance logits.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", batch_size: int = 16):
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = None
        self._load_model()

    def _load_model(self) -> None:
        try:
            import torch
            from sentence_transformers import CrossEncoder
            device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
            logger.info("Loading CrossEncoder model '%s' on device %s...", self.model_name, device)
            self._model = CrossEncoder(self.model_name, max_length=512, device=device)
            logger.info("CrossEncoder model successfully loaded.")
        except Exception as e:
            logger.error("Failed to load CrossEncoder model '%s': %s", self.model_name, e)
            raise RuntimeError(
                f"Could not initialize CrossEncoder model '{self.model_name}'. "
                f"Ensure sentence-transformers is installed and models can be downloaded or cached."
            ) from e

    def score_passages(self, query: str, passages: List[Passage]) -> List[Tuple[Passage, float]]:
        if not passages:
            return []
        if len(passages) == 1:
            return [(passages[0], 1.0)]

        # Prepare query-passage pairs
        pairs = [[query, f"{p.title}: {p.text}"] for p in passages]
        try:
            scores = self._model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
            scores = np.array(scores, dtype=float)

            # Sort descending
            ranked_indices = np.argsort(scores)[::-1]
            return [(passages[i], float(scores[i])) for i in ranked_indices]
        except Exception as e:
            logger.error("Error during CrossEncoder inference: %s", e)
            # Safe deterministic fallback
            return [(p, 0.0) for p in passages]


class MockReranker(BaseReranker):
    """Fast deterministic mock reranker for unit tests and smoke tests."""

    def score_passages(self, query: str, passages: List[Passage]) -> List[Tuple[Passage, float]]:
        """Mock scoring based on character-level lexical overlap with query."""
        scored = []
        q_tokens = set(query.lower().split())
        for p in passages:
            p_tokens = set((p.title + " " + p.text).lower().split())
            overlap = len(q_tokens & p_tokens) / max(1, len(q_tokens))
            scored.append((p, float(overlap)))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored


def order_passages_unranked(
    passages: List[Passage],
    question_id: str,
    random_seed: int = 42
) -> List[Passage]:
    """Condition C1 (No reranking): Deterministically shuffle candidate passages.

    CRITICAL METHODOLOGICAL GUARANTEE:
    - Uses the exact same candidate passages.
    - Order is independently reproducible using a deterministic seed + question_id hash.
    - Does NOT force the gold passage to the first slot.
    """
    if len(passages) <= 1:
        return list(passages)

    # Deterministic permutation based on seed and question_id hash
    salt = int(hashlib.md5(f"{question_id}_{random_seed}".encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(salt)
    permuted_indices = rng.permutation(len(passages))
    return [passages[i] for i in permuted_indices]
