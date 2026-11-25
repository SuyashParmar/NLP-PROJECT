"""Unit tests for reranking modules, baseline ordering, and candidate equality assertions."""

import pytest

from robustrag.corpus import Passage
from robustrag.reranking import (
    MockReranker,
    order_passages_unranked,
    verify_identical_candidate_sets,
)


@pytest.fixture
def sample_passages():
    return [
        Passage(passage_id="p1", title="Title 1", text="Text about Denver Broncos winning.", char_length=35),
        Passage(passage_id="p2", title="Title 2", text="Text about Apollo moon exploration.", char_length=36),
        Passage(passage_id="p3", title="Title 3", text="Text about Warsaw Poland capital.", char_length=33),
    ]


def test_verify_identical_candidate_sets_success(sample_passages):
    # Same passages in different order
    shuffled = [sample_passages[2], sample_passages[0], sample_passages[1]]
    # Should not raise exception
    verify_identical_candidate_sets(sample_passages, shuffled)


def test_verify_identical_candidate_sets_failure(sample_passages):
    # Candidate set differs
    subset = sample_passages[:2]
    with pytest.raises(ValueError, match="Candidate sets differ"):
        verify_identical_candidate_sets(sample_passages, subset)


def test_order_passages_unranked_preserves_elements(sample_passages):
    unranked = order_passages_unranked(sample_passages, question_id="q123", random_seed=42)
    assert len(unranked) == len(sample_passages)
    assert {p.passage_id for p in unranked} == {p.passage_id for p in sample_passages}


def test_mock_reranker(sample_passages):
    reranker = MockReranker()
    scored = reranker.score_passages("moon exploration", sample_passages)

    assert len(scored) == len(sample_passages)
    top_p, top_score = scored[0]
    # Passage 2 discusses Apollo moon exploration
    assert top_p.passage_id == "p2"
    assert top_score > 0.0
