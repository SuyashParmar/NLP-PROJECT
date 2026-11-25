"""Unit tests for reproducible question sampling and manifests."""

import pytest

from robustrag.data_loader import QAExample
from robustrag.sampling import QuestionSampler


@pytest.fixture
def sample_examples():
    return [
        QAExample(id=f"q_{i}", title=f"Title_{i}", context=f"Valid context sentence {i} with sufficient text.", question=f"Question {i}?", answers=[f"Answer {i}"])
        for i in range(20)
    ]


def test_sampling_determinism(sample_examples):
    sampler1 = QuestionSampler(sample_size=5, random_seed=42)
    selected1, manifest1 = sampler1.sample(sample_examples)

    sampler2 = QuestionSampler(sample_size=5, random_seed=42)
    selected2, manifest2 = sampler2.sample(sample_examples)

    assert len(selected1) == 5
    assert [q.id for q in selected1] == [q.id for q in selected2]
    assert manifest1.selected_ids == manifest2.selected_ids


def test_sampling_different_seeds(sample_examples):
    sampler_a = QuestionSampler(sample_size=5, random_seed=42)
    selected_a, _ = sampler_a.sample(sample_examples)

    sampler_b = QuestionSampler(sample_size=5, random_seed=999)
    selected_b, _ = sampler_b.sample(sample_examples)

    # Different seeds should yield different random permutations
    assert [q.id for q in selected_a] != [q.id for q in selected_b]


def test_sampling_exclusion_rules():
    invalid_examples = [
        QAExample(id="short_ctx", title="T", context="too short", question="Valid question?", answers=["Ans"]),
        QAExample(id="short_q", title="T", context="Valid context sentence with sufficient text length.", question="Q?", answers=["Ans"]),
        QAExample(id="no_ans", title="T", context="Valid context sentence with sufficient text length.", question="Valid question?", answers=[]),
        QAExample(id="valid_1", title="T", context="Valid context sentence with sufficient text length.", question="Valid question?", answers=["Ans"]),
    ]

    sampler = QuestionSampler(sample_size=5, random_seed=42)
    selected, manifest = sampler.sample(invalid_examples)

    assert len(selected) == 1
    assert selected[0].id == "valid_1"
    assert manifest.excluded_count == 3
    assert "short_ctx" in manifest.exclusion_reasons
    assert "short_q" in manifest.exclusion_reasons
    assert "no_ans" in manifest.exclusion_reasons
