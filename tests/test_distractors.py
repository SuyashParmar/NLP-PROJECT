"""Unit tests for random and hard distractor generators with exclusion rules."""

import pytest

from robustrag.corpus import DocumentCorpus
from robustrag.data_loader import QAExample
from robustrag.distractors import DistractorGenerator
from robustrag.retrieval import BM25Retriever


@pytest.fixture
def mock_corpus_and_retriever():
    corpus = DocumentCorpus()
    examples = [
        QAExample(
            id="q1",
            title="Football",
            context="The Denver Broncos defeated the Carolina Panthers in the championship game.",
            question="Which team won?",
            answers=["Denver Broncos"]
        ),
        QAExample(
            id="q2",
            title="Basketball",
            context="The Golden State Warriors won the basketball championship in California.",
            question="Where was the basketball final?",
            answers=["California"]
        ),
        QAExample(
            id="q3",
            title="Football History",
            context="Football games have been played for decades in various stadiums across the United States.",
            question="What sport is played for decades?",
            answers=["Football"]
        ),
        QAExample(
            id="q4",
            title="Astronomy",
            context="Apollo missions landed on the moon and explored lunar craters.",
            question="Where did Apollo missions land?",
            answers=["moon"]
        ),
    ]
    corpus.build_from_examples(examples)
    retriever = BM25Retriever(corpus)
    return corpus, retriever


def test_random_distractor_excludes_gold(mock_corpus_and_retriever):
    corpus, retriever = mock_corpus_and_retriever
    distractor_gen = DistractorGenerator(corpus, retriever, random_seed=42)

    gold_p = corpus.get_gold_passage("q1")
    distractors = distractor_gen.get_random_distractors(question_id="q1", count=2)

    assert len(distractors) == 2
    # Ensure none of the distractors is the gold passage
    distractor_ids = [d.passage_id for d in distractors]
    assert gold_p.passage_id not in distractor_ids


def test_hard_distractor_answer_leak_exclusion(mock_corpus_and_retriever):
    corpus, retriever = mock_corpus_and_retriever
    distractor_gen = DistractorGenerator(corpus, retriever, random_seed=42)

    gold_p = corpus.get_gold_passage("q1")
    # Query for q1 ("Which team won?"), with answer ["Denver Broncos"]
    distractors, stats = distractor_gen.get_hard_distractors(
        question_id="q1",
        question_text="Which team won the football championship?",
        reference_answers=["Denver Broncos"],
        count=2,
    )

    # Hard distractors must not contain the gold passage or reference answer
    for d in distractors:
        assert d.passage_id != gold_p.passage_id
        assert "denver broncos" not in d.text.lower()
