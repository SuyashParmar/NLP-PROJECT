"""Unit tests for SQuAD F1 and Exact Match metrics."""

from robustrag.metrics import evaluate_qa_prediction, exact_match_score, f1_score, normalize_answer


def test_normalize_answer():
    # Lowercase & punctuation
    assert normalize_answer("Denver Broncos.") == "denver broncos"
    # Articles removal
    assert normalize_answer("The Denver Broncos") == "denver broncos"
    assert normalize_answer("an apple, and a banana") == "apple and banana"
    # Whitespace normalization
    assert normalize_answer("  multiple   spaces \n here  ") == "multiple spaces here"
    # Empty string
    assert normalize_answer("") == ""


def test_exact_match_score():
    assert exact_match_score("Denver Broncos", "denver broncos") == 1.0
    assert exact_match_score("The Denver Broncos", "denver broncos") == 1.0
    assert exact_match_score("Denver Broncos.", "denver broncos") == 1.0
    assert exact_match_score("Carolina Panthers", "denver broncos") == 0.0
    assert exact_match_score("", "") == 1.0


def test_f1_score_edge_cases():
    # Exact match
    assert f1_score("Denver Broncos", "denver broncos") == 1.0
    # Complete mismatch
    assert f1_score("Denver Broncos", "Seattle Seahawks") == 0.0
    # Partial overlap
    # Pred: "denver broncos", Gold: "denver" -> common=1, precision=0.5, recall=1.0 -> F1 = 2*0.5*1 / 1.5 = 2/3 = 0.6667
    f1 = f1_score("Denver Broncos", "Denver")
    assert abs(f1 - (2.0 / 3.0)) < 1e-4
    # Empty string handling
    assert f1_score("", "") == 1.0
    assert f1_score("Denver", "") == 0.0
    assert f1_score("", "Denver") == 0.0


def test_evaluate_qa_prediction_multiple_references():
    ground_truths = ["Denver Broncos", "Denver Broncos Football Team", "Broncos"]

    # Matches third reference exactly
    f1, em = evaluate_qa_prediction("Broncos", ground_truths)
    assert em == 1.0
    assert f1 == 1.0

    # Partial match
    f1, em = evaluate_qa_prediction("Denver", ground_truths)
    assert em == 0.0
    assert f1 > 0.0

    # Empty prediction
    f1_empty, em_empty = evaluate_qa_prediction("", ground_truths)
    assert em_empty == 0.0
    assert f1_empty == 0.0
