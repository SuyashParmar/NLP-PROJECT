"""Evaluation metrics: Standard SQuAD Token-level F1 and Exact Match."""

import re
import string
from collections import Counter
from typing import List, Tuple


def normalize_answer(s: str) -> str:
    """Standard SQuAD evaluation answer normalization:

    1. Lowercase
    2. Remove punctuation
    3. Remove articles (a, an, the)
    4. Normalize whitespace
    """
    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    if not s:
        return ""

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def exact_match_score(prediction: str, ground_truth: str) -> float:
    """Compute binary Exact Match (1.0 or 0.0) between normalized strings."""
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def f1_score(prediction: str, ground_truth: str) -> float:
    """Compute token-level precision, recall, and harmonic mean F1."""
    norm_pred = normalize_answer(prediction)
    norm_gold = normalize_answer(ground_truth)

    pred_tokens = norm_pred.split()
    gold_tokens = norm_gold.split()

    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = 1.0 * num_same / len(pred_tokens)
    recall = 1.0 * num_same / len(gold_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return float(f1)


def evaluate_qa_prediction(prediction: str, ground_truths: List[str]) -> Tuple[float, float]:
    """Calculate the max F1 and Exact Match across all accepted ground-truth annotations."""
    if not ground_truths:
        return 0.0, 0.0

    em_scores = [exact_match_score(prediction, gt) for gt in ground_truths]
    f1_scores = [f1_score(prediction, gt) for gt in ground_truths]

    return max(f1_scores), max(em_scores)
