"""BM25 Lexical Retrieval and Recall@k Evaluation."""

import logging
import re
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
from rank_bm25 import BM25Okapi

from robustrag.corpus import DocumentCorpus, Passage

logger = logging.getLogger("robustrag.retrieval")


def tokenize_text(text: str) -> List[str]:
    """Simple alphanumeric tokenization with lowercasing."""
    return re.findall(r"\b\w+\b", text.lower())


@dataclass
class RetrievalResult:
    """Result of BM25 retrieval query."""
    passage_id: str
    score: float
    passage: Passage


class BM25Retriever:
    """BM25-based lexical passage retriever over the indexed DocumentCorpus."""

    def __init__(self, corpus: DocumentCorpus):
        self.corpus = corpus
        self.passage_ids: List[str] = corpus.get_all_passage_ids()
        self.corpus_passages: List[Passage] = [
            corpus.get_passage(pid) for pid in self.passage_ids if corpus.get_passage(pid) is not None
        ]

        # Tokenize corpus for BM25
        logger.info("Tokenizing %d passages for BM25 indexing...", len(self.corpus_passages))
        self.tokenized_corpus = [
            tokenize_text(f"{p.title} {p.text}") for p in self.corpus_passages
        ]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        logger.info("BM25 index successfully constructed.")

    def retrieve(self, query: str, top_k: int = 10) -> List[RetrievalResult]:
        """Rank corpus passages for query using BM25Okapi."""
        if not query.strip():
            return []

        tokenized_query = tokenize_text(query)
        if not tokenized_query:
            return []

        scores = self.bm25.get_scores(tokenized_query)
        top_k = min(top_k, len(scores))

        # Argsort in descending order
        ranked_indices = np.argsort(scores)[::-1][:top_k]

        results: List[RetrievalResult] = []
        for idx in ranked_indices:
            score = float(scores[idx])
            passage = self.corpus_passages[idx]
            results.append(RetrievalResult(passage_id=passage.passage_id, score=score, passage=passage))

        return results

    def evaluate_recall_at_k(
        self,
        questions: List[Dict[str, str]],  # List of {"question_id": str, "question": str}
        k_values: List[int] = [1, 3, 5, 10]
    ) -> Dict[str, float]:
        """Compute Recall@k over questions with verified gold passage IDs.

        Note: Recall@k measures whether the retriever finds the gold passage.
        This is kept strictly separate from the controlled context-noise experiment.
        """
        max_k = max(k_values)
        hits_at_k = {k: 0 for k in k_values}
        total_evaluable = 0

        for item in questions:
            qid = item["question_id"]
            q_text = item["question"]
            gold_pid = self.corpus.question_to_gold_id.get(qid)
            if not gold_pid:
                continue

            total_evaluable += 1
            retrieved = self.retrieve(q_text, top_k=max_k)
            retrieved_pids = [r.passage_id for r in retrieved]

            for k in k_values:
                if gold_pid in retrieved_pids[:k]:
                    hits_at_k[k] += 1

        metrics: Dict[str, float] = {}
        if total_evaluable > 0:
            for k in k_values:
                metrics[f"recall_at_{k}"] = round(hits_at_k[k] / total_evaluable, 4)
            metrics["total_evaluated_queries"] = total_evaluable
        else:
            for k in k_values:
                metrics[f"recall_at_{k}"] = 0.0
            metrics["total_evaluated_queries"] = 0

        logger.info("BM25 Retrieval Diagnostics: %s", metrics)
        return metrics
