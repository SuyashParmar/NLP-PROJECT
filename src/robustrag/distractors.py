"""Random and Hard Distractor generation with strict exclusion filters."""

import logging
import re
from typing import Dict, List

import numpy as np

from robustrag.corpus import DocumentCorpus, Passage
from robustrag.retrieval import BM25Retriever

logger = logging.getLogger("robustrag.distractors")


def normalize_string_for_check(text: str) -> str:
    """Normalize text for answer-string substring containment checking."""
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(text.split())


class DistractorGenerator:
    """Generates controlled random or hard distractor passages for given QA questions."""

    def __init__(self, corpus: DocumentCorpus, retriever: BM25Retriever, random_seed: int = 42):
        self.corpus = corpus
        self.retriever = retriever
        self.random_seed = random_seed
        self.all_pids: List[str] = corpus.get_all_passage_ids()

    def get_random_distractors(
        self,
        question_id: str,
        count: int,
        seed_offset: int = 0
    ) -> List[Passage]:
        """Sample 'count' random distractors from corpus, strictly excluding the gold passage."""
        if count <= 0:
            return []

        gold_pid = self.corpus.question_to_gold_id.get(question_id)
        candidate_pids = [pid for pid in self.all_pids if pid != gold_pid]

        if not candidate_pids:
            return []

        rng = np.random.default_rng(self.random_seed + seed_offset + hash(question_id) % 10000)
        chosen_count = min(count, len(candidate_pids))
        chosen_pids = rng.choice(candidate_pids, size=chosen_count, replace=False)

        passages = [self.corpus.get_passage(pid) for pid in chosen_pids]
        return [p for p in passages if p is not None]

    def get_hard_distractors(
        self,
        question_id: str,
        question_text: str,
        reference_answers: List[str],
        count: int,
        pool_size: int = 15,
        seed_offset: int = 0
    ) -> tuple[List[Passage], Dict[str, int]]:
        """Retrieve BM25-similar passages that do NOT contain reference answers or match gold passage.

        Returns:
            passages: Selected hard distractor passages.
            exclusion_stats: Counts of candidates excluded due to gold match or answer leak.
        """
        if count <= 0:
            return [], {}

        gold_pid = self.corpus.question_to_gold_id.get(question_id)
        norm_answers = [normalize_string_for_check(a) for a in reference_answers if normalize_string_for_check(a)]

        # Fetch top lexical candidates via BM25
        retrieved = self.retriever.retrieve(question_text, top_k=max(pool_size, count * 3))

        selected: List[Passage] = []
        excluded_gold = 0
        excluded_answer_leak = 0

        for res in retrieved:
            cand_passage = res.passage
            cand_pid = res.passage_id

            # Rule 1: Exclude exact gold passage
            if cand_pid == gold_pid:
                excluded_gold += 1
                continue

            # Rule 2: Exclude if candidate passage text contains any reference answer string
            norm_cand_text = normalize_string_for_check(cand_passage.text)
            has_answer = any(
                ans in norm_cand_text
                for ans in norm_answers
                if len(ans) >= 3  # Avoid excluding on single-letter stopwords
            )
            if has_answer:
                excluded_answer_leak += 1
                continue

            # Valid hard distractor
            selected.append(cand_passage)
            if len(selected) >= count:
                break

        # Fallback: If BM25 pool had too few non-leaking candidates, supplement with random non-leaking passages
        fallback_added = 0
        if len(selected) < count:
            selected_pids = {p.passage_id for p in selected}
            random_fallback = self.get_random_distractors(
                question_id, count=(count - len(selected)) * 2, seed_offset=seed_offset + 999
            )
            for p in random_fallback:
                if p.passage_id in selected_pids:
                    continue
                norm_p_text = normalize_string_for_check(p.text)
                if not any(ans in norm_p_text for ans in norm_answers if len(ans) >= 3):
                    selected.append(p)
                    selected_pids.add(p.passage_id)
                    fallback_added += 1
                    if len(selected) >= count:
                        break

        exclusion_stats = {
            "excluded_gold": excluded_gold,
            "excluded_answer_leak": excluded_answer_leak,
            "fallback_added": fallback_added,
        }
        return selected[:count], exclusion_stats
