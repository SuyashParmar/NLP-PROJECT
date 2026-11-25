"""Failure case extraction, automatic categorization, and researcher annotation."""

import logging
from typing import Dict, List

import pandas as pd

logger = logging.getLogger("robustrag.failures")


class FailureAnalyzer:
    """Extracts, categorizes, and tracks failure cases across experimental conditions."""

    CATEGORY_CLEAN_CORRECT_NOISE_WRONG = "Clean Correct -> Noisy Incorrect"
    CATEGORY_DEGRADATION_WITH_NOISE = "Monotonic Noise Degradation"
    CATEGORY_RERANK_RESTORED = "Reranking Restored Correct Answer"
    CATEGORY_RERANK_DEGRADED = "Reranking Hurt Previously Correct Answer"
    CATEGORY_HARD_DISTRACTOR_HARM = "Hard Distractor Seduction (BM25 Confusion)"
    CATEGORY_UNSUPPORTED = "Unsupported / Empty Output"

    @classmethod
    def extract_failure_cases(
        cls,
        df_predictions: pd.DataFrame,
        clean_condition_id: str = "E0_gold_only",
        no_rerank_noisy_id: str = "E2_3rand_no_rerank",
        reranked_noisy_id: str = "E2_3rand_reranked",
        hard_distractor_id: str = "E_hard_3dist_no_rerank",
    ) -> pd.DataFrame:
        """Analyze question-level predictions across conditions and extract failure candidates."""
        if df_predictions.empty:
            return pd.DataFrame()

        # Pivot to align questions across key conditions
        pivoted_f1 = df_predictions.pivot_table(index="question_id", columns="condition_id", values="f1")
        df_predictions.pivot_table(index="question_id", columns="condition_id", values="exact_match")

        failure_records: List[Dict[str, object]] = []

        for qid in pivoted_f1.index:
            # Gather question details from one record
            q_rows = df_predictions[df_predictions["question_id"] == qid]
            first_row = q_rows.iloc[0]
            q_text = first_row["question"]
            ref_answers = first_row["reference_answers"]

            clean_f1 = pivoted_f1.loc[qid, clean_condition_id] if clean_condition_id in pivoted_f1.columns else None
            noisy_f1 = pivoted_f1.loc[qid, no_rerank_noisy_id] if no_rerank_noisy_id in pivoted_f1.columns else None
            rerank_f1 = pivoted_f1.loc[qid, reranked_noisy_id] if reranked_noisy_id in pivoted_f1.columns else None
            hard_f1 = pivoted_f1.loc[qid, hard_distractor_id] if hard_distractor_id in pivoted_f1.columns else None

            # Category 1: Clean was correct, but adding noise broke the answer
            if clean_f1 is not None and noisy_f1 is not None and clean_f1 >= 0.8 and noisy_f1 < 0.5:
                noisy_row = q_rows[q_rows["condition_id"] == no_rerank_noisy_id].iloc[0]
                failure_records.append({
                    "question_id": qid,
                    "question": q_text,
                    "reference_answers": ref_answers,
                    "predicted_answer": noisy_row["predicted_answer"],
                    "condition_id": no_rerank_noisy_id,
                    "f1": round(noisy_f1, 4),
                    "exact_match": int(noisy_row["exact_match"]),
                    "passage_ids": noisy_row["passage_ids"],
                    "failure_category": cls.CATEGORY_CLEAN_CORRECT_NOISE_WRONG,
                    "researcher_notes": "Candidate distraction: noisy evidence competed with gold passage.",
                })

            # Category 2: Reranking restored the answer
            if noisy_f1 is not None and rerank_f1 is not None and noisy_f1 < 0.5 and rerank_f1 >= 0.8:
                rerank_row = q_rows[q_rows["condition_id"] == reranked_noisy_id].iloc[0]
                failure_records.append({
                    "question_id": qid,
                    "question": q_text,
                    "reference_answers": ref_answers,
                    "predicted_answer": rerank_row["predicted_answer"],
                    "condition_id": reranked_noisy_id,
                    "f1": round(rerank_f1, 4),
                    "exact_match": int(rerank_row["exact_match"]),
                    "passage_ids": rerank_row["passage_ids"],
                    "failure_category": cls.CATEGORY_RERANK_RESTORED,
                    "researcher_notes": "Cross-encoder placed gold evidence into leading rank, enabling correct extraction.",
                })

            # Category 3: Reranking degraded a previously correct answer
            if noisy_f1 is not None and rerank_f1 is not None and noisy_f1 >= 0.8 and rerank_f1 < 0.5:
                rerank_row = q_rows[q_rows["condition_id"] == reranked_noisy_id].iloc[0]
                failure_records.append({
                    "question_id": qid,
                    "question": q_text,
                    "reference_answers": ref_answers,
                    "predicted_answer": rerank_row["predicted_answer"],
                    "condition_id": reranked_noisy_id,
                    "f1": round(rerank_f1, 4),
                    "exact_match": int(rerank_row["exact_match"]),
                    "passage_ids": rerank_row["passage_ids"],
                    "failure_category": cls.CATEGORY_RERANK_DEGRADED,
                    "researcher_notes": "Reranker misestimated distractor relevance higher than gold passage.",
                })

            # Category 4: Hard distractors harmful compared to random distractors
            if noisy_f1 is not None and hard_f1 is not None and noisy_f1 >= 0.7 and hard_f1 < 0.4:
                hard_row = q_rows[q_rows["condition_id"] == hard_distractor_id].iloc[0]
                failure_records.append({
                    "question_id": qid,
                    "question": q_text,
                    "reference_answers": ref_answers,
                    "predicted_answer": hard_row["predicted_answer"],
                    "condition_id": hard_distractor_id,
                    "f1": round(hard_f1, 4),
                    "exact_match": int(hard_row["exact_match"]),
                    "passage_ids": hard_row["passage_ids"],
                    "failure_category": cls.CATEGORY_HARD_DISTRACTOR_HARM,
                    "researcher_notes": "Lexically similar distractor passage misled generator extraction.",
                })

        failures_df = pd.DataFrame(failure_records)
        logger.info("Identified %d potential failure/transition cases.", len(failures_df))
        return failures_df
