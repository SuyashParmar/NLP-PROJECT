"""Master Experiment Orchestrator for RobustRAG."""

import logging
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

from robustrag.config import AppConfig
from robustrag.corpus import DocumentCorpus
from robustrag.data_loader import DatasetLoader
from robustrag.distractors import DistractorGenerator
from robustrag.failures import FailureAnalyzer
from robustrag.generation import build_generator
from robustrag.metrics import evaluate_qa_prediction
from robustrag.reranking import (
    BaseReranker,
    CrossEncoderReranker,
    MockReranker,
    order_passages_unranked,
    verify_identical_candidate_sets,
)
from robustrag.retrieval import BM25Retriever
from robustrag.sampling import QuestionSampler
from robustrag.statistics import compute_paired_bootstrap_ci, summarize_condition_metrics
from robustrag.storage import ExperimentStore
from robustrag.utils import set_seed

logger = logging.getLogger("robustrag.experiment")


class ExperimentRunner:
    """Orchestrates controlled RAG experiments, data isolation, and metric calculation."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.run_id = ""
        self.store = ExperimentStore(config.storage.output_dir)
        self.loader = DatasetLoader(
            dataset_name=config.dataset.name,
            split=config.dataset.split,
            cache_dir=config.dataset.cache_dir,
            fallback_path=config.dataset.fallback_path,
        )

    def run(
        self,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> Dict[str, Any]:
        """Execute full experimental pipeline and persist all artifacts."""
        def update_progress(pct: float, msg: str) -> None:
            logger.info("[%.0f%%] %s", pct * 100, msg)
            if progress_callback:
                progress_callback(pct, msg)

        set_seed(self.config.random_seed)
        self.run_id = self.store.create_run_id(
            config_name=self.config.name,
            is_smoke_test=self.config.is_smoke_test
        )

        update_progress(0.05, "Loading dataset...")
        raw_examples = self.loader.load()

        update_progress(0.12, f"Sampling {self.config.dataset.sample_size} questions...")
        sampler = QuestionSampler(
            sample_size=self.config.dataset.sample_size,
            random_seed=self.config.random_seed
        )
        questions, manifest = sampler.sample(
            raw_examples,
            dataset_name=self.config.dataset.name,
            split=self.config.dataset.split
        )

        update_progress(0.20, "Building document corpus and BM25 index...")
        corpus = DocumentCorpus(dataset_name=self.config.dataset.name)
        # Index all available examples so the distractor pool is large and diverse
        corpus.build_from_examples(raw_examples)
        retriever = BM25Retriever(corpus)
        distractor_gen = DistractorGenerator(corpus, retriever, random_seed=self.config.random_seed)

        update_progress(0.28, "Initializing models...")
        generator = build_generator(self.config.generator)
        reranker: BaseReranker
        if self.config.reranker.backend == "mock":
            reranker = MockReranker()
        else:
            reranker = CrossEncoderReranker(
                model_name=self.config.reranker.model_name,
                batch_size=self.config.reranker.batch_size
            )

        # -------------------------------------------------------------
        # Controlled Experiment Execution
        # -------------------------------------------------------------
        prediction_rows: List[Dict[str, Any]] = []
        total_eval_steps = len(questions)
        completed_steps = 0

        update_progress(0.35, "Executing controlled context-noise experiments...")

        for q_idx, q in enumerate(questions):
            gold_passage = corpus.get_gold_passage(q.id)
            if not gold_passage:
                logger.warning("No gold passage found for question %s, skipping.", q.id)
                continue

            # ---------------------------------------------------------
            # Baseline: E0 (Gold passage only, 0 distractors)
            # ---------------------------------------------------------
            ans_e0 = generator.generate_answer(q.question, [gold_passage], use_cache=self.config.storage.cache_inference)
            f1_e0, em_e0 = evaluate_qa_prediction(ans_e0, q.answers)
            prediction_rows.append({
                "run_id": self.run_id,
                "question_id": q.id,
                "question": q.question,
                "reference_answers": "|".join(q.answers),
                "experiment_block": "baseline",
                "condition_id": "E0_gold_only",
                "distractor_count": 0,
                "distractor_type": "none",
                "reranking_method": "none",
                "passage_ordering": [gold_passage.passage_id],
                "passage_ids": [gold_passage.passage_id],
                "predicted_answer": ans_e0,
                "f1": f1_e0,
                "exact_match": em_e0,
                "generation_status": "success" if ans_e0 else "empty",
                "error_message": "",
            })

            # ---------------------------------------------------------
            # Experiment A & C: Noise Quantities with and without Reranking
            # ---------------------------------------------------------
            noisy_counts = [k for k in self.config.noise.quantities if k > 0]
            for k in noisy_counts:
                # 1. Sample candidate passages
                random_distractors = distractor_gen.get_random_distractors(q.id, count=k, seed_offset=k * 10)
                candidates = [gold_passage] + random_distractors

                # Condition C1: Unranked baseline (deterministic shuffle without forcing gold first)
                unranked_passages = order_passages_unranked(candidates, q.id, random_seed=self.config.random_seed)
                # Verify candidates
                verify_identical_candidate_sets(candidates, unranked_passages)

                ans_unranked = generator.generate_answer(
                    q.question, unranked_passages, use_cache=self.config.storage.cache_inference
                )
                f1_unranked, em_unranked = evaluate_qa_prediction(ans_unranked, q.answers)
                prediction_rows.append({
                    "run_id": self.run_id,
                    "question_id": q.id,
                    "question": q.question,
                    "reference_answers": "|".join(q.answers),
                    "experiment_block": "A",
                    "condition_id": f"E{k}_{k}rand_no_rerank",
                    "distractor_count": k,
                    "distractor_type": "random",
                    "reranking_method": "unranked",
                    "passage_ordering": [p.passage_id for p in unranked_passages],
                    "passage_ids": [p.passage_id for p in unranked_passages],
                    "predicted_answer": ans_unranked,
                    "f1": f1_unranked,
                    "exact_match": em_unranked,
                    "generation_status": "success" if ans_unranked else "empty",
                    "error_message": "",
                })

                # Condition C2: Cross-encoder reranked (same candidate set!)
                if self.config.experiments.run_experiment_c:
                    reranked_passages = reranker.rerank(q.question, candidates)
                    verify_identical_candidate_sets(candidates, reranked_passages)

                    ans_reranked = generator.generate_answer(
                        q.question, reranked_passages, use_cache=self.config.storage.cache_inference
                    )
                    f1_reranked, em_reranked = evaluate_qa_prediction(ans_reranked, q.answers)
                    prediction_rows.append({
                        "run_id": self.run_id,
                        "question_id": q.id,
                        "question": q.question,
                        "reference_answers": "|".join(q.answers),
                        "experiment_block": "C",
                        "condition_id": f"E{k}_{k}rand_reranked",
                        "distractor_count": k,
                        "distractor_type": "random",
                        "reranking_method": "cross_encoder",
                        "passage_ordering": [p.passage_id for p in reranked_passages],
                        "passage_ids": [p.passage_id for p in reranked_passages],
                        "predicted_answer": ans_reranked,
                        "f1": f1_reranked,
                        "exact_match": em_reranked,
                        "generation_status": "success" if ans_reranked else "empty",
                        "error_message": "",
                    })

                    # Condition D: Filtering ablation (reranked + top-k filter)
                    if self.config.experiments.run_experiment_d and k >= self.config.experiments.top_k_filter:
                        filtered_passages = reranked_passages[: self.config.experiments.top_k_filter]
                        ans_filtered = generator.generate_answer(
                            q.question, filtered_passages, use_cache=self.config.storage.cache_inference
                        )
                        f1_filtered, em_filtered = evaluate_qa_prediction(ans_filtered, q.answers)
                        prediction_rows.append({
                            "run_id": self.run_id,
                            "question_id": q.id,
                            "question": q.question,
                            "reference_answers": "|".join(q.answers),
                            "experiment_block": "D",
                            "condition_id": f"E{k}_top{self.config.experiments.top_k_filter}_filtered",
                            "distractor_count": k,
                            "distractor_type": "random",
                            "reranking_method": "cross_encoder_filtered",
                            "passage_ordering": [p.passage_id for p in filtered_passages],
                            "passage_ids": [p.passage_id for p in filtered_passages],
                            "predicted_answer": ans_filtered,
                            "f1": f1_filtered,
                            "exact_match": em_filtered,
                            "generation_status": "success" if ans_filtered else "empty",
                            "error_message": "",
                        })

            # ---------------------------------------------------------
            # Experiment B: Distractor Type (Random vs Hard) at fixed k
            # ---------------------------------------------------------
            if self.config.experiments.run_experiment_b:
                fixed_k = self.config.noise.fixed_distractor_count
                hard_distractors, _ = distractor_gen.get_hard_distractors(
                    question_id=q.id,
                    question_text=q.question,
                    reference_answers=q.answers,
                    count=fixed_k,
                    pool_size=self.config.noise.hard_distractor_pool_size,
                    seed_offset=fixed_k * 50
                )
                hard_candidates = [gold_passage] + hard_distractors
                hard_unranked = order_passages_unranked(hard_candidates, q.id, random_seed=self.config.random_seed)
                verify_identical_candidate_sets(hard_candidates, hard_unranked)

                ans_hard = generator.generate_answer(
                    q.question, hard_unranked, use_cache=self.config.storage.cache_inference
                )
                f1_hard, em_hard = evaluate_qa_prediction(ans_hard, q.answers)
                prediction_rows.append({
                    "run_id": self.run_id,
                    "question_id": q.id,
                    "question": q.question,
                    "reference_answers": "|".join(q.answers),
                    "experiment_block": "B",
                    "condition_id": f"E_hard_{fixed_k}dist_no_rerank",
                    "distractor_count": fixed_k,
                    "distractor_type": "hard",
                    "reranking_method": "unranked",
                    "passage_ordering": [p.passage_id for p in hard_unranked],
                    "passage_ids": [p.passage_id for p in hard_unranked],
                    "predicted_answer": ans_hard,
                    "f1": f1_hard,
                    "exact_match": em_hard,
                    "generation_status": "success" if ans_hard else "empty",
                    "error_message": "",
                })

            completed_steps += 1
            progress_frac = 0.35 + (0.45 * (completed_steps / total_eval_steps))
            if q_idx % max(1, total_eval_steps // 10) == 0:
                update_progress(progress_frac, f"Evaluated {completed_steps}/{total_eval_steps} questions...")

        update_progress(0.82, "Computing statistics and bootstrap confidence intervals...")
        df_predictions = pd.DataFrame(prediction_rows)
        df_summary = summarize_condition_metrics(df_predictions)

        # Compute paired bootstrap confidence intervals
        bootstrap_cis: Dict[str, Any] = {}
        pivot_f1 = df_predictions.pivot_table(index="question_id", columns="condition_id", values="f1")

        # 1. Noise degradation CI: (E2 3-rand no-rerank) - (E0 gold only)
        if "E2_3rand_no_rerank" in pivot_f1.columns and "E0_gold_only" in pivot_f1.columns:
            ci_deg = compute_paired_bootstrap_ci(
                scores_treatment=pivot_f1["E2_3rand_no_rerank"].tolist(),
                scores_control=pivot_f1["E0_gold_only"].tolist(),
                iterations=self.config.evaluation.bootstrap_iterations,
                confidence_level=self.config.evaluation.confidence_level,
                seed=self.config.evaluation.bootstrap_seed,
            )
            bootstrap_cis["noise_degradation_3rand_vs_clean"] = ci_deg.to_dict()

        # 2. Reranking mitigation CI: (E2 3-rand reranked) - (E2 3-rand no-rerank)
        if "E2_3rand_reranked" in pivot_f1.columns and "E2_3rand_no_rerank" in pivot_f1.columns:
            ci_mit = compute_paired_bootstrap_ci(
                scores_treatment=pivot_f1["E2_3rand_reranked"].tolist(),
                scores_control=pivot_f1["E2_3rand_no_rerank"].tolist(),
                iterations=self.config.evaluation.bootstrap_iterations,
                confidence_level=self.config.evaluation.confidence_level,
                seed=self.config.evaluation.bootstrap_seed,
            )
            bootstrap_cis["reranker_mitigation_3rand"] = ci_mit.to_dict()

        # 3. Distractor type harm CI: (Hard distractors) - (Random distractors)
        hard_cond = f"E_hard_{self.config.noise.fixed_distractor_count}dist_no_rerank"
        rand_cond = f"E{self.config.noise.fixed_distractor_count}_{self.config.noise.fixed_distractor_count}rand_no_rerank"
        if hard_cond in pivot_f1.columns and rand_cond in pivot_f1.columns:
            ci_hard = compute_paired_bootstrap_ci(
                scores_treatment=pivot_f1[hard_cond].tolist(),
                scores_control=pivot_f1[rand_cond].tolist(),
                iterations=self.config.evaluation.bootstrap_iterations,
                confidence_level=self.config.evaluation.confidence_level,
                seed=self.config.evaluation.bootstrap_seed,
            )
            bootstrap_cis["distractor_type_hard_vs_rand"] = ci_hard.to_dict()

        # Optional BM25 natural retrieval evaluation
        retrieval_diagnostics: Dict[str, Any] = {}
        if self.config.experiments.run_retrieval_realistic:
            update_progress(0.88, "Evaluating realistic BM25 natural retrieval...")
            q_dicts = [{"question_id": q.id, "question": q.question} for q in questions]
            retrieval_diagnostics = retriever.evaluate_recall_at_k(
                q_dicts, k_values=self.config.experiments.realistic_retrieval_k
            )

        update_progress(0.92, "Extracting failure cases...")
        df_failures = FailureAnalyzer.extract_failure_cases(df_predictions)

        summary_dict = {
            "run_id": self.run_id,
            "bootstrap_ci": bootstrap_cis,
            "retrieval_diagnostics": retrieval_diagnostics,
            "conditions_summary": df_summary.to_dict(orient="records"),
        }

        update_progress(0.96, "Persisting run artifacts and generating reports...")
        run_dir = self.store.save_run(
            run_id=self.run_id,
            config=self.config,
            df_predictions=df_predictions,
            df_summary=df_summary,
            df_failures=df_failures,
            summary_dict=summary_dict,
            status="completed"
        )

        from robustrag.reporting import ReportGenerator
        ReportGenerator.generate_figures(df_summary, run_dir / "figures")
        ReportGenerator.generate_markdown_report(
            run_id=self.run_id,
            metadata={
                "run_id": self.run_id,
                "is_smoke_test": self.config.is_smoke_test,
                "experiment_name": self.config.name,
                "sample_size": self.config.dataset.sample_size,
                "random_seed": self.config.random_seed,
                "generator_backend": self.config.generator.backend,
                "generator_model": self.config.generator.model_name,
                "reranker_model": self.config.reranker.model_name,
            },
            df_summary=df_summary,
            summary_dict=summary_dict,
            output_file=run_dir / "report.md"
        )

        update_progress(1.0, f"Experiment completed successfully: {self.run_id}")

        return {
            "run_id": self.run_id,
            "run_dir": run_dir,
            "summary": df_summary,
            "predictions": df_predictions,
            "failures": df_failures,
            "bootstrap_ci": bootstrap_cis,
            "retrieval_diagnostics": retrieval_diagnostics,
        }
