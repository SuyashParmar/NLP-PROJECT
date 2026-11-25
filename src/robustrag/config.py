"""Experiment configuration schemas and validators for RobustRAG."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import yaml


@dataclass
class GeneratorConfig:
    backend: str = "local"  # "local", "groq", or "mock"
    model_name: str = "google/flan-t5-small"
    groq_model: str = "qwen/qwen3.8-27b"
    max_input_length: int = 1024
    max_new_tokens: int = 32
    temperature: float = 0.0
    do_sample: bool = False


@dataclass
class RerankerConfig:
    backend: str = "local"  # "local" or "mock"
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    batch_size: int = 16


@dataclass
class DatasetConfig:
    name: str = "squad"
    split: str = "validation"
    sample_size: int = 100
    cache_dir: str = "./.cache/datasets"
    fallback_path: str = "./data/squad_sample_fallback.json"


@dataclass
class NoiseConfig:
    quantities: List[int] = field(default_factory=lambda: [0, 1, 3, 5])
    distractor_types: List[str] = field(default_factory=lambda: ["random", "hard"])
    fixed_distractor_count: int = 3
    hard_distractor_pool_size: int = 10


@dataclass
class ExperimentFlags:
    run_experiment_a: bool = True
    run_experiment_b: bool = True
    run_experiment_c: bool = True
    run_experiment_d: bool = True
    top_k_filter: int = 3
    run_retrieval_realistic: bool = True
    realistic_retrieval_k: List[int] = field(default_factory=lambda: [1, 3, 5, 10])


@dataclass
class EvaluationConfig:
    metrics: List[str] = field(default_factory=lambda: ["f1", "exact_match"])
    bootstrap_iterations: int = 1000
    confidence_level: float = 0.95
    bootstrap_seed: int = 42


@dataclass
class StorageConfig:
    output_dir: str = "./outputs/runs"
    cache_inference: bool = True


@dataclass
class AppConfig:
    name: str = "robustrag_experiment"
    description: str = "Empirical study on RAG noise robustness and reranking"
    random_seed: int = 42
    is_smoke_test: bool = False
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)
    reranker: RerankerConfig = field(default_factory=RerankerConfig)
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    experiments: ExperimentFlags = field(default_factory=ExperimentFlags)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AppConfig":
        """Instantiate AppConfig from a dictionary with nested schema parsing."""
        exp_meta = data.get("experiment", {})
        ds_data = data.get("dataset", {})
        models = data.get("models", {})
        gen_data = models.get("generator", {})
        rerank_data = models.get("reranker", {})
        noise_data = data.get("noise_settings", {})
        flags_data = data.get("experiments", {})
        eval_data = data.get("evaluation", {})
        storage_data = data.get("storage", {})

        return cls(
            name=exp_meta.get("name", "robustrag_experiment"),
            description=exp_meta.get("description", ""),
            random_seed=exp_meta.get("random_seed", 42),
            is_smoke_test=exp_meta.get("is_smoke_test", False),
            dataset=DatasetConfig(**ds_data),
            generator=GeneratorConfig(**gen_data),
            reranker=RerankerConfig(**rerank_data),
            noise=NoiseConfig(**noise_data),
            experiments=ExperimentFlags(**flags_data),
            evaluation=EvaluationConfig(**eval_data),
            storage=StorageConfig(**storage_data),
        )

    @classmethod
    def from_yaml(cls, path: Path | str) -> "AppConfig":
        """Load and parse YAML configuration file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls.from_dict(data)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration hierarchy to dictionary."""
        return {
            "experiment": {
                "name": self.name,
                "description": self.description,
                "random_seed": self.random_seed,
                "is_smoke_test": self.is_smoke_test,
            },
            "dataset": {
                "name": self.dataset.name,
                "split": self.dataset.split,
                "sample_size": self.dataset.sample_size,
                "cache_dir": self.dataset.cache_dir,
                "fallback_path": self.dataset.fallback_path,
            },
            "models": {
                "generator": {
                    "backend": self.generator.backend,
                    "model_name": self.generator.model_name,
                    "groq_model": self.generator.groq_model,
                    "max_input_length": self.generator.max_input_length,
                    "max_new_tokens": self.generator.max_new_tokens,
                    "temperature": self.generator.temperature,
                    "do_sample": self.generator.do_sample,
                },
                "reranker": {
                    "backend": self.reranker.backend,
                    "model_name": self.reranker.model_name,
                    "batch_size": self.reranker.batch_size,
                },
            },
            "noise_settings": {
                "quantities": self.noise.quantities,
                "distractor_types": self.noise.distractor_types,
                "fixed_distractor_count": self.noise.fixed_distractor_count,
                "hard_distractor_pool_size": self.noise.hard_distractor_pool_size,
            },
            "experiments": {
                "run_experiment_a": self.experiments.run_experiment_a,
                "run_experiment_b": self.experiments.run_experiment_b,
                "run_experiment_c": self.experiments.run_experiment_c,
                "run_experiment_d": self.experiments.run_experiment_d,
                "top_k_filter": self.experiments.top_k_filter,
                "run_retrieval_realistic": self.experiments.run_retrieval_realistic,
                "realistic_retrieval_k": self.experiments.realistic_retrieval_k,
            },
            "evaluation": {
                "metrics": self.evaluation.metrics,
                "bootstrap_iterations": self.evaluation.bootstrap_iterations,
                "confidence_level": self.evaluation.confidence_level,
                "bootstrap_seed": self.evaluation.bootstrap_seed,
            },
            "storage": {
                "output_dir": self.storage.output_dir,
                "cache_inference": self.storage.cache_inference,
            },
        }

    def save_yaml(self, path: Path | str) -> None:
        """Serialize configuration to a YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    def estimate_generator_calls(self) -> int:
        """Compute the exact theoretical number of generator calls for this configuration.

        Calculated directly from the active conditions and question sample size.
        """
        n = self.dataset.sample_size
        conditions = 0

        # Exp A: quantities without reranking
        # e.g. [0, 1, 3, 5] -> 4 conditions
        if self.experiments.run_experiment_a:
            conditions += len(self.noise.quantities)

        # Exp B: distractor type (hard distractors) at fixed count
        # (Random distractor condition at fixed count is already part of Exp A or C)
        if self.experiments.run_experiment_b:
            conditions += 1

        # Exp C: cross-encoder reranking across non-zero noise levels
        # (0 noise clean condition is invariant under reranking since there's only 1 passage)
        if self.experiments.run_experiment_c:
            noisy_counts = [q for q in self.noise.quantities if q > 0]
            conditions += len(noisy_counts)

        # Exp D: top-k filtering ablation
        if self.experiments.run_experiment_d:
            conditions += 1

        # Optional realistic retrieval
        if self.experiments.run_retrieval_realistic:
            conditions += 1

        return n * max(1, conditions)

    def validate(self) -> List[str]:
        """Validate configuration values and return a list of warning/error messages."""
        errors = []
        if self.dataset.sample_size <= 0:
            errors.append("Sample size must be greater than 0.")
        if self.generator.backend not in ["local", "groq", "mock"]:
            errors.append(f"Invalid generator backend: {self.generator.backend}")
        if self.reranker.backend not in ["local", "mock"]:
            errors.append(f"Invalid reranker backend: {self.reranker.backend}")
        if not self.noise.quantities or min(self.noise.quantities) < 0:
            errors.append("Noise quantities must contain non-negative integers.")
        if 0 not in self.noise.quantities:
            errors.append("Noise quantities should include 0 for the clean gold baseline.")
        return errors
