"""Answer generation backends: Local HuggingFace, Groq API, and Mock with persistent caching."""

import json
import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional

import torch

from robustrag.config import GeneratorConfig
from robustrag.corpus import Passage
from robustrag.utils import build_qa_prompt, compute_cache_key, load_env_variables

logger = logging.getLogger("robustrag.generation")


class BaseGenerator(ABC):
    """Abstract interface for RAG answer generators."""

    def __init__(self, config: GeneratorConfig, cache_dir: Optional[str] = "./outputs/.cache"):
        self.config = config
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self._memory_cache: Dict[str, str] = {}
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._load_disk_cache()

    def _load_disk_cache(self) -> None:
        if not self.cache_dir:
            return
        cache_file = self.cache_dir / "generation_cache.json"
        if cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    self._memory_cache = json.load(f)
                logger.info("Loaded %d cached generations from disk.", len(self._memory_cache))
            except Exception as e:
                logger.warning("Failed to read generation cache: %s", e)

    def _persist_cache_entry(self, key: str, value: str) -> None:
        self._memory_cache[key] = value
        if self.cache_dir:
            cache_file = self.cache_dir / "generation_cache.json"
            try:
                with open(cache_file, "w", encoding="utf-8") as f:
                    json.dump(self._memory_cache, f, indent=2)
            except Exception as e:
                logger.warning("Failed to persist generation cache: %s", e)

    @abstractmethod
    def _generate_raw(self, prompt: str) -> str:
        """Subclass implementation of actual inference."""
        pass

    def generate_answer(
        self,
        question: str,
        passages: List[Passage],
        use_cache: bool = True
    ) -> str:
        """Construct prompt, check cache, perform generation, and return answer."""
        passage_texts = [p.text for p in passages]
        prompt = build_qa_prompt(question, passage_texts)
        joined_context = "\n\n".join(passage_texts)

        model_identifier = f"{self.config.backend}:{self.config.model_name}"
        cache_key = compute_cache_key(
            question=question,
            context=joined_context,
            prompt=prompt,
            model_name=model_identifier,
            temperature=self.config.temperature,
            max_new_tokens=self.config.max_new_tokens,
        )

        if use_cache and cache_key in self._memory_cache:
            return self._memory_cache[cache_key]

        answer = self._generate_raw(prompt).strip()

        if use_cache and answer:
            self._persist_cache_entry(cache_key, answer)

        return answer


class LocalHuggingFaceGenerator(BaseGenerator):
    """Local HuggingFace Seq2Seq generator (e.g. google/flan-t5-small)."""

    def __init__(self, config: GeneratorConfig, cache_dir: Optional[str] = "./outputs/.cache"):
        super().__init__(config, cache_dir)
        self.tokenizer = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        self._load_local_model()

    def _load_local_model(self) -> None:
        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            logger.info(
                "Loading local generator model '%s' on device %s...",
                self.config.model_name, self.device
            )
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_name).to(self.device)
            self.model.eval()
            logger.info("Local generator model successfully loaded and ready for inference.")
        except Exception as e:
            logger.error("Failed to load local model '%s': %s", self.config.model_name, e)
            raise RuntimeError(
                f"Could not load HuggingFace model '{self.config.model_name}'. "
                f"Details: {e}"
            ) from e

    def _generate_raw(self, prompt: str) -> str:
        try:
            inputs = self.tokenizer(
                prompt,
                max_length=self.config.max_input_length,
                truncation=True,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_new_tokens,
                    do_sample=self.config.do_sample,
                    temperature=None if not self.config.do_sample else self.config.temperature,
                )

            answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return answer
        except Exception as e:
            logger.error("Generation error in LocalHuggingFaceGenerator: %s", e)
            return ""


class GroqGenerator(BaseGenerator):
    """Ultra-fast cloud inference generator using Groq API (e.g. llama-3.1-8b-instant)."""

    def __init__(self, config: GeneratorConfig, cache_dir: Optional[str] = "./outputs/.cache"):
        super().__init__(config, cache_dir)
        load_env_variables()
        self.api_key = os.environ.get("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError(
                "GROQ_API_KEY is not set in environment or .env file. "
                "Switch backend to 'local' or provide GROQ_API_KEY."
            )
        try:
            from groq import Groq
            self.client = Groq(api_key=self.api_key)
            logger.info("Groq client initialized with model: %s", self.config.groq_model)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Groq client: {e}") from e

    def _generate_raw(self, prompt: str) -> str:
        import time
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.config.groq_model,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a precise Question Answering model. Answer questions using ONLY "
                                "the facts provided in the context passages. Be concise (1-5 words where possible). "
                                "Do not hallucinate or add explanations."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_new_tokens,
                )
                ans = response.choices[0].message.content or ""
                return ans.strip()
            except Exception as e:
                logger.warning("Groq attempt %d/%d failed: %s", attempt + 1, max_retries, e)
                if attempt < max_retries - 1:
                    time.sleep(1.5 * (attempt + 1))
                else:
                    logger.error("All Groq attempts failed: %s", e)
                    return ""
        return ""


class MockGenerator(BaseGenerator):
    """Fast deterministic mock generator for unit tests and rapid smoke tests.

    Extracts high-overlap substrings from candidate passages to simulate realistic QA.
    """

    def _generate_raw(self, prompt: str) -> str:
        if "Context:" not in prompt or "Question:" not in prompt:
            return "Mock Answer"
        try:
            parts = prompt.split("Question:")
            q_part = parts[1].split("Answer:")[0].strip()
            c_part = parts[0].split("Context:")[1].strip()

            # Find matching passage sentences
            for line in c_part.split("\n"):
                if any(w in line.lower() for w in q_part.lower().split() if len(w) > 4):
                    words = line.split()
                    if len(words) > 3:
                        return " ".join(words[-4:]).rstrip(".:,;")
            return "Mock Extracted Answer"
        except Exception:
            return "Mock Answer"


def build_generator(config: GeneratorConfig, cache_dir: Optional[str] = "./outputs/.cache") -> BaseGenerator:
    """Factory function to build appropriate generator based on configuration."""
    backend = config.backend.lower()
    if backend == "local":
        return LocalHuggingFaceGenerator(config, cache_dir=cache_dir)
    elif backend == "groq":
        return GroqGenerator(config, cache_dir=cache_dir)
    elif backend == "mock":
        return MockGenerator(config, cache_dir=cache_dir)
    else:
        raise ValueError(f"Unknown generator backend: {config.backend}")
