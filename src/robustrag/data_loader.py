"""Data loading module for SQuAD 1.1 with caching and offline fallback support."""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("robustrag.data_loader")


@dataclass
class QAExample:
    """Represents a single verified Question-Answering record from SQuAD."""
    id: str
    title: str
    context: str
    question: str
    answers: List[str]
    answer_starts: List[int] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "context": self.context,
            "question": self.question,
            "answers": self.answers,
            "answer_starts": self.answer_starts,
        }


class DatasetLoader:
    """Loads and standardizes SQuAD 1.1 validation data from Hugging Face or local cache."""

    def __init__(
        self,
        dataset_name: str = "squad",
        split: str = "validation",
        cache_dir: Optional[str] = "./.cache/datasets",
        fallback_path: Optional[str] = "./data/squad_sample_fallback.json",
    ):
        self.dataset_name = dataset_name
        self.split = split
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.fallback_path = Path(fallback_path) if fallback_path else None

    def load(self) -> List[QAExample]:
        """Load raw dataset split, normalize into QAExample objects, and validate fields."""
        examples: List[QAExample] = []
        loaded_from = "none"

        # 1. Attempt Hugging Face datasets load
        try:
            from datasets import load_dataset
            cache_path = str(self.cache_dir) if self.cache_dir else None
            names_to_try = [self.dataset_name]
            if "/" not in self.dataset_name and self.dataset_name == "squad":
                names_to_try.append("rajpurkar/squad")

            ds = None
            for candidate_name in names_to_try:
                try:
                    logger.info("Attempting to load %s (%s split) via Hugging Face...", candidate_name, self.split)
                    ds = load_dataset(candidate_name, split=self.split, cache_dir=cache_path)
                    if ds:
                        loaded_from = f"Hugging Face ({candidate_name}/{self.split})"
                        break
                except Exception:
                    continue

            if ds is not None:
                for item in ds:
                    ex = self._parse_squad_item(item)
                    if ex:
                        examples.append(ex)
                logger.info("Successfully loaded %d records from %s", len(examples), loaded_from)
                return examples
        except Exception as e:
            logger.warning("Could not load dataset from Hugging Face hub (%s). Attempting local fallback.", e)

        # 2. Attempt local fallback if network/hub load failed
        if self.fallback_path and self.fallback_path.exists():
            try:
                with open(self.fallback_path, "r", encoding="utf-8") as f:
                    raw_data = json.load(f)
                for item in raw_data:
                    ex = self._parse_squad_item(item)
                    if ex:
                        examples.append(ex)
                loaded_from = f"Local Fallback ({self.fallback_path})"
                logger.info("Successfully loaded %d records from fallback %s", len(examples), self.fallback_path)
                return examples
            except Exception as fe:
                logger.error("Failed to load local fallback: %s", fe)

        if not examples:
            raise RuntimeError(
                f"Unable to load dataset '{self.dataset_name}' from Hugging Face Hub, and "
                f"fallback path '{self.fallback_path}' is unavailable or invalid."
            )
        return examples

    def _parse_squad_item(self, item: Dict[str, Any]) -> Optional[QAExample]:
        """Parse and sanitize a raw dictionary entry into a verified QAExample."""
        try:
            q_id = str(item.get("id", "")).strip()
            title = str(item.get("title", "Unknown")).strip()
            context = str(item.get("context", "")).strip()
            question = str(item.get("question", "")).strip()

            # Discard invalid or empty entries
            if not q_id or not context or not question:
                return None

            raw_answers = item.get("answers", {})
            answers_list: List[str] = []
            starts_list: List[int] = []

            if isinstance(raw_answers, dict):
                text_field = raw_answers.get("text", [])
                start_field = raw_answers.get("answer_start", [])
                if isinstance(text_field, list):
                    answers_list = [str(a).strip() for a in text_field if str(a).strip()]
                if isinstance(start_field, list):
                    starts_list = [int(s) for s in start_field if isinstance(s, (int, float))]
            elif isinstance(raw_answers, list):
                for a in raw_answers:
                    if isinstance(a, str) and a.strip():
                        answers_list.append(a.strip())
                    elif isinstance(a, dict) and "text" in a:
                        answers_list.append(str(a["text"]).strip())

            # Ensure at least one accepted reference answer exists
            if not answers_list:
                return None

            return QAExample(
                id=q_id,
                title=title,
                context=context,
                question=question,
                answers=answers_list,
                answer_starts=starts_list,
            )
        except Exception:
            return None
