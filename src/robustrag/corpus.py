"""Corpus construction, deduplication, and passage indexing for RobustRAG."""

import hashlib
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd

from robustrag.data_loader import QAExample

logger = logging.getLogger("robustrag.corpus")


@dataclass
class Passage:
    """Represents a unique context passage stored in the research corpus."""
    passage_id: str
    title: str
    text: str
    char_length: int
    dataset_name: str = "squad"


class DocumentCorpus:
    """Indexed document corpus with deduplication and question-to-gold-passage mappings."""

    def __init__(self, dataset_name: str = "squad"):
        self.dataset_name = dataset_name
        self.passages: Dict[str, Passage] = {}
        self.question_to_gold_id: Dict[str, str] = {}
        self.raw_duplicate_count: int = 0

    @staticmethod
    def compute_passage_id(title: str, text: str) -> str:
        """Generate deterministic 12-character hex ID from title and text."""
        normalized = f"{title.strip().lower()}:::{text.strip()}".encode("utf-8")
        return hashlib.sha256(normalized).hexdigest()[:12]

    def build_from_examples(self, examples: List[QAExample]) -> None:
        """Construct corpus from QA examples, deduplicating identical context texts."""
        seen_texts: Dict[str, str] = {}  # text -> passage_id

        for ex in examples:
            clean_text = ex.context.strip()
            if not clean_text:
                continue

            if clean_text in seen_texts:
                pid = seen_texts[clean_text]
                self.raw_duplicate_count += 1
            else:
                pid = self.compute_passage_id(ex.title, clean_text)
                passage = Passage(
                    passage_id=pid,
                    title=ex.title,
                    text=clean_text,
                    char_length=len(clean_text),
                    dataset_name=self.dataset_name,
                )
                self.passages[pid] = passage
                seen_texts[clean_text] = pid

            # Link question to its designated gold passage
            self.question_to_gold_id[ex.id] = pid

        logger.info(
            "Corpus built: %d unique passages indexed from %d questions (%d duplicates merged)",
            len(self.passages), len(examples), self.raw_duplicate_count
        )

    def get_passage(self, passage_id: str) -> Optional[Passage]:
        return self.passages.get(passage_id)

    def get_gold_passage(self, question_id: str) -> Optional[Passage]:
        pid = self.question_to_gold_id.get(question_id)
        if pid:
            return self.passages.get(pid)
        return None

    def get_all_passage_ids(self) -> List[str]:
        return sorted(list(self.passages.keys()))

    def to_dataframe(self) -> pd.DataFrame:
        """Convert corpus passages to a pandas DataFrame for inspection and export."""
        records = [
            {
                "passage_id": p.passage_id,
                "title": p.title,
                "char_length": p.char_length,
                "text_snippet": p.text[:120] + "..." if len(p.text) > 120 else p.text,
            }
            for p in self.passages.values()
        ]
        return pd.DataFrame(records)
