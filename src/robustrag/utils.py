"""Utility functions for RobustRAG: determinism, hashing, logging, and environment."""

import hashlib
import json
import logging
import os
import random
from pathlib import Path
from typing import Dict, Optional

import numpy as np


def set_seed(seed: int = 42) -> None:
    """Set random seeds across standard library, numpy, and torch for determinism."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def compute_cache_key(
    question: str,
    context: str,
    prompt: str,
    model_name: str,
    temperature: float = 0.0,
    max_new_tokens: int = 32
) -> str:
    """Compute a deterministic SHA-256 hash representing a specific generator inference call."""
    payload = {
        "question": question.strip(),
        "context": context.strip(),
        "prompt": prompt.strip(),
        "model_name": model_name,
        "temperature": round(temperature, 4),
        "max_new_tokens": max_new_tokens
    }
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def setup_logger(name: str = "robustrag", log_file: Optional[Path] = None, level: int = logging.INFO) -> logging.Logger:
    """Configure a clean research logger with stream and optional file handlers."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Prevent duplicate handlers
    if not logger.handlers:
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        if log_file:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(str(log_file), mode="a", encoding="utf-8")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger


def load_env_variables() -> Dict[str, str]:
    """Load environment variables from .env file if available."""
    env_vars: Dict[str, str] = {}
    env_file = Path(".env")
    if env_file.exists():
        try:
            with open(env_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        env_vars[k.strip()] = v.strip().strip("'\"")
                        if k.strip() not in os.environ:
                            os.environ[k.strip()] = v.strip().strip("'\"")
        except Exception:
            pass
    return env_vars


def build_qa_prompt(question: str, context_passages: list[str]) -> str:
    """Format standard context-grounded Question Answering prompt.

    Instructs the model to answer using only the provided context and avoid fabricating facts.
    """
    joined_context = "\n\n".join(
        [f"[Passage {i+1}]: {p.strip()}" for i, p in enumerate(context_passages)]
    )
    prompt = (
        "Answer the question based strictly on the provided context passages. "
        "If the answer cannot be found in the context, respond with 'Unanswerable'.\n\n"
        f"Context:\n{joined_context}\n\n"
        f"Question: {question.strip()}\n"
        "Answer:"
    )
    return prompt
