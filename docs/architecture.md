# RobustRAG System Architecture

## 1. High-Level Modular Design

RobustRAG follows a modular Python-first architecture designed for rigorous scientific isolation, modular component substitution, and local reproducibility.

```
                              ┌──────────────────────────┐
                              │  SQuAD 1.1 Dataset Split │
                              └────────────┬─────────────┘
                                           │
                                           ▼
                              ┌──────────────────────────┐
                              │    QuestionSampler       │ (Deterministic Seed=42)
                              └────────────┬─────────────┘
                                           │
                      ┌────────────────────┴────────────────────┐
                      ▼                                         ▼
         ┌─────────────────────────┐               ┌─────────────────────────┐
         │     DocumentCorpus      │               │     Gold Passage Map    │
         │  (Deduplicated Contexts)│               │ (Per Question Isolation)│
         └────────────┬────────────┘               └────────────┬────────────┘
                      │                                         │
                      ▼                                         │
         ┌─────────────────────────┐                            │
         │      BM25Retriever      │                            │
         └────────────┬────────────┘                            │
                      │                                         │
                      ▼                                         ▼
         ┌─────────────────────────┐               ┌─────────────────────────┐
         │   DistractorGenerator   │               │     Clean Baseline      │
         │  - Random Distractors   │               │     (Condition E0)      │
         │  - BM25 Hard (No Leak)  │               └────────────┬────────────┘
         └────────────┬────────────┘                            │
                      │                                         │
                      ▼                                         │
         ┌─────────────────────────┐                            │
         │ Candidate Passages Set  │                            │
         │ {1 Gold + k Distractors}│                            │
         └────────────┬────────────┘                            │
                      │                                         │
         ┌────────────┴────────────────────────┐                │
         ▼                                     ▼                │
┌───────────────────┐               ┌─────────────────────┐     │
│   Condition C1:   │               │    Condition C2:    │     │
│  Unranked Order   │               │ Cross-Encoder Order │     │
│ (Det. Permutation)│               │  (Descending Logit) │     │
└────────┬──────────┘               └──────────┬──────────┘     │
         │                                     │                │
         │                                     ▼                │
         │                          ┌─────────────────────┐     │
         │                          │    Condition D:     │     │
         │                          │ Top-k Context Filter│     │
         │                          └──────────┬──────────┘     │
         │                                     │                │
         └──────────────────┬──────────────────┘                │
                            ▼                                   │
               ┌─────────────────────────┐                      │
               │      BaseGenerator      │ <────────────────────┘
               │ (FLAN-T5 / Groq / Mock) │
               └────────────┬────────────┘
                            │
                            ▼
               ┌─────────────────────────┐
               │  Evaluation & Metrics   │
               │ - Token F1 & Exact Match│
               │ - Paired Bootstrap CI   │
               └────────────┬────────────┘
                            │
                            ▼
               ┌─────────────────────────┐
               │    ExperimentStore      │
               │ (CSV, JSON, LaTeX, PNG) │
               └─────────────────────────┘
```

---

## 2. Core Modules Breakdown

1. **`data_loader.py`**:
   - `DatasetLoader`: Fetches SQuAD 1.1 validation examples via Hugging Face `datasets` with namespace fallbacks and local cached JSON fallback.
2. **`sampling.py`**:
   - `QuestionSampler`: Selects reproducible subsets and serializes `SamplingManifest` with exclusion tracking.
3. **`corpus.py`**:
   - `DocumentCorpus`: Extracts unique contexts, hashes passage IDs, and builds the question-to-gold-passage index.
4. **`retrieval.py`**:
   - `BM25Retriever`: Implements lexical indexing using `rank-bm25` (BM25Okapi) and computes unconfounded Recall@k diagnostics.
5. **`distractors.py`**:
   - `DistractorGenerator`: Selects random corpus passages (excluding gold) or BM25 hard distractors (excluding gold and reference-answer substrings).
6. **`reranking.py`**:
   - `CrossEncoderReranker`: SentenceTransformers wrapper (`cross-encoder/ms-marco-MiniLM-L-6-v2`).
   - `order_passages_unranked`: Deterministic shuffling independent of gold relevance.
   - `verify_identical_candidate_sets`: Mathematical candidate-equality assertion.
7. **`generation.py`**:
   - `LocalHuggingFaceGenerator`: Greedy decoding on `google/flan-t5-small`.
   - `GroqGenerator`: Accelerated cloud inference on `llama-3.1-8b-instant`.
   - `MockGenerator`: Deterministic double for tests.
   - Persistent SHA-256 prompt-context disk cache.
8. **`metrics.py`**:
   - SQuAD normalization and token-level F1 / EM evaluation.
9. **`statistics.py`**:
   - Non-parametric paired bootstrap confidence intervals (95% CI).
10. **`failures.py`**:
    - `FailureAnalyzer`: Automated taxonomy classification and researcher annotation persistence.
11. **`storage.py`**:
    - `ExperimentStore`: Multi-asset persistence per run ID.
12. **`reporting.py`**:
    - Scientific Matplotlib figure rendering and IEEE LaTeX table generation.
