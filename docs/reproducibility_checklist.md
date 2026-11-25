# RobustRAG Reproducibility Checklist

This checklist adheres to empirical NLP standards outlined in the NeurIPS ML Reproducibility Checklist and ACL Methodology Guidelines.

---

## 1. Experimental Integrity & Data Handling
- [x] **Verified Dataset Provenance**: SQuAD 1.1 official validation split used with verified schema.
- [x] **Deterministic Question Sampling**: Fixed random seeds (`seed=42`) used for subset extraction.
- [x] **Identical Question Preservation**: The exact same question IDs are evaluated across all conditions.
- [x] **Single-Variable Manipulation**: Each experimental block changes only one independent variable (noise count, distractor type, or reranking order).
- [x] **Deterministic Candidate Permutation**: Unranked conditions use deterministic pseudo-random shuffling and never force gold passages to the first slot.
- [x] **Strict Candidate Set Equality**: Assertions explicitly verify that $\{P_{unranked}\} \equiv \{P_{reranked}\}$ before generator calls.
- [x] **Answer-Leakage Filtering**: Hard distractor selection explicitly screens candidates containing any normalized reference answer string.
- [x] **Gold Evidence Isolation**: The main context-noise experiment is explicitly separated from the natural BM25 retrieval diagnostic.

---

## 2. Models & Generation Settings
- [x] **Deterministic Decoding**: Greedy decoding (`temperature=0.0`, `do_sample=False`) enforced across all generator models.
- [x] **Identical Prompt Construction**: Prompt templates strictly standardized across all conditions.
- [x] **Standard Model Releases**: Open-source Hugging Face identifiers (`google/flan-t5-small`, `cross-encoder/ms-marco-MiniLM-L-6-v2`) used as default configurations.
- [x] **Offline Resilience**: Offline fallback dataset provided to ensure zero execution barriers during internet disruptions.
- [x] **Model Caching**: SHA-256 prompt-context hashing guarantees identical repeated runs yield identical outputs without duplicate computation.

---

## 3. Evaluation & Statistical Rigor
- [x] **Standard SQuAD Metrics**: Official token-level F1 (precision/recall harmonic mean) and Exact Match implemented.
- [x] **Answer Normalization**: Lowercase, punctuation removal, article removal, and whitespace fixing applied to predictions and reference answers.
- [x] **Paired Statistical Testing**: 1,000-iteration paired bootstrap confidence intervals (95% CI) computed on matched question differences.
- [x] **No Fabricated Claims**: Experimental results, confidence intervals, and tables reflect genuine outputs from completed runs.
- [x] **Negative Results Transparency**: Cases where rerankers fail or hard distractors degrade output are systematically logged in the Failure Analysis module.

---

## 4. Software & Environment Tracking
- [x] **Environment Metadata Recorded**: Python version, OS platform, package versions, and timestamps saved in `metadata.json`.
- [x] **Run Immutability**: Each run receives a unique timestamped run ID preventing accidental overwriting.
- [x] **Smoke Test Flagging**: Fast mock/smoke tests are flagged explicitly and excluded from empirical research summaries.
