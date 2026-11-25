# Undergraduate NLP Research Proposal: RobustRAG

**Project Title**: RobustRAG: How Irrelevant Retrieved Documents Affect Question Answering  
**Candidate Anchor Paper**: *Quantifying and Improving the Robustness of Retrieval-Augmented Language Models Against Spurious Features in Grounding Data* (ACL 2026, SURE Framework)  
**Target Venue**: IEEE Conference on Artificial Intelligence / Undergraduate Research Symposium  

---

## 1. Motivation & Problem Statement
Retrieval-Augmented Generation (RAG) is the dominant paradigm for grounding Large Language Models (LLMs) on factual, non-parametric knowledge. Standard RAG pipelines retrieve top-$k$ documents using lexical (e.g., BM25) or dense embedding models and concatenate them into the prompt context for reader models.

However, real-world information retrieval is inherently noisy. Retrievers frequently return passages that exhibit superficial keyword overlap with the question but contain completely irrelevant or distracting information. Recent foundational studies show that language models are not immune to context clutter: irrelevant documents compete with supportive facts, bias attention heads, and cause extraction failures or hallucinations.

While industry solutions frequently advocate adding heavier cross-encoder rerankers, the exact quantitative dynamics of noise degradation and the extent of reranker mitigation on compact, edge-deployable models remain underexplored in an accessible, reproducible framework.

---

## 2. Research Questions
1. **RQ1 (Noise Sensitivity)**: What is the degradation slope of Question Answering accuracy (token-level F1 and Exact Match) as the quantity of irrelevant passages in retrieved context increases from 0 to 5?
2. **RQ2 (Distractor Semantic Hardness)**: Do lexically similar "hard" distractors (retrieved via BM25 without containing the answer) induce significantly greater degradation than random corpus distractors?
3. **RQ3 (Reranking Mitigation)**: Can a pretrained cross-encoder reranker restore question-answering accuracy when operating on the *exact same candidate set*, simply by reorganizing passage positions?
4. **RQ4 (Context Truncation Ablation)**: Does post-reranking top-$k$ filtering yield superior performance compared to feeding all reranked documents into the reader?

---

## 3. Related Work & Anchor Paper Rationale
We surveyed five recent peer-reviewed contributions:
1. **SURE Framework (ACL 2026)**: Formalized spurious feature reliance in grounding data. Identified as our **candidate anchor paper** due to its rigorous treatment of contextual noise.
2. **The Distracting Effect (ACL 2025)**: Demonstrated that even a single irrelevant passage degrades LLM extraction.
3. **RAGGED (ICML 2025)**: Evaluated stability bounds and retrieval trade-offs.
4. **InfoGain-RAG (EMNLP 2025)**: Proposed information-gain reranking to filter redundant evidence.
5. **Astute RAG (ACL 2025)**: Analyzed knowledge conflicts when passages contradict internal facts.

**Selection Rationale**:  
While SURE provides the theoretical formulation, its empirical setup relied on massive 70B models and proprietary APIs. RobustRAG's independent contribution is to adapt these core principles into a clean, reproducible, controlled student-led investigation that isolates noise quantity, distractor type, and cross-encoder reordering on open-source compact backends (`google/flan-t5-small` and `ms-marco-MiniLM-L-6-v2`) runnable on local hardware without paid API keys.

---

## 4. Methodology & Experimental Design

### Dataset
- **SQuAD 1.1** (Validation split): Extensively benchmarked extractive QA dataset with human-annotated gold contexts and accepted reference answers.
- **Corpus Indexing**: Over 2,000 deduplicated contexts indexed with BM25Okapi.
- **Deterministic Sampling**: Fixed random seed ($S=42$) locking exact question IDs across all conditions.

### Experimental Matrix
- **Condition E0 (Clean Baseline)**: 1 Gold passage, 0 distractors.
- **Experiment A (Noise Quantity Curve)**:
  - E1: 1 Gold + 1 Random distractor
  - E2: 1 Gold + 3 Random distractors
  - E3: 1 Gold + 5 Random distractors
- **Experiment B (Distractor Type)**:
  - 1 Gold + 3 Hard distractors (BM25-similar passages, excluding gold ID and reference answers).
- **Experiment C (Reranker Intervention)**:
  - For each noise level, compare Condition C1 (Unranked deterministic shuffle) vs Condition C2 (Cross-Encoder score descending). Identical candidates strictly verified.
- **Experiment D (Filtering Ablation)**:
  - Reranked candidates truncated to Top-3 documents.

### Evaluation Metrics
- Standard SQuAD Token-level F1 (Precision, Recall, harmonic mean; max over reference answers).
- Exact Match (EM, binary 0/1; max over reference answers).
- Paired Bootstrap Confidence Intervals (95% CI, 1,000 resamples).

---

## 5. Feasibility, Compute & Risk Management
- **Hardware Requirements**: Runs seamlessly on standard laptops (CPU or Apple Silicon MPS).
- **Execution Time**: Pilot experiment (25 questions) finishes in under 3 minutes; main experiment (100 questions) in under 15 minutes.
- **No Paid API Dependency**: Employs open-source local Hugging Face weights (`FLAN-T5-small`, `MiniLM`). Optional high-speed Groq inference supported.
- **Fallback Data**: Local cached dataset ensures execution even during internet dropouts.

---

## 6. Project Timeline

| Milestone | Target Date | Status |
| :--- | :--- | :--- |
| Environment setup & Core pipeline implementation | Week 1 | ✅ Completed |
| Preregistration & Literature Review Matrix | Week 2 | ✅ Completed |
| Streamlit Experiment Management Dashboard | Week 3 | ✅ Completed |
| Empirical pilot & main evaluations | Week 4 | ✅ Verified (Smoke & Pilot ready) |
| Failure case annotation & Statistical analysis | Week 5 | ✅ In Progress |
| IEEE Paper draft compilation & Final defense | Week 6 | ✅ Draft Ready |
