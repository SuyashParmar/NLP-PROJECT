# RobustRAG: Empirical Study Preregistration

**Protocol Identifier**: `PREREG-ROBUSTRAG-2026-V1`  
**Date of Preregistration**: October 29, 2026  
**Document Version**: 1.0 (Initial Preregistration prior to full empirical evaluation)  
**Authors**: RobustRAG Undergraduate NLP Research Group  

---

## 1. Study Title & Core Research Question

- **Project Title**: RobustRAG: How Irrelevant Retrieved Documents Affect Question Answering
- **Research Question**:  
  *How does increasing the number of irrelevant documents in retrieved context affect question-answering quality, and to what extent can a pretrained cross-encoder reranking mechanism mitigate the resulting degradation?*

---

## 2. Theoretical Hypothesis & Directional Predictions

### Primary Hypothesis ($H_1$)
Injecting irrelevant passages into the reader model's context window creates competing surface signals that distract the generation model, resulting in a monotonic decline in answer quality (token-level F1 and Exact Match).

### Secondary Mitigation Hypothesis ($H_2$)
Applying a pretrained cross-encoder reranker (`cross-encoder/ms-marco-MiniLM-L-6-v2`) to reorder the candidate context will significantly mitigate noise-induced degradation by prioritizing the gold passage in leading positions, even when evaluated on identical candidate passage sets.

### Tertiary Distractor Type Hypothesis ($H_3$)
Hard distractors (lexically similar passages retrieved via BM25 that do not contain the answer) will induce greater performance degradation than random corpus distractors at the same context quantity, because superficial lexical overlap mimics genuine supportive evidence.

---

## 3. Predicted Numerical Changes (Preregistered Estimates)

*Note: In accordance with scientific integrity guidelines, researcher predictions are logged here prior to examining full experimental outcomes.*

| Condition Comparison | Metric | Predicted Direction | Researcher Predicted Approximate $\Delta$ |
| :--- | :--- | :--- | :--- |
| **Noise Degradation (E3: 5 random vs E0: Clean Gold)** | Token F1 | Negative (Degradation) | `[ Fill prior to main run: ~ -15% to -25% F1 ]` |
| **Reranking Mitigation (E2: 3 random Reranked vs Unranked)** | Token F1 | Positive (Improvement) | `[ Fill prior to main run: ~ +8% to +14% F1 ]` |
| **Distractor Harm (E_hard: 3 Hard vs E2: 3 Random)** | Token F1 | Negative (Harder) | `[ Fill prior to main run: ~ -5% to -10% F1 ]` |
| **Top-k Filter Ablation (Top-3 Filtered vs Full Candidate Set)** | Token F1 | Positive (Cleaner context)| `[ Fill prior to main run: ~ +2% to +5% F1 ]` |

---

## 4. Variables Specification

### Independent Variables
1. **Distractor Quantity ($k$)**: Number of irrelevant passages added to gold context ($k \in \{0, 1, 3, 5\}$).
2. **Distractor Type**: Random corpus sampling vs. BM25 lexical similarity (Hard Distractor).
3. **Reranking Intervention**: Unranked deterministic baseline vs. Cross-Encoder reranking (`ms-marco-MiniLM-L-6-v2`).
4. **Context Filtering**: Full candidate context vs. Top-$k$ truncated context ($k=3$).

### Dependent Variables
1. **Primary Metric**: SQuAD Token-level F1 (Precision, Recall, harmonic mean; max over reference answers).
2. **Secondary Metric**: SQuAD Exact Match (EM, binary 0/1; max over reference answers).
3. **Diagnostic Metric**: BM25 Retrieval Recall@k ($k \in \{1, 3, 5, 10\}$) on separate natural retrieval test.

### Controlled Variables
- **Question Sample**: Exact fixed subset of questions sampled from SQuAD 1.1 validation split (Seed = 42).
- **Candidate Passage Sets**: Exact set of passage IDs strictly conserved between unranked and reranked conditions.
- **Generator Model**: Fixed local model (`google/flan-t5-small`) or Groq model (`llama-3.1-8b-instant`).
- **Prompt Template**: Strict standardized context-grounding template across all conditions.
- **Decoding**: Greedy deterministic decoding (`temperature = 0.0`, `do_sample = False`).
- **Answer Normalization**: Lowercase, punctuation removal, article removal, whitespace normalization.

---

## 5. Statistical Analysis Plan

- **Descriptive Statistics**: Condition means for F1 and EM percentages.
- **Inferential Testing**: Non-parametric paired bootstrap confidence intervals (95% CI) computed across 1,000 resamples over matched questions.
- **Decision Rule**: A performance difference is deemed statistically discernible from noise if and only if the 95% bootstrap confidence interval strictly excludes zero ($\Delta_{lower} > 0$ or $\Delta_{upper} < 0$).

---

## 6. Deviations and Justifications Log

*(Any deviations from this preregistered protocol must be documented below with date and rationale.)*

| Date | Protocol Section | Change Description | Justification |
| :--- | :--- | :--- | :--- |
| *None* | — | *Initial protocol established* | *N/A* |
