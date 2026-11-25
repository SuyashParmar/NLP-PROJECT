# Research Gap Analysis

## 1. Context and Problem Statement
Retrieval-Augmented Generation (RAG) is increasingly deployed across sensitive enterprise, medical, and legal domains where factual precision is paramount. While industrial literature emphasizes retrieving larger numbers of documents ($k \ge 10$) to maximize recall, recent academic investigations (e.g., *The Distracting Effect*, ACL 2025; *RAGGED*, ICML 2025) demonstrate that extraneous context induces hallucinations and misdirection.

## 2. Identified Literature Deficiencies

### Gap 1: Over-Reliance on Proprietary Cluster Compute
Most state-of-the-art studies benchmark 70B+ parameter models or proprietary API endpoints (e.g., GPT-4). This creates an evaluation barrier for academic reproducibility, student researchers, and edge deployments where compact, local models (e.g., `FLAN-T5-small`, `Llama-3.1-8B`) must operate within constrained compute budgets. The exact noise tolerance boundaries of compact reader models remain inadequately characterized.

### Gap 2: Confounding Retrieval Recall with Reader Noise Resilience
Existing benchmarks often evaluate the entire RAG pipeline end-to-end, measuring accuracy against raw retrieval output. This fundamentally confounds two orthogonal factors:
- **Retriever Failure**: The retriever failed to locate the relevant document.
- **Reader Failure**: The retriever found the document, but extraneous retrieved documents distracted the reader model.
Without holding the gold passage constant in a controlled noise injection setup, it is impossible to isolate the reader's intrinsic distraction susceptibility.

### Gap 3: Insufficient Candidate Isolation in Reranker Evaluations
Prior reranking studies frequently compare a pipeline using a dense retriever (fetching set $S_{raw}$) against a pipeline where a reranker fetches an expanded set $S_{expanded}$ and selects top-$k$. This introduces a confounding candidate-set discrepancy. To rigorously evaluate the ordering effect of a cross-encoder, the model must be tested on the *identical candidate set*, altering solely the permutation order of passages passed to the generator.

### Gap 4: Superficial vs. Semantically Confusing Noise
Standard distractor evaluations often use randomly sampled Wikipedia paragraphs. However, random paragraphs have distinct topics and vocabulary from the question, making them trivial for attention mechanisms to disregard. Lexically similar "hard" distractors—passages that match question keywords but do not answer the prompt—pose a far greater risk of spurious grounding (SURE framework, ACL 2026), yet are rarely benchmarked side-by-side with random noise in controlled steps.

---

## 3. The RobustRAG Contribution
RobustRAG fills these gaps by establishing an open, controlled experimental platform:
1. **Isolated Reader Stress-Testing**: Locks the gold evidence passage and systematically varies distractor quantities ($k \in \{0, 1, 3, 5\}$).
2. **Lexical Adversarial Hard Distractors**: Mines BM25 candidates with strict reference-answer exclusion filtering.
3. **Rigorous Permutation Isolation**: Asserts mathematical equality between candidate sets in unranked and reranked conditions.
4. **Statistical Inferential Validation**: Implements paired bootstrap confidence intervals (95% CI) over identical question seeds.
5. **Completely Local & Zero-Cost**: Runs on consumer laptops without requiring paid API tokens or cluster GPUs.
