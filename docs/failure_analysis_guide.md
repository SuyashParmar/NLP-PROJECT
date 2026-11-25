# Failure Analysis Guide & Annotation Taxonomy

This guide instructs researchers on reviewing, categorizing, and interpreting generation failures extracted during RobustRAG experiments.

---

## 1. Automated Failure Taxonomy

The `FailureAnalyzer` module flags and segments question predictions into four primary transition categories:

### Category 1: `Clean Correct -> Noisy Incorrect`
- **Definition**: The model answered correctly (F1 $\ge 0.8$) when presented with clean gold context (Condition E0), but degraded to incorrect (F1 $< 0.5$) when distractors were introduced.
- **Diagnostic Cause**: Attention dilution or distractor interference. The extraneous text distracted the attention heads from the factual span.

### Category 2: `Reranking Restored Correct Answer`
- **Definition**: The unranked noisy condition failed (F1 $< 0.5$), but cross-encoder reranking recovered a correct answer (F1 $\ge 0.8$) on the **exact same candidate documents**.
- **Diagnostic Cause**: Primacy positioning. The cross-encoder promoted the gold passage into top positions, countering distractors that previously preceded gold evidence.

### Category 3: `Reranking Hurt Previously Correct Answer`
- **Definition**: The unranked condition succeeded (F1 $\ge 0.8$), but cross-encoder reranking induced an extraction error (F1 $< 0.5$).
- **Diagnostic Cause**: Reranker false positive. The cross-encoder misestimated a distractor's topical relevance higher than the gold passage, pushing gold evidence down the context window.

### Category 4: `Hard Distractor Seduction (BM25 Confusion)`
- **Definition**: Random distractors did not degrade the answer (F1 $\ge 0.7$), but BM25-selected hard distractors caused failure (F1 $< 0.4$).
- **Diagnostic Cause**: Spurious lexical overlap. The hard distractor contained entity mentions and keywords matching the question, seducing the reader into extracting an irrelevant entity.

---

## 2. Researcher Qualitative Annotation Protocol

In the Streamlit **Failure Analysis** dashboard:
1. Filter by category or search specific question IDs.
2. Inspect the **Passage IDs in Context Window** and compare the model's prediction against accepted reference answers.
3. Record a concise diagnostic explanation in `researcher_notes`. Address:
   - *Which specific distractor span caused the distraction?*
   - *Did the model extract an entity of the correct semantic type (e.g., date vs. team name)?*
   - *Did the reranker rank gold evidence below distractors?*
4. Click **Save Researcher Annotations** to write updates directly to `failure_cases.csv`.
