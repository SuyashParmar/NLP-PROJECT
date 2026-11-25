# Limitations and Threats to Validity

Honest scientific reporting requires explicit documentation of experimental boundaries, assumptions, and potential validity threats.

---

## 1. Threats to Internal Validity
- **Candidate Reordering Effects**: In condition C1 (unranked), passages are permuted deterministically using a fixed hash of `(question_id, seed)`. Although this prevents artificial placement of the gold passage in position 0, the specific random position assigned to gold evidence in unranked lists can produce variance on small sample sizes ($N < 50$).
- **Answer-Leakage Exclusion Imperfections**: While hard distractor generation strictly filters passages containing the normalized reference answer string, a distractor might contain a partial synonym or paraphrased mention of the answer that escapes literal substring matching.
- **Cross-Encoder Score Calibration**: Cross-encoder logits represent relative pairwise orderings rather than calibrated posterior probabilities $P(\text{Relevant} \mid Q, P)$.

---

## 2. Threats to External Validity
- **Dataset Domain Specificity**: SQuAD 1.1 consists of extractive factoid questions over English Wikipedia paragraphs. These findings may not fully generalize to non-extractive abstractive QA, conversational multi-turn dialogues, or specialized domains (e.g., clinical health records, legal contracts).
- **Single-Gold Assumption**: In our controlled setup, exactly one gold passage is injected alongside $k$ distractors. Real-world open-domain queries often require aggregating multi-hop evidence across multiple disparate passages.
- **Generator Architecture Scale**: The default local reader model is `google/flan-t5-small` (~80M parameters). While compact models are ideal for analyzing baseline noise sensitivity, multi-billion parameter LLMs (e.g., Llama-3-70B) possess higher attention capacity and may exhibit different noise saturation thresholds.

---

## 3. Threats to Construct Validity
- **Token F1 Limitations**: Token-level F1 relies on surface string overlap. Semantically equivalent answers that use alternative phrasing or synonyms receive lower F1 scores unless captured by the human reference annotations.
- **Distractor Hardness Definition**: Hard distractors are operationalized via BM25 lexical overlap. Alternative definitions of hardness—such as dense bi-encoder false positives or LLM-generated counterfactuals—may introduce different failure patterns.

---

## 4. Threats to Conclusion Validity
- **Sample Size Considerations**: Pilot evaluations ($N=25$) provide rapid directional feedback but have wider confidence intervals. Final academic conclusions must be drawn from runs with $N \ge 100$ and paired bootstrap testing.
- **Multiple Hypothesis Testing**: When evaluating multiple noise conditions and distractor types simultaneously, we report exact paired confidence intervals rather than isolated p-values to preserve clarity.
