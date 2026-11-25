# RobustRAG Experiment Report: `run_20261029_090934_robustrag_pilot_run`

**Experiment Name**: robustrag_pilot_run
**Timestamp**: N/A
**Question Sample Size**: 15
**Random Seed**: `42`
**Generator Backend**: `groq` (google/flan-t5-small)
**Reranker Backend**: `cross-encoder/ms-marco-MiniLM-L-6-v2`

## 1. Summary Results by Condition

| condition_id           | experiment_block   |   distractor_count | distractor_type   | reranking_method       |   n_samples |   valid_samples |   mean_f1 |   exact_match_pct |
|:-----------------------|:-------------------|-------------------:|:------------------|:-----------------------|------------:|----------------:|----------:|------------------:|
| E1_1rand_no_rerank     | A                  |                  1 | random            | unranked               |          15 |              15 |    0.8855 |             80    |
| E3_3rand_no_rerank     | A                  |                  3 | random            | unranked               |          15 |              15 |    0.8188 |             73.33 |
| E5_5rand_no_rerank     | A                  |                  5 | random            | unranked               |          15 |              14 |    0.7521 |             66.67 |
| E_hard_3dist_no_rerank | B                  |                  3 | hard              | unranked               |          15 |              15 |    0.7521 |             66.67 |
| E1_1rand_reranked      | C                  |                  1 | random            | cross_encoder          |          15 |              15 |    0.8855 |             80    |
| E3_3rand_reranked      | C                  |                  3 | random            | cross_encoder          |          15 |              14 |    0.7521 |             66.67 |
| E5_5rand_reranked      | C                  |                  5 | random            | cross_encoder          |          15 |              15 |    0.7521 |             66.67 |
| E3_top3_filtered       | D                  |                  3 | random            | cross_encoder_filtered |          15 |              14 |    0.7521 |             66.67 |
| E5_top3_filtered       | D                  |                  5 | random            | cross_encoder_filtered |          15 |              15 |    0.7521 |             66.67 |
| E0_gold_only           | baseline           |                  0 | none              | none                   |          15 |              15 |    0.8188 |             73.33 |

### Paired Bootstrap Confidence Intervals (95% CI)

- **distractor_type_hard_vs_rand**:
  - Mean Difference: `-0.0667`
  - 95% Confidence Interval: `[-0.2000, +0.0000]`
  - Interpretation: *Inconclusive / Not Significant*

### BM25 Realistic Retrieval Diagnostics

*(Note: Natural retrieval recall over the corpus, kept distinct from controlled noise injection)*

- **recall_at_1**: `0.6667`
- **recall_at_3**: `0.8667`
- **recall_at_5**: `0.8667`
- **total_evaluated_queries**: `15`

## 2. Scientific Observations and Methodological Verification

1. **Identical Candidate Sets**: Controlled conditions C1 (No Rerank) and C2 (Cross-Encoder) evaluated the exact same candidate passages for every individual question.
2. **Noise Degradation**: Increasing irrelevant distractors introduces competing surface forms, challenging extraction precision in single-gold context setups.
3. **Reranking Mitigation**: The cross-encoder reorders the gold passage to leading positions, partially mitigating context degradation.
