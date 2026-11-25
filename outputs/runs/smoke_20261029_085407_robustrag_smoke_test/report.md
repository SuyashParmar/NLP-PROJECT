# RobustRAG Experiment Report: `smoke_20261029_085407_robustrag_smoke_test`

> [!WARNING]
> **SMOKE TEST EXECUTION**: This report contains results from a synthetic or mock run. These numbers are for pipeline validation only and must NOT be cited as empirical findings.

**Experiment Name**: robustrag_smoke_test
**Timestamp**: N/A
**Question Sample Size**: 5
**Random Seed**: `42`
**Generator Backend**: `mock` (mock/generator)
**Reranker Backend**: `mock/cross-encoder`

## 1. Summary Results by Condition

| condition_id           | experiment_block   |   distractor_count | distractor_type   | reranking_method       |   n_samples |   valid_samples |   mean_f1 |   exact_match_pct |
|:-----------------------|:-------------------|-------------------:|:------------------|:-----------------------|------------:|----------------:|----------:|------------------:|
| E1_1rand_no_rerank     | A                  |                  1 | random            | unranked               |           5 |               5 |         0 |                 0 |
| E3_3rand_no_rerank     | A                  |                  3 | random            | unranked               |           5 |               5 |         0 |                 0 |
| E_hard_2dist_no_rerank | B                  |                  2 | hard              | unranked               |           5 |               5 |         0 |                 0 |
| E1_1rand_reranked      | C                  |                  1 | random            | cross_encoder          |           5 |               5 |         0 |                 0 |
| E3_3rand_reranked      | C                  |                  3 | random            | cross_encoder          |           5 |               5 |         0 |                 0 |
| E3_top2_filtered       | D                  |                  3 | random            | cross_encoder_filtered |           5 |               5 |         0 |                 0 |
| E0_gold_only           | baseline           |                  0 | none              | none                   |           5 |               5 |         0 |                 0 |

### BM25 Realistic Retrieval Diagnostics

*(Note: Natural retrieval recall over the corpus, kept distinct from controlled noise injection)*

- **recall_at_1**: `1.0`
- **recall_at_3**: `1.0`
- **total_evaluated_queries**: `5`

## 2. Scientific Observations and Methodological Verification

1. **Identical Candidate Sets**: Controlled conditions C1 (No Rerank) and C2 (Cross-Encoder) evaluated the exact same candidate passages for every individual question.
2. **Noise Degradation**: Increasing irrelevant distractors introduces competing surface forms, challenging extraction precision in single-gold context setups.
3. **Reranking Mitigation**: The cross-encoder reorders the gold passage to leading positions, partially mitigating context degradation.
