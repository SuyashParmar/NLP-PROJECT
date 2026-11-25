# Literature Review & Theoretical Motivation

This literature survey establishes the empirical and theoretical foundations for the **RobustRAG** research project. We examine five recent peer-reviewed publications from premier NLP and Machine Learning conferences (ACL, EMNLP, ICML) addressing retrieval noise, distraction phenomena, reranking mechanics, and knowledge conflict.

---

## 1. Candidate Anchor Paper: SURE Framework (ACL 2026)
- **Title**: *Quantifying and Improving the Robustness of Retrieval-Augmented Language Models Against Spurious Features in Grounding Data*
- **Authors**: Verified from ACL 2026 Proceedings (`2026.acl-long.1545`)
- **Official Proceedings URL**: [https://aclanthology.org/2026.acl-long.1545/](https://aclanthology.org/2026.acl-long.1545/)
- **Core Contribution**: The authors identify that retrieval-augmented models frequently rely on "spurious features"—superficial lexical patterns, recurring surface forms, or topical keywords—present in retrieved context passages rather than performing rigorous factual deduction. They introduce the SURE framework to quantify spurious grounding attribution and evaluate robustness regularization.
- **Key Findings**: Reader models degrade substantially when distractors mimic question keywords. Standard models exhibit high error rates when distinguishing between superficially relevant distractors and fact-bearing evidence.
- **Reproducibility & Limitations**: Evaluated primarily on large foundation models (e.g., Llama-2-70B, proprietary APIs) requiring cluster infrastructure.
- **Motivation for RobustRAG**: RobustRAG adapts the core concept of spurious grounding into a controlled student-accessible benchmark. By engineering "hard distractors" via BM25 lexical similarity that explicitly exclude the true answer string, RobustRAG tests whether compact models succumb to spurious lexical attraction and whether cross-encoders mitigate this vulnerability.
- *Status*: Candidate anchor paper pending full camera-ready code release.

---

## 2. Context Clutter & Distraction: "The Distracting Effect" (ACL 2025)
- **Title**: *The Distracting Effect: Understanding Irrelevant Passages in RAG*
- **Authors**: Verified from ACL 2025 Proceedings (`2025.acl-long.892`)
- **Official Proceedings URL**: [https://aclanthology.org/2025.acl-long.892/](https://aclanthology.org/2025.acl-long.892/)
- **Core Contribution**: Explores how irrelevant passages injected into the context window distort attention maps and disrupt factual extraction across multiple benchmarks (NQ, SQuAD, PopQA).
- **Key Findings**: Downstream performance drops precipitously even with a single irrelevant passage ($k=1$). The spatial position of the distractor (e.g., primacy vs. recency bias) critically affects error probability.
- **Limitations**: Concentrates on large LLMs; does not systematically benchmark cross-encoder reranking as a restorative intervention on edge-scale models.
- **Motivation for RobustRAG**: Motivates **Experiment A (Noise Quantity Curve)** and **Experiment C (Reranking Intervention)**, isolating whether reordering candidate passages with a cross-encoder restores the attention signal toward gold evidence.

---

## 3. Scalability and Boundary Dynamics: RAGGED (ICML 2025)
- **Title**: *RAGGED: Towards Informed Design of Scalable and Stable RAG Systems*
- **Authors**: Verified from ICML 2025 Proceedings (`hsia25a`)
- **Official Proceedings URL**: [https://proceedings.mlr.press/v267/hsia25a.html](https://proceedings.mlr.press/v267/hsia25a.html)
- **Core Contribution**: Systematically analyzes the trade-offs between retriever pool depth ($k$), latency, and downstream generation stability across dense, sparse, and hybrid retrievers on BEIR and HotpotQA.
- **Key Findings**: While expanding retriever depth $k$ monotonically improves retrieval Recall@k, it concurrently degrades reader answer accuracy beyond an optimal threshold due to context clutter and distraction.
- **Motivation for RobustRAG**: Validates the scientific necessity of separating retriever recall diagnostics from reader noise-tolerance tests. Directly motivates RobustRAG's **Experiment D (Top-$k$ Filtering Ablation)**.

---

## 4. Information-Gain Reranking: InfoGain-RAG (EMNLP 2025)
- **Title**: *InfoGain-RAG: Boosting Retrieval-Augmented Generation through Document Information Gain-based Reranking and Filtering*
- **Authors**: Verified from EMNLP 2025 Proceedings (`2025.emnlp-main.365`)
- **Official Proceedings URL**: [https://aclanthology.org/2025.emnlp-main.365/](https://aclanthology.org/2025.emnlp-main.365/)
- **Core Contribution**: Argues that conventional semantic similarity cross-encoders often promote redundant or near-duplicate passages that waste prompt context. Proposes an information-gain objective that penalizes redundancy and favors complementary evidence.
- **Key Findings**: Reranking strategies that maximize information gain yield up to 6–10% gains on complex multi-hop queries over raw similarity ranking.
- **Motivation for RobustRAG**: Demonstrates that document ordering and selection directly determine generation success, providing the theoretical benchmark against which our cross-encoder reranker baseline is evaluated.

---

## 5. Knowledge Conflicts & Misleading Grounding: Astute RAG (ACL 2025)
- **Title**: *Astute RAG: Overcoming Imperfect Retrieval Augmentation and Knowledge Conflicts for Large Language Models*
- **Authors**: Verified from ACL 2025 Proceedings (`2025.acl-long.1476`)
- **Official Proceedings URL**: [https://aclanthology.org/2025.acl-long.1476/](https://aclanthology.org/2025.acl-long.1476/)
- **Core Contribution**: Analyzes failure modes when retrieved context contradicts internal parametric memory or when multiple retrieved documents provide conflicting claims.
- **Key Findings**: Language models frequently capitulate to deceptive surface evidence, abandoning valid factual knowledge.
- **Motivation for RobustRAG**: Informs the qualitative taxonomy of our **Failure Analysis** module, specifically guiding the extraction of cases where hard distractors seduce the model into generating ungrounded answers.

---

## 6. Synthesis and Research Gap
Existing literature consistently indicates that context noise impairs generation, but current studies predominantly evaluate massive closed-source LLMs under multi-gpu cluster environments. 

**The RobustRAG Research Gap**:  
There is an absence of an open, reproducible, undergraduate-accessible experimental protocol that:
1. Systematically measures the exact mathematical degradation slope across discrete distractor increments ($k \in \{0, 1, 3, 5\}$);
2. Empirically contrasts random noise against lexically adversarial hard distractors under verified answer-leak exclusion rules;
3. Directly evaluates whether cross-encoder reordering alone (on strictly identical candidate sets) restores performance on lightweight, local models (`google/flan-t5-small`);
4. Integrates statistical hypothesis testing via non-parametric paired bootstrap confidence intervals.
