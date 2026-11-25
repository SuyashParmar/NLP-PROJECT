# Conference Presentation Script (10–12 Minutes)

**Title**: RobustRAG: How Irrelevant Retrieved Documents Affect Question Answering  
**Speaker**: Undergraduate NLP Researcher  
**Audience**: Faculty Review Committee & Academic Conference Attendees  

---

### Slide 1: Title & Authors (0:00 - 1:00)
> "Good morning, everyone. Today I am presenting **RobustRAG: How Irrelevant Retrieved Documents Affect Question Answering**. 
> Retrieval-Augmented Generation is the standard approach for grounding language models on factual knowledge. But what happens when the retriever makes a mistake and returns irrelevant or distracting documents? Does adding more context help, or does it actively sabotage answer quality? And can a cross-encoder reranker save us? That is the empirical focus of our research."

### Slide 2: Motivation & The Grounding Problem (1:00 - 2:30)
> "In conventional RAG, standard practice is to retrieve top-k passages—say, 5 or 10—and feed them all into the generator. 
> However, modern literature such as the SURE framework from ACL 2026 and 'The Distracting Effect' from ACL 2025 demonstrates that language models can be severely distracted by irrelevant passages, especially when those passages contain superficial keyword overlap. 
> Existing research mostly focuses on massive 70B models requiring GPU clusters. Our goal was to design a clean, fully reproducible experimental benchmark that isolates the effect of context noise on accessible, edge-scale architectures."

### Slide 3: Research Questions & Preregistered Hypotheses (2:30 - 4:00)
> "We formulated four core research questions:
> First: How steep is the degradation slope as irrelevant documents increase from 0 to 5?
> Second: Are lexically similar 'hard' distractors more harmful than random corpus passages?
> Third: Can cross-encoder reranking mitigate this degradation when evaluated on the exact same candidate documents?
> And fourth: Does truncating context after reranking improve stability?
> We preregistered our directional hypotheses before running the main experiments, predicting a monotonic drop in F1 and a significant partial recovery via cross-encoder reordering."

### Slide 4: Experimental Methodology & Scientific Controls (4:00 - 6:00)
> "Methodology is where scientific rigor matters most. We evaluated on SQuAD 1.1 validation data.
> Crucially, we separated the controlled noise stress-test from natural retrieval evaluation. In our stress-test, we lock the known gold passage and inject controlled distractors. This isolates the generator's noise tolerance independently of retriever recall.
> Furthermore, in our reranker comparison, we enforced an assertion check: the candidate document set in the unranked condition is mathematically identical to the reranked condition. Only the ordering changes. Deterministic greedy decoding was enforced across all runs."

### Slide 5: Empirical Results & Degradation Trajectories (6:00 - 8:00)
> "Looking at our empirical findings:
> In the clean baseline (E0), the model achieves high extraction precision.
> As we introduce 1, 3, and 5 random distractors in Experiment A, answer F1 declines steadily. 
> In Experiment B, hard distractors selected via BM25 lexical similarity induced significantly greater extraction confusion than random distractors, confirming our hypothesis that superficial lexical overlap seduces the reader model.
> In Experiment C, the cross-encoder successfully reorders the gold passage toward the top of the context, recovering a substantial portion of lost F1 score. Paired bootstrap confidence intervals confirm that this mitigation is statistically discernible from zero at the 95% confidence level."

### Slide 6: Failure Analysis & Qualitative Insights (8:00 - 9:30)
> "Through our automated failure taxonomy and manual case studies, we identified four key patterns:
> The most prominent was 'Clean Correct to Noisy Incorrect', where distractors diluted attention away from the true entity.
> We also observed 'Reranker Degradations', where the cross-encoder assigned high topical relevance to a distractor that shared question keywords, pushing the gold passage down and misdirecting the model."

### Slide 7: Conclusion & Reproducibility (9:30 - 10:30)
> "In conclusion:
> Context noise is not benign in RAG; adding more retrieved documents without filtering is actively detrimental.
> Pretrained cross-encoders provide an effective, compute-efficient mitigation by optimizing context ordering.
> Our entire pipeline is open-source, runs locally on consumer hardware without paid API keys, and includes an interactive Streamlit dashboard for exploration.
> Thank you, and I look forward to your questions."
