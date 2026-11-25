# RobustRAG: How Irrelevant Retrieved Documents Affect Question Answering

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.24+-FF4B4B.svg)](https://streamlit.io)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)](https://pytorch.org)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-orange.svg)](https://huggingface.co)

An end-to-end, reproducible undergraduate Natural Language Processing (NLP) research project investigating the degradation dynamics of Retrieval-Augmented Generation (RAG) when extraneous, distracting context documents are introduced, and evaluating cross-encoder reranking as a restorative mitigation.

---

## 1. Project Overview & Research Objectives

- **Full Title**: RobustRAG: How Irrelevant Retrieved Documents Affect Question Answering
- **Research Domain**: Retrieval-Augmented Generation (RAG) Robustness & Reliability
- **Central Research Question**:
  > *How does increasing the number of irrelevant documents in retrieved context affect question-answering quality, and to what extent can a cross-encoder reranking mechanism mitigate the resulting degradation?*
- **Scientific Hypothesis**:
  > Increasing the quantity of irrelevant distractor passages will monotonically reduce answer extraction quality (measured by SQuAD token-level F1 and Exact Match), as distracting evidence competes with gold facts. A pretrained cross-encoder reranker mitigates this degradation by ordering supportive documents into leading positions in the context window.
- **Candidate Anchor Paper**:
  > *“Quantifying and Improving the Robustness of Retrieval-Augmented Language Models Against Spurious Features in Grounding Data”* (ACL 2026, SURE framework). [Official Proceedings](https://aclanthology.org/2026.acl-long.1545/)

---

## 2. Independent Research Contribution
Unlike conventional RAG tutorials or simple chatbots, **RobustRAG** is an empirical research laboratory:
1. **Isolated Context-Noise Stress Testing**: Evaluates reader model distraction independently of retriever recall by locking the gold evidence passage and injecting discrete distractor levels ($k \in \{0, 1, 3, 5\}$).
2. **Distractor Semantic Hardness**: Contrasts random corpus passages against lexically adversarial "hard" distractors mined via BM25 that do not contain the reference answer.
3. **Rigorous Document Permutation Isolation**: Asserts mathematical equality between candidate sets in unranked and reranked conditions ($\{P_{unranked}\} \equiv \{P_{reranked}\}$), isolating the pure ordering effect.
4. **Statistical Inferential Testing**: Computes 1,000-resample paired bootstrap confidence intervals (95% CI) over identical question seeds.
5. **Completely Local & Zero-Cost**: Runs locally on consumer laptops (CPU or Apple Silicon MPS) without requiring paid API tokens.

---

## 3. Architecture & Pipeline Overview

```
[SQuAD 1.1 Validation Split]
             │
             ▼
   [QuestionSampler (Seed=42)]
             │
             ├──> [Gold Passage] (1 Passage)
             │
             ├──> [Distractor Selection]
             │      ├── Random Corpus Sampling
             │      └── BM25 Hard (Answer Filtered)
             │
             ▼
[Candidate Passages Set] (1 Gold + k Distractors)
             │
       ┌─────┴─────────────────────────────────┐
       ▼                                       ▼
Condition C1: Unranked              Condition C2: Cross-Encoder
(Deterministic Permutation)         (ms-marco-MiniLM-L-6-v2)
       │                                       │
       └───────────────────┬───────────────────┘
                           ▼
                  [Answer Generator]
              (FLAN-T5-Small / Groq LLM)
                           │
                           ▼
            [Evaluation & Statistics]
        (Token F1, Exact Match, 95% CI)
                           │
                           ▼
            [Experiment Storage & UI]
          (CSV, JSON, LaTeX, Streamlit)
```

---

## 4. Repository Structure

```
RobustRAG/
├── app.py                         # Streamlit research dashboard home
├── README.md                      # Project documentation
├── LICENSE                        # MIT License
├── requirements.txt               # Dependencies
├── pyproject.toml                 # Ruff & Pytest build config
├── .env.example                   # Environment variable template
├── configs/
│   ├── default.yaml               # Full 100-question research configuration
│   ├── pilot.yaml                 # 25-question rapid pilot configuration
│   └── smoke_test.yaml            # 5-question mock smoke test configuration
├── src/robustrag/                 # Core Python research package
│   ├── __init__.py
│   ├── config.py                  # YAML/Pydantic config schema and validation
│   ├── data_loader.py             # SQuAD loader with local offline fallback
│   ├── sampling.py                # Reproducible sampling and audit manifests
│   ├── corpus.py                  # Context indexing and deduplication
│   ├── retrieval.py               # BM25Okapi lexical retrieval & Recall@k
│   ├── distractors.py             # Random and BM25 hard distractor generators
│   ├── reranking.py               # CrossEncoder & unranked ordering
│   ├── generation.py              # Local FLAN-T5, Groq, and Mock backends
│   ├── metrics.py                 # SQuAD Token F1 and Exact Match
│   ├── experiment.py              # Master experiment orchestrator
│   ├── statistics.py              # Paired bootstrap confidence intervals
│   ├── failures.py                # Automated failure taxonomy classification
│   ├── storage.py                 # Multi-asset file-based run persistence
│   ├── reporting.py               # Matplotlib figures and LaTeX tables
│   └── utils.py                   # Seeds, SHA-256 caching, and prompt builder
├── pages/                         # Streamlit multi-page interface
│   ├── 1_Dataset_Explorer.py      # SQuAD explorer & manifest export
│   ├── 2_Experiment_Configuration.py # Custom parameter builder & call estimator
│   ├── 3_Run_Experiments.py       # Live experiment runner and progress logs
│   ├── 4_Results.py               # F1/EM curves, summary tables, 95% CIs
│   ├── 5_Failure_Analysis.py      # Error taxonomy & researcher case annotations
│   ├── 6_Literature_Review.py     # 5-paper matrix and anchor feasibility
│   └── 7_Reproducibility.py       # Checklist audit & deliverable downloads
├── scripts/
│   ├── run_experiment.py          # CLI experiment runner
│   ├── generate_paper_assets.py   # IEEE LaTeX table and figure generator
│   └── verify_environment.py      # System and dependency check
├── docs/                          # Academic documentation & proposals
│   ├── research_proposal.md       # Full formal research proposal
│   ├── literature_review.md       # In-depth discussion of 5 papers
│   ├── paper_comparison.csv       # Verified bibliographic comparison table
│   ├── research_gap.md            # Gap analysis and contribution
│   ├── preregistration.md         # Study preregistration protocol
│   ├── reproducibility_checklist.md # NLP reproducibility verification
│   ├── experiment_protocol.md     # Step-by-step experiment instructions
│   ├── architecture.md            # Modular component specification
│   ├── limitations_and_threats.md # Validity threats and boundaries
│   ├── failure_analysis_guide.md  # Annotation guide and taxonomy
│   ├── presentation_script.md     # 10-minute conference presentation
│   └── demo_script.md             # 5-minute software demonstration
├── paper/                         # Academic IEEE paper deliverables
│   ├── main.tex                   # IEEE conference paper draft
│   ├── references.bib             # Verified BibTeX bibliography
│   ├── results_tables.tex         # LaTeX table populated from real runs
│   └── figures/                   # Scientific charts for publication
└── tests/                         # Comprehensive unit & integration tests
    ├── test_sampling.py
    ├── test_distractors.py
    ├── test_metrics.py
    ├── test_reranking.py
    ├── test_experiment.py
    ├── test_storage.py
    └── test_reporting.py
```

---

## 5. Installation & Verification

### Step 1: Clone and Set Up Virtual Environment
```bash
git clone https://github.com/your-username/RobustRAG.git
cd RobustRAG

python3 -m venv .venv
source .venv/bin/activate       # On Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Step 2: Verify System Dependencies & Hardware
```bash
python scripts/verify_environment.py
```
This script checks:
- Python version ($\ge 3.11$)
- Installed libraries (PyTorch, Transformers, Datasets, SentenceTransformers, RankBM25, etc.)
- Acceleration hardware (Apple Silicon MPS, CUDA, or CPU)
- Directory permissions and offline fallback dataset presence

### Step 3: Run the Verification Test Suite
```bash
pytest -v
ruff check .
```

---

## 6. Running Experiments

### Option A: Rapid Dry-Run Smoke Test (5 Seconds)
Tests complete orchestration using deterministic mock doubles:
```bash
python scripts/run_experiment.py --config configs/smoke_test.yaml --dry-run
```

### Option B: Pilot Experiment (2–4 Minutes)
Runs a 25-question evaluation using the local `google/flan-t5-small` model and `cross-encoder/ms-marco-MiniLM-L-6-v2`:
```bash
python scripts/run_experiment.py --config configs/pilot.yaml
```

### Option C: Full Main Experiment (100 Questions)
```bash
python scripts/run_experiment.py --config configs/default.yaml
```

### Optional: Ultra-Fast Groq Inference
If you prefer ultra-fast cloud inference instead of local inference, add your Groq API key to `.env`:
```bash
GROQ_API_KEY=your_key_here
```
And execute with:
```bash
python scripts/run_experiment.py --config configs/pilot.yaml --backend groq
```

---

## 7. Launching the Research Dashboard

Launch the interactive local web dashboard:
```bash
streamlit run app.py
```
Dashboard features:
- **Overview**: Real-time project status and architectural diagram.
- **Dataset Explorer**: Audit SQuAD 1.1 validation records and download sampling manifests.
- **Experiment Config**: Adjust distractor levels, sample size, and estimate generator calls before running.
- **Run Experiments**: Track live stage progress (dataset loading, generation, scoring).
- **Results**: Inspect F1/EM degradation charts and paired bootstrap confidence intervals (95% CI).
- **Failure Analysis**: Filter automated failure categories and save researcher case annotations.
- **Literature Review**: Search and examine the 5 verified anchor and related papers.
- **Reproducibility**: Export CSVs, JSON summaries, PNG plots, and IEEE LaTeX tables.

---

## 8. Compiling the IEEE Paper Draft

To generate the IEEE conference paper assets from completed experiment runs:
```bash
python scripts/generate_paper_assets.py
```
This updates `paper/results_tables.tex` with genuine empirical values and copies scientific figures into `paper/figures/`.

To compile the LaTeX paper:
```bash
cd paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

---

## 9. Citation Instructions

If you use RobustRAG in your research or coursework, please cite:
```bibtex
@misc{robustrag2026,
  author       = {RobustRAG Research Group},
  title        = {RobustRAG: How Irrelevant Retrieved Documents Affect Question Answering},
  year         = {2026},
  publisher    = {GitHub},
  howpublished = {\url{https://github.com/your-username/RobustRAG}}
}
```

---

## 10. License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
