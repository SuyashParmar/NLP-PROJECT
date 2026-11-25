"""RobustRAG Research Project - Main Streamlit Dashboard Entry Point."""

import json
from pathlib import Path

import streamlit as st

# Custom academic UI styling
st.set_page_config(
    page_title="RobustRAG Research Dashboard",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    /* Academic styling palette */
    :root {
        --primary-blue: #0284c7;
        --dark-navy: #0f172a;
        --slate-gray: #334155;
        --card-bg: #f8fafc;
        --border-color: #e2e8f0;
    }
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }
    h1, h2, h3 {
        color: #0f172a;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 16px 20px;
        margin-bottom: 12px;
    }
    .metric-value {
        font-size: 24px;
        font-weight: 700;
        color: #0284c7;
    }
    .metric-label {
        font-size: 13px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748b;
        font-weight: 600;
    }
    .badge-completed {
        background-color: #ecfdf5;
        color: #065f46;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 600;
        border: 1px solid #a7f3d0;
    }
    .badge-pending {
        background-color: #fffbeb;
        color: #92400e;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 600;
        border: 1px solid #fde68a;
    }
    .callout-box {
        border-left: 4px solid #0284c7;
        background-color: #f0f9ff;
        padding: 14px 18px;
        border-radius: 0 6px 6px 0;
        margin: 16px 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def get_latest_run_info():
    runs_dir = Path("./outputs/runs")
    if not runs_dir.exists():
        return None, 0
    runs = []
    for p in runs_dir.iterdir():
        if p.is_dir() and (p / "metadata.json").exists():
            try:
                with open(p / "metadata.json", "r", encoding="utf-8") as f:
                    meta = json.load(f)
                    runs.append(meta)
            except Exception:
                pass
    runs.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    completed_empirical = [r for r in runs if not r.get("is_smoke_test") and r.get("status") == "completed"]
    latest = completed_empirical[0] if completed_empirical else (runs[0] if runs else None)
    return latest, len(runs)


def main():
    st.title("RobustRAG Research Dashboard")
    st.caption("How Irrelevant Retrieved Documents Affect Question Answering: An Empirical Investigation")

    latest_run, total_runs = get_latest_run_info()

    # Top Status Banner & Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Saved Experiments</div>
                <div class="metric-value">{total_runs}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        latest_label = latest_run["run_id"] if latest_run else "None yet"
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Latest Execution</div>
                <div class="metric-value" style="font-size: 15px; word-break: break-all;">{latest_label}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-label">Default Generator</div>
                <div class="metric-value" style="font-size: 16px;">FLAN-T5-Small / Groq</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-label">Target Dataset</div>
                <div class="metric-value" style="font-size: 16px;">SQuAD 1.1 (Validation)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Research Pillars & Methodology
    col_left, col_right = st.columns([1.6, 1.0])

    with col_left:
        st.subheader("Research Objective and Hypothesis")
        st.markdown(
            """
            **Central Research Question:**
            > *How does increasing the number of irrelevant documents in retrieved context affect question-answering quality, and to what extent can a cross-encoder reranking mechanism mitigate the resulting degradation?*

            **Scientific Hypothesis:**
            Increasing the quantity of irrelevant distractor passages will monotonically reduce question-answering accuracy (measured by token-level F1 and Exact Match), as distracting evidence competes with gold facts. A pretrained cross-encoder reranker can mitigate this degradation by ordering more relevant documents earlier in the context window.
            """
        )

        st.markdown(
            """
            <div class="callout-box">
                <strong>Methodological Distinction: Controlled Context-Noise vs. Natural Retrieval</strong><br>
                This investigation explicitly isolates <em>noise resilience</em>. By fixing the gold evidence passage and injecting controlled distractor passages (random or lexically similar hard distractors), we evaluate generator extraction robustness independently of retriever recall. A separate, unconfounded BM25 natural retrieval evaluation is provided to benchmark baseline retrieval performance.
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.subheader("Experimental Pipeline Architecture")
        st.code(
            """
[SQuAD 1.1 Validation Set]
          │
          ▼
[Reproducible Question Sampler (Seed=42)]
          │
          ├──> [Gold Evidence Passage] (1 Passage)
          │
          ├──> [Distractor Selection]
          │      ├── Random Corpus Sampling
          │      └── Lexical BM25 (Hard Distractors, Answer Excluded)
          │
          ▼
[Candidate Passages Set] (e.g., 1 Gold + k Distractors)
          │
    ┌─────┴─────────────────────────────────┐
    ▼                                       ▼
Condition C1: Unranked              Condition C2: Cross-Encoder
(Deterministic Permutation)         (ms-marco-MiniLM-L-6-v2)
    │                                       │
    └─────────────────┬─────────────────────┘
                      ▼
            [Answer Generator]
        (Local Flan-T5 / Groq LLM)
                      │
                      ▼
        [SQuAD Normalized Metrics]
    (Token F1, Exact Match, Bootstrap CI)
            """,
            language="text",
        )

    with col_right:
        st.subheader("Deliverables Status")
        status_items = [
            ("Literature Review (5 Verified Papers)", True),
            ("Experimental Preregistration", True),
            ("Controlled Pipeline Core", True),
            ("Streamlit Experiment Dashboard", True),
            ("Statistical Bootstrapping (95% CI)", True),
            ("IEEE Conference Paper Draft", True),
            ("Empirical Execution (Pilot/Main)", bool(latest_run and not latest_run.get("is_smoke_test"))),
        ]

        for item, is_done in status_items:
            badge = '<span class="badge-completed">Completed</span>' if is_done else '<span class="badge-pending">Pending Run</span>'
            st.markdown(f"- {item}: {badge}", unsafe_allow_html=True)

        st.markdown("---")
        st.subheader("Quick Navigation")
        st.markdown(
            """
            - **1. Dataset Explorer**: Preview and audit SQuAD questions and corpus passages.
            - **2. Experiment Config**: Set sample size, noise levels, models, and random seeds.
            - **3. Run Experiments**: Execute the live experiment with progress monitoring.
            - **4. Results**: Inspect degradation curves, F1/EM tables, and 95% CIs.
            - **5. Failure Analysis**: Categorize and annotate individual error cases.
            - **6. Literature Review**: Study the 5 verified anchor and related papers.
            - **7. Reproducibility**: Export datasets, figures, and compiled LaTeX tables.
            """
        )

    if not latest_run:
        st.info("No experimental results yet. Navigate to **Run Experiments** or execute a pilot run via CLI to populate real data.")


if __name__ == "__main__":
    main()
