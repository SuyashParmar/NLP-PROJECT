"""Streamlit Page 6: Literature Review & 5-Paper Comparative Matrix."""

from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Literature Review - RobustRAG", page_icon="📖", layout="wide")

st.title("📖 Literature Review & Anchor Paper Matrix")
st.caption("Systematic evaluation of 5 peer-reviewed RAG robustness papers from ACL, EMNLP, and ICML proceedings.")

csv_path = Path("docs/paper_comparison.csv")
if csv_path.exists():
    df_papers = pd.read_csv(csv_path)
else:
    st.error("Paper comparison database not found at docs/paper_comparison.csv")
    st.stop()

st.markdown(
    """
    <div style="background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 16px; margin-bottom: 20px;">
        <strong>Academic Grounding & Candidate Anchor Paper:</strong><br>
        The candidate anchor paper is <em>“Quantifying and Improving the Robustness of Retrieval-Augmented Language Models Against Spurious Features in Grounding Data”</em> (ACL 2026, SURE framework).
        RobustRAG adapts its research focus into an accessible, reproducible undergraduate investigation: quantifying how distracting contexts degrade Question Answering on SQuAD 1.1 and testing cross-encoder reranking mitigation on CPU/consumer hardware without paid APIs.
    </div>
    """,
    unsafe_allow_html=True,
)

tab1, tab2, tab3 = st.tabs(["5-Paper Comparison Matrix", "Individual Paper Deep-Dives", "Anchor Paper Feasibility Report"])

with tab1:
    st.subheader("Comparative Overview of Candidate Anchor & Related Papers")

    # Filter by venue
    venues = ["All"] + list(df_papers["venue_year"].unique())
    chosen_venue = st.selectbox("Filter by Venue / Year:", venues)
    display_df = df_papers if chosen_venue == "All" else df_papers[df_papers["venue_year"] == chosen_venue]

    st.dataframe(
        display_df[["title", "venue_year", "research_problem", "proposed_method", "models_evaluated", "metrics", "robustrag_extension"]],
        use_container_width=True,
    )

    st.download_button(
        "Download Paper Matrix (CSV)",
        data=df_papers.to_csv(index=False),
        file_name="robustrag_literature_comparison.csv",
        mime="text/csv",
    )

with tab2:
    st.subheader("Detailed Verified Bibliographic Details")
    paper_titles = df_papers["title"].tolist()
    selected_title = st.selectbox("Select Paper to Review:", paper_titles)
    p_data = df_papers[df_papers["title"] == selected_title].iloc[0]

    col_p1, col_p2 = st.columns([1.2, 1.0])
    with col_p1:
        st.markdown(f"### {p_data['title']}")
        st.markdown(f"**Venue & Year:** `{p_data['venue_year']}` | **Authors:** *{p_data['authors']}*")
        st.markdown(f"**Official Proceedings Page:** [{p_data['source_link']}]({p_data['source_link']})")
        st.markdown(f"**Research Problem:** {p_data['research_problem']}")
        st.markdown(f"**Proposed Method:** {p_data['proposed_method']}")
        st.markdown(f"**Evaluated Models:** `{p_data['models_evaluated']}`")
        st.markdown(f"**Metrics:** `{p_data['metrics']}`")

    with col_p2:
        st.markdown("#### Empirical Findings & Constraints")
        st.info(f"**Main Findings:** {p_data['main_findings']}")
        st.warning(f"**Limitations:** {p_data['limitations']}")
        st.markdown(f"**Code / Data Availability:** `{p_data['code_availability']}`")
        st.markdown(f"**Compute Requirements:** `{p_data['compute_notes']}`")
        st.markdown(f"**Connection to RobustRAG:** {p_data['robustrag_extension']}")

with tab3:
    st.subheader("Anchor Paper Reproducibility & Feasibility Assessment")
    st.markdown(
        """
        ### Why the SURE Paper (ACL 2026) was Selected as the Candidate Anchor
        1. **Scientific Relevance**: The SURE paper formally analyzes how spurious lexical correlation and grounding distractors trick reader models into confident but unsupported extractions.
        2. **Methodological Extensibility**: While the original paper benchmarks 7B–70B parameter models requiring cluster compute, its core controlled context-perturbation methodology can be rigorously adapted to compact local seq2seq models (`FLAN-T5-small`) and open cross-encoder rerankers (`ms-marco-MiniLM-L-6-v2`).
        3. **Clear Independent Contribution**:
           - RobustRAG benchmarks the exact F1 degradation slope against discrete distractor steps (k=0, 1, 3, 5).
           - RobustRAG isolates the effect of lexical hard distractors vs random distractors.
           - RobustRAG verifies whether cross-encoder reordering alone restores performance on identical candidate sets.
        4. **Compute Feasibility**: RobustRAG operates entirely on consumer hardware (CPU or Apple Silicon MPS) in under 5 minutes for pilot runs, without requiring external API credits.
        """
    )
