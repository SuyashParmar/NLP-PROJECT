"""Streamlit Page 5: Failure Analysis and Researcher Case Annotation."""

import streamlit as st

from robustrag.storage import ExperimentStore

st.set_page_config(page_title="Failure Analysis - RobustRAG", page_icon="🔍", layout="wide")

st.title("🔍 Failure Analysis & Case Annotation")
st.caption("Deep-dive into systematic extraction errors, distractor seduction, reranker recovery, and manual case studies.")

store = ExperimentStore("./outputs/runs")
runs = store.list_runs()

if not runs:
    st.info("No experiment runs found. Please run an experiment first.")
    st.stop()

run_options = {
    f"{r['run_id']} ({r.get('experiment_name', 'run')})": r["run_id"]
    for r in runs
}
selected_label = st.sidebar.selectbox("Select Experiment Run:", list(run_options.keys()), index=0)
selected_run_id = run_options[selected_label]

run_data = store.load_run(selected_run_id)
df_failures = run_data["failures"]

if df_failures.empty:
    st.info(f"No failure or transition cases identified in run `{selected_run_id}`.")
    st.stop()

# Category Breakdown Metrics
categories = df_failures["failure_category"].value_counts()
st.subheader("Automated Failure Category Distribution")
cols = st.columns(len(categories))
for idx, (cat_name, count) in enumerate(categories.items()):
    with cols[idx]:
        st.metric(cat_name, count)

st.markdown("---")

tab1, tab2 = st.tabs(["Interactive Failure Table & Annotations", "Single Case Study Inspector"])

with tab1:
    st.subheader("Extracted Candidate Failure Cases")
    st.caption("Edit the 'researcher_notes' column directly in the table, then click 'Save Annotations' to persist.")

    # Data editor for interactive notes
    edited_df = st.data_editor(
        df_failures[["question_id", "question", "predicted_answer", "reference_answers", "f1", "failure_category", "researcher_notes"]],
        use_container_width=True,
        num_rows="fixed",
        key="failures_editor",
    )

    col_s1, col_s2 = st.columns([1, 1])
    with col_s1:
        if st.button("💾 Save Researcher Annotations", type="primary"):
            df_failures["researcher_notes"] = edited_df["researcher_notes"]
            store.update_failure_annotations(selected_run_id, df_failures)
            st.success("Annotations successfully updated and saved to disk!")
    with col_s2:
        st.download_button(
            "Download Failure Cases (CSV)",
            data=df_failures.to_csv(index=False),
            file_name=f"failure_cases_{selected_run_id}.csv",
            mime="text/csv",
        )

with tab2:
    st.subheader("Deep Example Inspection")
    q_options = df_failures["question_id"].unique()
    chosen_qid = st.selectbox("Select Failure Case to Inspect:", q_options)

    chosen_row = df_failures[df_failures["question_id"] == chosen_qid].iloc[0]

    col_c1, col_c2 = st.columns([1, 1])
    with col_c1:
        st.markdown(f"**Question ID:** `{chosen_row['question_id']}`")
        st.markdown(f"**Question Text:** {chosen_row['question']}")
        st.markdown(f"**Failure Category:** `{chosen_row['failure_category']}`")
        st.markdown(f"**Evaluated Condition:** `{chosen_row['condition_id']}`")
        st.markdown(f"**F1 Score:** `{chosen_row['f1']}` | **Exact Match:** `{chosen_row['exact_match']}`")

    with col_c2:
        st.markdown(f"**Ground Truth Reference:** `{chosen_row['reference_answers']}`")
        st.markdown(f"**Model Predicted Answer:** `{chosen_row['predicted_answer']}`")
        st.markdown(f"**Active Researcher Notes:** *{chosen_row['researcher_notes']}*")

    st.markdown("---")
    st.markdown("#### Candidate Passages Present During Generation")
    st.code(f"Passage IDs in Context Window: {chosen_row['passage_ids']}", language="text")
