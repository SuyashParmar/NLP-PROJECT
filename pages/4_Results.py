"""Streamlit Page 4: Results Dashboard, Charts, and Confidence Intervals."""

import pandas as pd
import streamlit as st

from robustrag.reporting import ReportGenerator
from robustrag.storage import ExperimentStore

st.set_page_config(page_title="Results Dashboard - RobustRAG", page_icon="📊", layout="wide")

st.title("📊 Empirical Results & Statistical Analysis")
st.caption("Quantitative evaluations, noise degradation trajectories, reranking mitigation effects, and paired bootstrap CIs.")

store = ExperimentStore("./outputs/runs")
runs = store.list_runs()

if not runs:
    st.info("No saved experiment runs found. Please run an experiment from **Run Experiments** or via CLI first.")
    st.stop()

# Run Selector in Sidebar
run_options = {
    f"{r['run_id']} ({r.get('experiment_name', 'run')}) [{'SMOKE' if r.get('is_smoke_test') else 'EMPIRICAL'}]": r["run_id"]
    for r in runs
}
selected_label = st.sidebar.selectbox("Select Experiment Run:", list(run_options.keys()), index=0)
selected_run_id = run_options[selected_label]

try:
    run_data = store.load_run(selected_run_id)
except Exception as e:
    st.error(f"Error loading run: {e}")
    st.stop()

meta = run_data["metadata"]
df_summary = run_data["summary"]
df_predictions = run_data["predictions"]
summary_dict = run_data["summary_dict"]

# Banner if smoke test
if meta.get("is_smoke_test"):
    st.warning("⚠️ **Notice: This run was flagged as a SMOKE / MOCK test.** Results are synthetic for pipeline validation.")

# Top Metadata Summary Cards
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Questions Evaluated", meta.get("sample_size", "N/A"))
with col2:
    st.metric("Random Seed", meta.get("random_seed", "N/A"))
with col3:
    st.metric("Generator Backend", f"{meta.get('generator_backend', 'N/A')}")
with col4:
    st.metric("Reranker Model", "Cross-Encoder (MiniLM)")

st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs(["Condition Summary Table", "Performance Charts", "Paired Bootstrap CIs", "Question-Level Predictions"])

with tab1:
    st.subheader("Aggregated Performance by Experimental Condition")
    st.dataframe(df_summary, use_container_width=True)

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        st.download_button(
            "Download Summary Scores (CSV)",
            data=df_summary.to_csv(index=False),
            file_name=f"scores_{selected_run_id}.csv",
            mime="text/csv",
        )
    with col_btn2:
        st.download_button(
            "Download Summary JSON",
            data=pd.Series(summary_dict).to_json(indent=2),
            file_name=f"summary_{selected_run_id}.json",
            mime="application/json",
        )

with tab2:
    st.subheader("Scientific Degradation & Comparison Curves")
    figures_dir = run_data["run_dir"] / "figures"

    # Refresh or generate figures if not existing
    if not (figures_dir / "noise_degradation_f1.png").exists():
        ReportGenerator.generate_figures(df_summary, figures_dir)

    f1_fig_path = figures_dir / "noise_degradation_f1.png"
    type_fig_path = figures_dir / "distractor_type_comparison.png"

    col_fig1, col_fig2 = st.columns([1.2, 1.0])
    with col_fig1:
        if f1_fig_path.exists():
            st.image(str(f1_fig_path), caption="Experiment A & C: Answer F1 vs Distractor Count (No-Rerank vs Cross-Encoder)", use_column_width=True)
        else:
            st.info("F1 vs Noise plot not available for this run.")

    with col_fig2:
        if type_fig_path.exists():
            st.image(str(type_fig_path), caption="Experiment B: Distractor Type (Random vs Hard at fixed k)", use_column_width=True)
        else:
            st.info("Distractor type comparison plot not available.")

with tab3:
    st.subheader("Paired Bootstrap Confidence Intervals (95% CI)")
    st.markdown(
        """
        Paired bootstrap resampling assesses whether performance differences across identical questions
        are statistically discernible from zero without relying on normality assumptions.
        """
    )
    ci_data = summary_dict.get("bootstrap_ci", {})
    if ci_data:
        ci_records = []
        for pair_name, d in ci_data.items():
            ci_records.append({
                "Intervention / Comparison": pair_name.replace("_", " ").title(),
                "Mean Difference (Δ)": f"{d['mean_diff']:+.4f}",
                "95% CI Lower": f"{d['ci_lower']:+.4f}",
                "95% CI Upper": f"{d['ci_upper']:+.4f}",
                "Iterations": d["iterations"],
                "Statistical Significance (p < 0.05)": "✅ Yes (Zero Excluded)" if d["is_significant"] else "❌ Inconclusive (Includes Zero)",
            })
        st.table(pd.DataFrame(ci_records))
    else:
        st.info("No paired bootstrap confidence intervals recorded for this run.")

    # Retrieval Realistic Diagnostics
    diag = summary_dict.get("retrieval_diagnostics", {})
    if diag:
        st.markdown("---")
        st.subheader("Natural BM25 Retrieval Diagnostics (Separate Evaluation)")
        col_diag1, col_diag2, col_diag3, col_diag4 = st.columns(4)
        with col_diag1:
            st.metric("Recall@1", f"{diag.get('recall_at_1', 0.0)*100:.1f}%")
        with col_diag2:
            st.metric("Recall@3", f"{diag.get('recall_at_3', 0.0)*100:.1f}%")
        with col_diag3:
            st.metric("Recall@5", f"{diag.get('recall_at_5', 0.0)*100:.1f}%")
        with col_diag4:
            st.metric("Evaluated Queries", diag.get("total_evaluated_queries", 0))

with tab4:
    st.subheader("Question-Level Predictions & Diagnostics")

    # Interactive filtering
    col_flt1, col_flt2, col_flt3 = st.columns(3)
    with col_flt1:
        cond_filter = st.multiselect("Filter by Condition:", df_predictions["condition_id"].unique(), default=df_predictions["condition_id"].unique())
    with col_flt2:
        min_f1 = st.slider("Minimum F1 Score:", min_value=0.0, max_value=1.0, value=0.0, step=0.1)
    with col_flt3:
        search_query = st.text_input("Search Question or Answer Text:")

    filtered_df = df_predictions[
        (df_predictions["condition_id"].isin(cond_filter)) &
        (df_predictions["f1"] >= min_f1)
    ]
    if search_query:
        filtered_df = filtered_df[
            filtered_df["question"].str.contains(search_query, case=False, na=False) |
            filtered_df["predicted_answer"].str.contains(search_query, case=False, na=False)
        ]

    st.dataframe(
        filtered_df[["question_id", "condition_id", "question", "predicted_answer", "reference_answers", "f1", "exact_match"]],
        use_container_width=True,
    )

    st.download_button(
        "Download Filtered Predictions (CSV)",
        data=filtered_df.to_csv(index=False),
        file_name=f"predictions_{selected_run_id}.csv",
        mime="text/csv",
    )
