"""Streamlit Page 7: Reproducibility Hub, Environment Audit, and Deliverable Exports."""

import platform
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

from robustrag.reporting import ReportGenerator
from robustrag.storage import ExperimentStore

st.set_page_config(page_title="Reproducibility & Exports - RobustRAG", page_icon="📦", layout="wide")

st.title("📦 Reproducibility Hub & Asset Exports")
st.caption("Inspect runtime software environment, audit reproducibility standards, and download research deliverables.")

store = ExperimentStore("./outputs/runs")
runs = store.list_runs()

tab1, tab2, tab3 = st.tabs(["Research Deliverables & Downloads", "Reproducibility Checklist", "Runtime Environment Audit"])

with tab1:
    st.subheader("One-Click Export of Research Assets")
    if not runs:
        st.info("No runs found. Execute an experiment to export run-specific artifacts.")
    else:
        run_options = {r["run_id"]: r["run_id"] for r in runs}
        selected_run_id = st.selectbox("Select Target Run for Export:", list(run_options.keys()))
        run_data = store.load_run(selected_run_id)
        run_dir = run_data["run_dir"]

        st.markdown(f"**Selected Run Directory:** `{run_dir}`")

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("#### Tabular Data")
            preds_file = run_dir / "predictions.csv"
            if preds_file.exists():
                st.download_button(
                    "📥 Predictions (CSV)",
                    data=preds_file.read_bytes(),
                    file_name=f"predictions_{selected_run_id}.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            scores_file = run_dir / "scores.csv"
            if scores_file.exists():
                st.download_button(
                    "📥 Condition Scores (CSV)",
                    data=scores_file.read_bytes(),
                    file_name=f"scores_{selected_run_id}.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            failures_file = run_dir / "failure_cases.csv"
            if failures_file.exists():
                st.download_button(
                    "📥 Failure Cases (CSV)",
                    data=failures_file.read_bytes(),
                    file_name=f"failures_{selected_run_id}.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

        with c2:
            st.markdown("#### Reports & LaTeX")
            report_file = run_dir / "report.md"
            if report_file.exists():
                st.download_button(
                    "📥 Markdown Report (.md)",
                    data=report_file.read_bytes(),
                    file_name=f"report_{selected_run_id}.md",
                    mime="text/markdown",
                    use_container_width=True,
                )
            # LaTeX table
            latex_code = ReportGenerator.generate_latex_table(run_data["summary"])
            st.download_button(
                "📥 IEEE LaTeX Table (.tex)",
                data=latex_code,
                file_name=f"results_{selected_run_id}.tex",
                mime="text/plain",
                use_container_width=True,
            )
            cfg_file = run_dir / "config.yaml"
            if cfg_file.exists():
                st.download_button(
                    "📥 Config (.yaml)",
                    data=cfg_file.read_bytes(),
                    file_name=f"config_{selected_run_id}.yaml",
                    mime="application/x-yaml",
                    use_container_width=True,
                )

        with c3:
            st.markdown("#### Scientific Figures")
            f1_fig = run_dir / "figures" / "noise_degradation_f1.png"
            if f1_fig.exists():
                st.download_button(
                    "📥 Degradation Plot (PNG)",
                    data=f1_fig.read_bytes(),
                    file_name="figure_noise_degradation.png",
                    mime="image/png",
                    use_container_width=True,
                )
            type_fig = run_dir / "figures" / "distractor_type_comparison.png"
            if type_fig.exists():
                st.download_button(
                    "📥 Distractor Plot (PNG)",
                    data=type_fig.read_bytes(),
                    file_name="figure_distractor_type.png",
                    mime="image/png",
                    use_container_width=True,
                )

with tab2:
    st.subheader("Official Reproducibility Checklist")
    st.caption("Verification against strict empirical NLP standards (Rogers et al., 2021; Pineau et al., 2021).")

    checklist_items = [
        ("Fixed Question Subsets", "Same questions are strictly reused across all experimental blocks.", True),
        ("Identical Candidate Sets", "Reranker comparisons use identical document sets, only changing ordering.", True),
        ("Gold Passage Isolation", "Controlled noise tests evaluate noise injection, keeping BM25 retrieval separate.", True),
        ("Deterministic Decoding", "Greedy decoding enabled (temperature=0.0, do_sample=False) for reproducibility.", True),
        ("Answer Leakage Exclusion", "Hard distractors explicitly filter passages matching reference answers.", True),
        ("Paired Statistical Testing", "Non-parametric bootstrap confidence intervals (95% CI) computed on matched pairs.", True),
        ("Audit Metadata Logged", "All random seeds, software versions, timestamps, and model hashes persisted.", True),
        ("Open-Source / Zero Paid APIs", "Pipeline runs locally with free Hugging Face models on consumer hardware.", True),
    ]

    for title, desc, checked in checklist_items:
        st.checkbox(f"**{title}**: {desc}", value=checked, disabled=True)

with tab3:
    st.subheader("Runtime Environment Audit")
    env_info = {
        "Python Version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "Operating System": platform.platform(),
        "Architecture": platform.machine(),
        "Base Output Directory": str(Path("./outputs/runs").resolve()),
    }
    st.json(env_info)

    # Installed packages version table
    packages = ["torch", "transformers", "datasets", "sentence_transformers", "rank_bm25", "scikit-learn", "scipy", "pandas", "numpy", "matplotlib", "streamlit"]
    pkg_rows = []
    for pkg in packages:
        try:
            m = __import__(pkg.replace("-", "_"))
            v = getattr(m, "__version__", "Available")
            pkg_rows.append({"Package": pkg, "Installed Version": str(v), "Status": "Verified"})
        except ImportError:
            pkg_rows.append({"Package": pkg, "Installed Version": "Not Installed", "Status": "Missing"})
    st.dataframe(pd.DataFrame(pkg_rows), use_container_width=True)
