"""Streamlit Page 3: Live Experiment Execution and Progress Monitoring."""

from pathlib import Path

import streamlit as st

from robustrag.config import AppConfig
from robustrag.experiment import ExperimentRunner

st.set_page_config(page_title="Run Experiments - RobustRAG", page_icon="🚀", layout="wide")

st.title("🚀 Run Empirical Experiments")
st.caption("Execute controlled RAG context-noise trials with real-time stage tracking and audit logging.")

# Choose configuration file to run
config_options = {
    "Pilot Run (25 Questions, Fast)": "configs/pilot.yaml",
    "Full Main Experiment (100 Questions)": "configs/default.yaml",
    "Rapid Smoke Test (5 Questions, Mock)": "configs/smoke_test.yaml",
}
custom_cfg_file = Path("configs/custom_active.yaml")
if custom_cfg_file.exists():
    config_options["Custom Saved Configuration"] = str(custom_cfg_file)

selected_label = st.selectbox("Select Experiment Configuration:", list(config_options.keys()), index=0)
selected_path = config_options[selected_label]

col_cfg1, col_cfg2 = st.columns([1, 1])
with col_cfg1:
    try:
        cfg = AppConfig.from_yaml(selected_path)
        st.markdown(f"**Experiment Name:** `{cfg.name}`")
        st.markdown(f"**Sample Size:** `{cfg.dataset.sample_size}` questions (Seed: `{cfg.random_seed}`)")
        st.markdown(f"**Generator Backend:** `{cfg.generator.backend}` (`{cfg.generator.model_name}`)")
        st.markdown(f"**Reranker:** `{cfg.reranker.model_name}`")
    except Exception as e:
        st.error(f"Error loading configuration: {e}")
        st.stop()

with col_cfg2:
    est_calls = cfg.estimate_generator_calls()
    is_mock = (cfg.generator.backend == "mock")
    st.markdown(f"**Total Generator Invocations:** ~`{est_calls}` calls")
    st.markdown(f"**Active Distractor Levels:** `{cfg.noise.quantities}`")
    if is_mock or cfg.is_smoke_test:
        st.warning("Running in MOCK / SMOKE mode. Results will be marked as synthetic.")
    else:
        st.info("Running in REAL MODEL mode. Generations will be evaluated empirically.")

st.markdown("---")

# Execution Control
start_run = st.button("▶️ Launch Experiment Pipeline", type="primary", use_container_width=True)

if start_run:
    progress_bar = st.progress(0.0)
    status_text = st.empty()
    log_area = st.empty()

    logs = []

    def progress_callback(pct: float, msg: str):
        progress_bar.progress(min(1.0, max(0.0, pct)))
        status_text.markdown(f"**Status:** {msg}")
        logs.append(msg)
        log_area.text("\n".join(logs[-8:]))

    try:
        runner = ExperimentRunner(cfg)
        results = runner.run(progress_callback=progress_callback)

        progress_bar.progress(1.0)
        status_text.success(f"🎉 Experiment Completed Successfully! Run ID: `{results['run_id']}`")

        st.markdown("### Execution Summary")
        st.dataframe(results["summary"], use_container_width=True)

        st.success(
            f"All assets persisted to `{results['run_dir']}`. "
            f"Navigate to **Results** or **Failure Analysis** to inspect."
        )

    except Exception as e:
        status_text.error(f"Experiment execution encountered an error: {e}")
        st.exception(e)
        st.markdown("---")
        st.subheader("Troubleshooting Tips:")
        st.markdown(
            """
            1. **Hugging Face Model Downloads**: If downloading `google/flan-t5-small` or `cross-encoder` fails due to network, switch backend to `groq` (instant cloud API) or `mock` (rapid test double).
            2. **Groq API Key**: If using Groq, verify that `GROQ_API_KEY` is provided in `.env`.
            3. **Out of Memory**: On limited RAM, reduce `sample_size` to 10-20 questions or lower `batch_size` in config.
            """
        )
