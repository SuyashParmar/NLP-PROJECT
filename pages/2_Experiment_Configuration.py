"""Streamlit Page 2: Experiment Configuration and Call Estimator."""

from pathlib import Path

import streamlit as st
import yaml

from robustrag.config import (
    AppConfig,
    DatasetConfig,
    EvaluationConfig,
    ExperimentFlags,
    GeneratorConfig,
    NoiseConfig,
    RerankerConfig,
    StorageConfig,
)

st.set_page_config(page_title="Experiment Config - RobustRAG", page_icon="⚙️", layout="wide")

st.title("⚙️ Experiment Configuration")
st.caption("Customize experimental blocks, models, noise quantities, and statistical settings.")

# Load default config template
default_cfg = AppConfig()

with st.form("experiment_config_form"):
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("1. General & Dataset Parameters")
        exp_name = st.text_input("Experiment Run Name", value="robustrag_controlled_evaluation")
        random_seed = st.number_input("Global Random Seed", min_value=0, max_value=999999, value=42)
        sample_size = st.number_input("Question Sample Size (N)", min_value=5, max_value=500, value=25, step=5)
        is_smoke = st.checkbox("Mark as Smoke / Mock Test (Excludes from empirical tables)", value=False)

        st.subheader("2. Model Selection")
        gen_backend = st.selectbox(
            "Generator Backend",
            ["local", "groq", "mock"],
            index=0,
            help="'local' runs HuggingFace FLAN-T5; 'groq' uses high-speed Groq API; 'mock' uses fast dry-run double."
        )
        gen_model = st.text_input("Local Model Name", value="google/flan-t5-small")
        groq_model = st.text_input("Groq Cloud Model", value="qwen/qwen3.8-27b")
        max_tokens = st.number_input("Max Generation Tokens", min_value=8, max_value=128, value=32)

        rerank_backend = st.selectbox(
            "Reranker Backend",
            ["local", "mock"],
            index=0 if gen_backend != "mock" else 1,
            help="'local' loads SentenceTransformers CrossEncoder; 'mock' uses lexical overlap."
        )
        rerank_model = st.text_input("Reranker Model Name", value="cross-encoder/ms-marco-MiniLM-L-6-v2")

    with col2:
        st.subheader("3. Noise & Distractor Controls")
        quantities_str = st.text_input("Injected Distractor Quantities (comma-separated)", value="0, 1, 3, 5")
        fixed_distractor_count = st.number_input("Fixed Distractor Count for Type Comparison (Exp B)", min_value=1, max_value=10, value=3)
        hard_pool_size = st.number_input("BM25 Candidate Pool for Hard Distractors", min_value=5, max_value=30, value=10)

        st.subheader("4. Experimental Blocks & Interventions")
        run_exp_a = st.checkbox("Experiment A: Noise Quantity Curve (0, 1, 3, 5 random)", value=True)
        run_exp_b = st.checkbox("Experiment B: Distractor Type (Random vs Hard at fixed k)", value=True)
        run_exp_c = st.checkbox("Experiment C: Cross-Encoder Reranker Intervention", value=True)
        run_exp_d = st.checkbox("Experiment D: Top-k Context Filtering Ablation", value=True)
        top_k_val = st.number_input("Top-k Documents to Retain in Exp D", min_value=1, max_value=5, value=3)
        run_retrieval_real = st.checkbox("Separate Natural BM25 Retrieval Evaluation (Recall@k)", value=True)

        st.subheader("5. Statistical Analysis")
        bootstrap_iters = st.number_input("Bootstrap Iterations (Paired 95% CI)", min_value=100, max_value=5000, value=500, step=100)

    submit_button = st.form_submit_button("Save & Validate Configuration")

# Parse quantities
try:
    noise_quantities = [int(x.strip()) for x in quantities_str.split(",") if x.strip()]
except ValueError:
    st.error("Invalid format for distractor quantities. Use comma-separated integers, e.g. '0, 1, 3, 5'")
    noise_quantities = [0, 1, 3, 5]

# Build AppConfig object
configured_app = AppConfig(
    name=exp_name,
    random_seed=random_seed,
    is_smoke_test=is_smoke,
    dataset=DatasetConfig(sample_size=sample_size),
    generator=GeneratorConfig(
        backend=gen_backend,
        model_name=gen_model,
        groq_model=groq_model,
        max_new_tokens=max_tokens,
    ),
    reranker=RerankerConfig(
        backend=rerank_backend,
        model_name=rerank_model,
    ),
    noise=NoiseConfig(
        quantities=noise_quantities,
        fixed_distractor_count=fixed_distractor_count,
        hard_distractor_pool_size=hard_pool_size,
    ),
    experiments=ExperimentFlags(
        run_experiment_a=run_exp_a,
        run_experiment_b=run_exp_b,
        run_experiment_c=run_exp_c,
        run_experiment_d=run_exp_d,
        top_k_filter=top_k_val,
        run_retrieval_realistic=run_retrieval_real,
    ),
    evaluation=EvaluationConfig(
        bootstrap_iterations=bootstrap_iters,
    ),
    storage=StorageConfig(output_dir="./outputs/runs"),
)

val_errors = configured_app.validate()

st.markdown("---")
st.subheader("Configuration Analysis & Pre-Run Audit")

col_est1, col_est2 = st.columns(2)
with col_est1:
    total_calls = configured_app.estimate_generator_calls()
    st.markdown(
        f"""
        <div style="background-color: #f0fdf4; border: 1px solid #bbf7d0; padding: 16px; border-radius: 8px;">
            <div style="font-size: 13px; font-weight: 600; color: #166534; text-transform: uppercase;">Estimated Generator Invocations</div>
            <div style="font-size: 28px; font-weight: 700; color: #15803d;">{total_calls:,} calls</div>
            <div style="font-size: 13px; color: #374151; margin-top: 4px;">
                Calculated strictly from <strong>{sample_size}</strong> questions across active experimental conditions.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_est2:
    if val_errors:
        for err in val_errors:
            st.error(f"Validation Error: {err}")
    else:
        st.success("All configuration parameters passed validation checks!")

if submit_button and not val_errors:
    save_path = Path("configs/custom_active.yaml")
    configured_app.save_yaml(save_path)
    st.success(f"Configuration saved successfully to `{save_path}`! You can now proceed to **Run Experiments**.")

# Show YAML preview
with st.expander("Inspect Configuration YAML"):
    st.code(yaml.dump(configured_app.to_dict(), default_flow_style=False, sort_keys=False), language="yaml")
