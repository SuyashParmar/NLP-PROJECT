"""Streamlit Page 1: Dataset Explorer and Sampling Manifest Inspector."""

import json

import pandas as pd
import streamlit as st

from robustrag.corpus import DocumentCorpus
from robustrag.data_loader import DatasetLoader
from robustrag.sampling import QuestionSampler

st.set_page_config(page_title="Dataset Explorer - RobustRAG", page_icon="📚", layout="wide")

st.title("📚 Dataset Explorer & Sampling Manifest")
st.caption("Inspect and audit the SQuAD 1.1 validation dataset, corpus construction, and deterministic question sampling.")

# Caching dataset load so UI is fast and responsive
@st.cache_data(show_spinner=False)
def load_and_index_dataset(dataset_name: str, split: str):
    loader = DatasetLoader(dataset_name=dataset_name, split=split)
    examples = loader.load()
    corpus = DocumentCorpus(dataset_name=dataset_name)
    corpus.build_from_examples(examples)
    return examples, corpus


with st.sidebar:
    st.header("Dataset Parameters")
    dataset_name = st.selectbox("Dataset", ["squad"], index=0)
    split = st.selectbox("Split", ["validation"], index=0)
    sample_size = st.number_input("Sample Size (N)", min_value=5, max_value=500, value=25, step=5)
    random_seed = st.number_input("Random Seed", min_value=0, max_value=999999, value=42, step=1)

with st.spinner("Loading dataset and building corpus index (no model inference)..."):
    try:
        raw_examples, corpus = load_and_index_dataset(dataset_name, split)
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        st.stop()

# Perform deterministic sampling
sampler = QuestionSampler(sample_size=sample_size, random_seed=random_seed)
sampled_examples, manifest = sampler.sample(raw_examples, dataset_name=dataset_name, split=split)

# Top metrics
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Raw Questions", len(raw_examples))
with col2:
    st.metric("Unique Corpus Passages", len(corpus.passages))
with col3:
    st.metric("Merged Context Duplicates", corpus.raw_duplicate_count)
with col4:
    st.metric("Sampled Subset Size", len(sampled_examples))

st.markdown("---")

tab1, tab2, tab3 = st.tabs(["Sampled Questions Preview", "Full Corpus Viewer", "Audit Manifest & Export"])

with tab1:
    st.subheader(f"Sampled Question Manifest (N={len(sampled_examples)}, Seed={random_seed})")
    records = []
    for ex in sampled_examples:
        gold_p = corpus.get_gold_passage(ex.id)
        records.append({
            "Question ID": ex.id,
            "Article Title": ex.title,
            "Question": ex.question,
            "Reference Answers": " | ".join(ex.answers),
            "Gold Passage ID": gold_p.passage_id if gold_p else "None",
            "Gold Context Snippet": (ex.context[:120] + "...") if len(ex.context) > 120 else ex.context,
        })
    df_sampled = pd.DataFrame(records)
    st.dataframe(df_sampled, use_container_width=True)

    # Detailed single question inspector
    st.markdown("#### Detailed Example Inspector")
    selected_qid = st.selectbox("Select Question to inspect:", [ex.id for ex in sampled_examples])
    chosen_ex = next(ex for ex in sampled_examples if ex.id == selected_qid)

    col_q1, col_q2 = st.columns([1, 1])
    with col_q1:
        st.markdown(f"**Question ID:** `{chosen_ex.id}`")
        st.markdown(f"**Article Title:** *{chosen_ex.title}*")
        st.markdown(f"**Question Text:** {chosen_ex.question}")
        st.markdown(f"**Reference Annotations:** `{chosen_ex.answers}`")
    with col_q2:
        st.markdown("**Gold Passage Text:**")
        st.info(chosen_ex.context)

with tab2:
    st.subheader("Unique Indexed Document Corpus")
    st.caption("Passages extracted and deduplicated from SQuAD contexts, serving as the source pool for random and hard distractors.")
    df_corpus = corpus.to_dataframe()
    st.dataframe(df_corpus, use_container_width=True)

with tab3:
    st.subheader("Reproducibility Manifest Export")
    st.markdown(
        """
        The sampling manifest locks the exact question IDs and random seed. Every experimental condition
        in RobustRAG reuses this identical subset to eliminate sampling variance across conditions.
        """
    )
    manifest_dict = manifest.to_dict()
    st.json(manifest_dict)

    col_d1, col_d2 = st.columns(2)
    with col_d1:
        manifest_json_str = json.dumps(manifest_dict, indent=2)
        st.download_button(
            "Download Manifest (JSON)",
            data=manifest_json_str,
            file_name=f"sampling_manifest_N{sample_size}_seed{random_seed}.json",
            mime="application/json",
        )
    with col_d2:
        manifest_csv_str = manifest.to_dataframe().to_csv(index=False)
        st.download_button(
            "Download Question IDs (CSV)",
            data=manifest_csv_str,
            file_name=f"question_ids_N{sample_size}_seed{random_seed}.csv",
            mime="text/csv",
        )
