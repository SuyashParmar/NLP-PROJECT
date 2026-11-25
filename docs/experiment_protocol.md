# RobustRAG Experimental Protocol

This document provides step-by-step operating procedures for executing and validating RobustRAG experiments.

---

## 1. Prerequisites and Environment Verification
Before executing experiments, ensure the environment is verified:
```bash
python scripts/verify_environment.py
```
Expected output: `ALL CHECKS PASSED: Environment is ready for RobustRAG research!`.

---

## 2. Experimental Execution Modes

### Mode A: Rapid Smoke Test (Validation)
Used for pipeline integrity checks without downloading large model weights:
```bash
python scripts/run_experiment.py --config configs/smoke_test.yaml --dry-run
```
- Sample Size: 5 questions
- Evaluates Blocks: A, B, C, D
- Expected runtime: < 5 seconds
- Output tagged as: `is_smoke_test: true`

### Mode B: Pilot Experiment (Recommended First Run)
Evaluates a lightweight 25-question subset to observe degradation slopes and reranker effects:
```bash
python scripts/run_experiment.py --config configs/pilot.yaml
```
- Sample Size: 25 questions
- Default generator: `google/flan-t5-small` (or Groq `llama-3.1-8b-instant` if configured)
- Expected runtime: ~2–4 minutes on CPU/MPS
- Output saved in: `outputs/runs/run_YYYYMMDD_HHMMSS_robustrag_pilot_run/`

### Mode C: Full Empirical Experiment
Full evaluation across 100 questions for academic publication:
```bash
python scripts/run_experiment.py --config configs/default.yaml
```
- Sample Size: 100 questions
- Bootstrap Iterations: 1,000 resamples
- Expected runtime: ~10–15 minutes on local hardware

---

## 3. Post-Experiment Verification & Deliverable Generation

1. **Inspect Results via Streamlit Dashboard**:
   ```bash
   streamlit run app.py
   ```
   Navigate to **Results** and **Failure Analysis**.

2. **Generate IEEE LaTeX Assets**:
   ```bash
   python scripts/generate_paper_assets.py
   ```
   Refreshes `paper/results_tables.tex` and syncs PNG figures to `paper/figures/`.

3. **Annotate Failure Cases**:
   Open **Failure Analysis** in the dashboard, review 5–10 representative failure cases, add `researcher_notes`, and click **Save Researcher Annotations**.
