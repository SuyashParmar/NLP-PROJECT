# RobustRAG Software Demonstration Script (5 Minutes)

**Purpose**: Guide a live software walkthrough for faculty reviewers, academic advisors, or thesis committees.

---

### Phase 1: Environment & Architecture Overview (1 Minute)
1. **Terminal Startup**:
   - Run the environment verification script:
     ```bash
     python scripts/verify_environment.py
     ```
   - Highlight: Python 3.11, PyTorch with Apple Silicon MPS / CPU acceleration, and local Hugging Face model backends.
2. **Launch Streamlit Dashboard**:
   ```bash
   streamlit run app.py
   ```
   - Point out the restrained academic aesthetic: deep navy typography, clean metric cards, and clear separation between controlled context-noise testing and natural retrieval evaluation.

---

### Phase 2: Dataset Explorer & Audit Manifest (1 Minute)
1. In the sidebar, navigate to **1. Dataset Explorer**.
2. Show the SQuAD 1.1 validation split preview.
3. Adjust the sample size (e.g., $N=25$) and seed ($S=42$).
4. Highlight that clicking through datasets does not execute model inference.
5. Click **Download Manifest (JSON)** to demonstrate reproducible question ID locking.

---

### Phase 3: Live Experiment Execution (1 Minute)
1. Navigate to **3. Run Experiments**.
2. Select **Pilot Run (25 Questions, Fast)**.
3. Click **Launch Experiment Pipeline**.
4. Observe the live progress bar and status transitions:
   - Dataset preparation
   - Document corpus indexing
   - Controlled noise injection
   - Cross-encoder reranking
   - Paired bootstrap statistical calculation
5. Note: Completed outputs are atomically saved to `outputs/runs/<run_id>/`.

---

### Phase 4: Results Dashboard & Scientific Visualizations (1 Minute)
1. Navigate to **4. Results**.
2. Review the **Condition Summary Table**:
   - Compare Clean Baseline (E0) vs. Noisy (E1, E2, E3).
   - Point out the F1 degradation curve and recovery with cross-encoder reranking.
3. Examine the **Performance Charts** tab:
   - Show Figure 1: Answer F1 vs. Number of Injected Distractor Passages.
   - Show Figure 2: Random vs. Hard Distractor comparison.
4. Open the **Paired Bootstrap CIs** tab:
   - Highlight the 95% confidence intervals and statistical significance indicators.

---

### Phase 5: Failure Analysis & Deliverables Export (1 Minute)
1. Navigate to **5. Failure Analysis**.
2. Show the automated taxonomy distribution:
   - `Clean Correct -> Noisy Incorrect`
   - `Reranking Restored Correct Answer`
   - `Hard Distractor Seduction`
3. Click into a failure case, edit the `researcher_notes` field, and click **Save Researcher Annotations**.
4. Navigate to **7. Reproducibility**:
   - Demonstrate one-click download of the IEEE LaTeX Table (`.tex`) and Markdown report (`.md`).
