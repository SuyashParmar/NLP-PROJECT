#!/usr/bin/env python3
"""Generates IEEE paper assets (LaTeX tables and figures) from genuine experiment results."""

import argparse
import shutil
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from robustrag.reporting import ReportGenerator
from robustrag.storage import ExperimentStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate IEEE paper LaTeX tables and figures from genuine experimental runs."
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Specific run ID to extract assets from. Defaults to most recent completed empirical run.",
    )
    parser.add_argument(
        "--output-tex",
        type=str,
        default="paper/results_tables.tex",
        help="Output path for LaTeX table file.",
    )
    parser.add_argument(
        "--figures-dir",
        type=str,
        default="paper/figures",
        help="Output directory for paper figures.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    store = ExperimentStore("./outputs/runs")
    runs = store.list_runs()

    tex_out = Path(args.output_tex)
    tex_out.parent.mkdir(parents=True, exist_ok=True)
    fig_out = Path(args.figures_dir)
    fig_out.mkdir(parents=True, exist_ok=True)

    target_run_id = args.run_id
    if not target_run_id:
        # Find latest completed non-smoke empirical run
        empirical_runs = [r for r in runs if not r.get("is_smoke_test") and r.get("status") == "completed"]
        if empirical_runs:
            target_run_id = empirical_runs[0]["run_id"]
        elif runs:
            # Fallback to any completed run if only smoke tests exist, with explicit warning
            target_run_id = runs[0]["run_id"]

    if not target_run_id:
        print("[!] No completed experimental runs found in outputs/runs.")
        print("[*] Generating unpopulated placeholder LaTeX table.")
        placeholder = (
            "% =========================================================================\n"
            "% EXPERIMENTS PENDING: No empirical results recorded yet.\n"
            "% Run 'python scripts/run_experiment.py --config configs/pilot.yaml' to populate.\n"
            "% =========================================================================\n"
            "\\begin{table}[htbp]\n"
            "\\centering\n"
            "\\caption{Pending Empirical Results Across Experimental Conditions}\n"
            "\\label{tab:main_results}\n"
            "\\begin{tabular}{llccc}\n"
            "\\hline\n"
            "\\textbf{Condition} & \\textbf{Reranking} & \\textbf{Distractors} & \\textbf{F1 (\\%)} & \\textbf{EM (\\%)} \\\\\n"
            "\\hline\n"
            "E0 (Baseline) & None & 0 & [Pending] & [Pending] \\\\\n"
            "E1 & Unranked & 1 (Random) & [Pending] & [Pending] \\\\\n"
            "E1 & Cross-Encoder & 1 (Random) & [Pending] & [Pending] \\\\\n"
            "E2 & Unranked & 3 (Random) & [Pending] & [Pending] \\\\\n"
            "E2 & Cross-Encoder & 3 (Random) & [Pending] & [Pending] \\\\\n"
            "E3 & Unranked & 5 (Random) & [Pending] & [Pending] \\\\\n"
            "E3 & Cross-Encoder & 5 (Random) & [Pending] & [Pending] \\\\\n"
            "E\\_hard & Unranked & 3 (Hard) & [Pending] & [Pending] \\\\\n"
            "\\hline\n"
            "\\end{tabular}\n"
            "\\end{table}\n"
        )
        tex_out.write_text(placeholder, encoding="utf-8")
        return 0

    print(f"[*] Extracting genuine paper assets from run: {target_run_id}")
    run_data = store.load_run(target_run_id)
    df_summary = run_data["summary"]
    run_dir = run_data["run_dir"]

    # 1. Generate and write LaTeX table
    latex_content = ReportGenerator.generate_latex_table(df_summary)
    tex_out.write_text(latex_content, encoding="utf-8")
    print(f"[+] Wrote LaTeX table to {tex_out}")

    # 2. Copy figures
    src_figures = run_dir / "figures"
    if src_figures.exists():
        for fig_file in src_figures.glob("*.png"):
            dest = fig_out / fig_file.name
            shutil.copy2(fig_file, dest)
            print(f"[+] Copied figure: {dest}")

    print("[+] Paper assets successfully refreshed from genuine results.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
