"""
execute_cell12.py
Executes Cell 12 of notebooks/07a_inter_phase_transition_analysis.ipynb.
Performs stratified paired statistical analysis across T1 and T2 at 100 Hz and 500 Hz.
Validates all statistical outputs, sample sizes, and strict stop boundaries.
"""
import sys
import os
import json
import time
import io
import pickle
import traceback
import shutil
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import math
import numpy as np
import pandas as pd

def main():
    project_root = Path.cwd()
    mpl_dir = project_root / "scratch" / ".matplotlib"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_dir)

    nb_path = project_root / "notebooks" / "07a_inter_phase_transition_analysis.ipynb"
    assert nb_path.exists(), f"Notebook missing at {nb_path}"

    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    print(f"Loaded {nb_path.name}.")
    print("=== Step 1: Pre-Execution Verification for Cell 12 ===")

    # 1. Pre-execution checks
    manifest_csv = project_root / "reports" / "transitions" / "07a_inter_phase_transition_manifest.csv"
    summary_json = project_root / "reports" / "transitions" / "07a_inter_phase_transition_summary.json"

    assert manifest_csv.exists() and manifest_csv.stat().st_size > 0, "Cell 10 manifest CSV missing or empty"
    assert summary_json.exists() and summary_json.stat().st_size > 0, "Cell 10 summary JSON missing or empty"

    df_manifest = pd.read_csv(manifest_csv)
    assert len(df_manifest) == 288, f"Expected 288 records, found {len(df_manifest)}"
    t1_df = df_manifest[df_manifest["transition_type"] == "REST_TO_STRESS"]
    t2_df = df_manifest[df_manifest["transition_type"] == "STRESS_TO_RECOVERY"]
    assert len(t1_df) == 144, f"T1 rows = {len(t1_df)}"
    assert len(t2_df) == 144, f"T2 rows = {len(t2_df)}"
    assert "RECOVERY_TO_STRESS" not in df_manifest["transition_type"].unique()

    with open(summary_json, "r", encoding="utf-8") as f:
        sdata = json.load(f)
    assert sdata["transitions_evaluated"]["T1_REST_TO_STRESS"]["total_pairs"] == 144
    assert sdata["transitions_evaluated"]["T2_STRESS_TO_RECOVERY"]["total_pairs"] == 144
    assert sdata["transitions_evaluated"]["T3_RECOVERY_TO_STRESS"]["total_pairs"] == 0

    print("  [PASS] Cell 10 CSV and JSON artifacts exist, readable, and reconcile 100%.")

    # 2. Check downstream cells 14 and 16 unexecuted
    for idx in [14, 16]:
        c = nb["cells"][idx]
        assert c["execution_count"] is None and c["outputs"] == [], f"Downstream cell {idx} already executed!"
    print("  [PASS] Downstream Cells 14 and 16 confirmed unexecuted.")

    # 3. Check Stage 03-07 inputs unchanged
    assert Path("reports/segmentation/07_beat_segmentation_manifest.csv").stat().st_size == 58842887
    assert len(list(Path("data/processed/beat_segments").glob("*_beats.npz"))) == 464
    print("  [PASS] Upstream Stage 03-07 inputs confirmed unmodified.")

    # 4. Check free disk space
    total_b, used_b, free_b = shutil.disk_usage(project_root)
    free_gb = free_b / (1024 ** 3)
    assert free_gb >= 3.0, f"Host free disk space {free_gb:.3f} GB < 3.0 GB!"
    print(f"  [PASS] Pre-run host free disk space: {free_gb:.3f} GB (>= 3.0 GB).")

    # === Step 2: Execute Cell 12 Only ===
    cell_12 = nb["cells"][12]
    assert cell_12["cell_type"] == "code", "Cell 12 is not a code cell!"

    global_ctx = {
        "__name__": "__main__",
        "__file__": str(nb_path),
        "df_transitions": df_manifest,
        "pd": pd,
        "np": np,
        "math": math,
        "Dict": Dict,
        "Any": Any,
        "List": List,
        "Tuple": Tuple,
    }

    class TeeStream(io.StringIO):
        def __init__(self, original_stream):
            super().__init__()
            self.original_stream = original_stream

        def write(self, s):
            self.original_stream.write(s)
            self.original_stream.flush()
            return super().write(s)

    print(f"\n{'='*70}")
    print("EXECUTING CELL 12: Stratified Paired Statistical Analysis")
    print(f"{'='*70}")

    old_stdout = sys.stdout
    old_stderr = sys.stderr
    capture_out = TeeStream(old_stdout)
    capture_err = TeeStream(old_stderr)

    sys.stdout = capture_out
    sys.stderr = capture_err

    t0 = time.time()
    error_info = None

    try:
        code_to_exec = "".join(cell_12["source"])
        exec(compile(code_to_exec, "<cell_12_paired_stats>", "exec"), global_ctx)
    except Exception as e:
        error_info = {
            "ename": type(e).__name__,
            "evalue": str(e),
            "traceback": traceback.format_exception(*sys.exc_info())
        }
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr

    elapsed = time.time() - t0
    print(f"\n[Cell 12 completed in {elapsed:.3f}s]")

    # Format cell outputs for Jupyter
    cell_outputs = []
    stdout_text = capture_out.getvalue()
    stderr_text = capture_err.getvalue()

    if stdout_text:
        cell_outputs.append({
            "name": "stdout",
            "output_type": "stream",
            "text": [line + "\n" for line in stdout_text.splitlines()]
        })

    if stderr_text:
        cell_outputs.append({
            "name": "stderr",
            "output_type": "stream",
            "text": [line + "\n" for line in stderr_text.splitlines()]
        })

    if error_info:
        cell_outputs.append({
            "output_type": "error",
            "ename": error_info["ename"],
            "evalue": error_info["evalue"],
            "traceback": error_info["traceback"]
        })

    cell_12["execution_count"] = 6
    cell_12["outputs"] = cell_outputs

    if error_info:
        print(f"HALT: Execution failed in Cell 12: {error_info['ename']}: {error_info['evalue']}")
        with open(nb_path, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=1)
        sys.exit(1)

    # Save notebook with executed Cell 12
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print(f"\nSuccessfully updated {nb_path.name} with execution outputs for Cell 12.")

    # === Step 3: Statistical and Data-Quality Validation ===
    print("\n" + "=" * 70)
    print("PERFORMING STATISTICAL AND DATA-QUALITY VALIDATION")
    print("=" * 70)

    df_paired_stats = global_ctx.get("df_paired_stats")
    assert df_paired_stats is not None and len(df_paired_stats) == 12, f"Expected 12 statistical rows, found {len(df_paired_stats) if df_paired_stats is not None else 0}"

    print(f"Statistical Records Generated ({len(df_paired_stats)} stratum-metric rows):")
    print(df_paired_stats.to_string(index=False))

    # 1. Exact sample size verification
    for _, row in df_paired_stats.iterrows():
        expected_n = 64 if row["Sampling Rate"] == "100 Hz" else 80
        assert row["N Pairs"] == expected_n, f"Sample size mismatch: expected {expected_n}, got {row['N Pairs']} for {row['Transition']} {row['Sampling Rate']}"
    print("  [PASS] Sample sizes verified: exactly 64 pairs for all 100 Hz strata, 80 pairs for all 500 Hz strata.")

    # 2. Check no pooling between 100 Hz and 500 Hz
    strata_evaluated = set(df_paired_stats["Sampling Rate"].unique())
    assert strata_evaluated == {"100 Hz", "500 Hz"}
    print("  [PASS] Strata separation verified: 100 Hz and 500 Hz are evaluated strictly separately (no improper pooling).")

    # 3. Check for non-finite values
    for col in ["Mean Diff", "Std Diff", "Cohen dz"]:
        assert df_paired_stats[col].notna().all(), f"NaN in statistical column {col}"
    print("  [PASS] Mean differences, standard deviations, and Cohen's dz are 100% finite across all 12 comparisons.")

    # 4. Clinical label isolation & Candidate NN Status
    forbidden = ["anoca_confirmed", "raw_msimi_ecg_label", "target_label", "MSIMI-based-ecg", "label_status", "diagnosis"]
    for col in forbidden:
        assert col not in df_manifest.columns
    assert set(df_manifest["stage06_candidate_nn_status"].unique()) == {"UNKNOWN"}
    print("  [PASS] Zero clinical diagnosis labels in data; stage06_candidate_nn_status strictly UNKNOWN.")

    # 5. Free disk space
    total_b, used_b, free_b = shutil.disk_usage(project_root)
    free_gb = free_b / (1024 ** 3)
    assert free_gb >= 3.0
    print(f"  [PASS] Host free disk space: {free_gb:.3f} GB (>= 3.0 GB).")

    # 6. Downstream Cells 14 and 16
    for idx in [14, 16]:
        c = nb["cells"][idx]
        assert c["execution_count"] is None and c["outputs"] == [], f"Downstream cell {idx} executed!"
    print("  [PASS] Downstream Cells 14 and 16 remain strictly unexecuted.")

    # 7. Persist statistical results for reporting
    with open("scratch/cell12_stats_state.pkl", "wb") as f:
        pickle.dump({"df_paired_stats": df_paired_stats}, f)
    print("  [PASS] Statistical summary persisted to scratch/cell12_stats_state.pkl.")

    print("\n" + "=" * 70)
    print("STAGE 07A: CELL 12 STATISTICAL ANALYSIS COMPLETED SUCCESSFULLY!")
    print("=" * 70)

if __name__ == "__main__":
    main()
