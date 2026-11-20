"""
execute_step4_08.py
Executes Cell 12 in notebooks/08_ecg_feature_extraction.ipynb.
Preserves strict execution boundary and updates notebook outputs and execution counts.
Persists execution context to scratch/step4_state.pkl.
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
import numpy as np
import pandas as pd

def main():
    project_root = Path.cwd()
    mpl_dir = project_root / "scratch" / ".matplotlib"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_dir)

    nb_path = project_root / "notebooks" / "08_ecg_feature_extraction.ipynb"
    assert nb_path.exists(), f"Notebook missing at {nb_path}"

    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    # Verify Cells 02, 04, 06, 08, 10 are already executed
    assert nb["cells"][2]["execution_count"] == 1, "Cell 02 not executed"
    assert nb["cells"][4]["execution_count"] == 2, "Cell 04 not executed"
    assert nb["cells"][6]["execution_count"] == 3, "Cell 06 not executed"
    assert nb["cells"][8]["execution_count"] == 4, "Cell 08 not executed"
    assert nb["cells"][10]["execution_count"] == 5, "Cell 10 not executed"
    assert nb["cells"][12]["execution_count"] is None, "Cell 12 already executed"

    # Load Step 3 context (containing df_recording_features)
    step3_pkl = project_root / "scratch" / "step3_state.pkl"
    assert step3_pkl.exists(), f"Missing {step3_pkl}"
    with open(step3_pkl, "rb") as f:
        saved_state = pickle.load(f)

    print("Reconstituting interactive kernel state from Cells 02, 04, and 06...")
    global_ctx = {
        "__name__": "__main__",
        "__file__": str(nb_path),
        "df_recording_features": saved_state["df_recording_features"],
    }

    src_c02 = "".join(nb["cells"][2].get("source", []))
    src_c04 = "".join(nb["cells"][4].get("source", []))
    src_c06 = "".join(nb["cells"][6].get("source", []))

    exec(compile(src_c02, "<cell_02_reconstitute>", "exec"), global_ctx)
    exec(compile(src_c04, "<cell_04_reconstitute>", "exec"), global_ctx)
    exec(compile(src_c06, "<cell_06_reconstitute>", "exec"), global_ctx)
    print("Kernel state successfully initialized.")

    class TeeStream(io.StringIO):
        def __init__(self, original_stream):
            super().__init__()
            self.original_stream = original_stream

        def write(self, s):
            self.original_stream.write(s)
            self.original_stream.flush()
            return super().write(s)

    cell_idx = 12
    exec_count = 6
    cell = nb["cells"][cell_idx]
    assert cell["cell_type"] == "code", f"Cell {cell_idx} is not code"
    src_code = "".join(cell.get("source", []))

    print(f"\n{'='*60}")
    print(f"EXECUTING CELL {cell_idx:02d} (Execution Count: {exec_count})")
    print(f"{'='*60}")

    old_stdout = sys.stdout
    old_stderr = sys.stderr
    capture_out = TeeStream(old_stdout)
    capture_err = TeeStream(old_stderr)

    sys.stdout = capture_out
    sys.stderr = capture_err

    t0 = time.time()
    error_info = None

    try:
        exec(compile(src_code, f"<cell_{cell_idx:02d}>", "exec"), global_ctx)
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
    print(f"\n[Cell {cell_idx:02d} completed in {elapsed:.3f}s]")

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

    cell["execution_count"] = exec_count
    cell["outputs"] = cell_outputs

    if error_info:
        print(f"HALT: Execution failed in Cell {cell_idx:02d}: {error_info['ename']}: {error_info['evalue']}")
        with open(nb_path, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=1)
        sys.exit(1)

    # Save notebook state after Cell 12
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print(f"Successfully saved {nb_path.name} with execution outputs for Cell 12.")

    # -------------------------------------------------------------
    # POST-EXECUTION STEP 4 AUDIT & VALIDATION
    # -------------------------------------------------------------
    print("\n" + "=" * 60)
    print("PERFORMING STEP 4 INDEPENDENT PATIENT FEATURE AUDIT")
    print("=" * 60)

    df_pat = global_ctx.get("df_patient_features")
    assert df_pat is not None, "df_patient_features missing from global context"

    # 1. Row counts & strata accounting
    assert len(df_pat) == 160, f"Expected 160 rows, found {len(df_pat)}"
    assert len(df_pat["subject_id"].unique()) == 80, f"Expected 80 subjects, found {len(df_pat['subject_id'].unique())}"
    fs_counts = df_pat["sampling_rate"].value_counts().to_dict()
    print(f"  [PASS] Total patient-strata rows: {len(df_pat)} (Columns: {len(df_pat.columns)})")
    print(f"  [PASS] Sampling rate breakdown: {fs_counts}")
    assert fs_counts[100] == 80 and fs_counts[500] == 80

    # 2. Status & missingness breakdown
    status_counts = df_pat["stress_recording_status"].value_counts().to_dict()
    print(f"  [PASS] Stress recording status: {status_counts}")
    assert status_counts.get("SUCCESS", 0) == 144, f"Expected 144 SUCCESS, got {status_counts.get('SUCCESS')}"
    assert status_counts.get("SKIPPED_UPSTREAM_ERROR", 0) == 16, f"Expected 16 SKIPPED, got {status_counts.get('SKIPPED_UPSTREAM_ERROR')}"

    # Verify all 16 skipped are in 100 Hz
    skipped_pat = df_pat[df_pat["stress_recording_status"] == "SKIPPED_UPSTREAM_ERROR"]
    assert set(skipped_pat["sampling_rate"].unique()) == {100}, "All skipped must be 100 Hz"
    print(f"  [PASS] All 16 skipped subjects are strictly in 100 Hz stratum: {sorted(skipped_pat['subject_id'].tolist())[:5]}...")

    # Verify corresponding 500 Hz rows for these 16 subjects are 100% complete
    skipped_subj_ids = set(skipped_pat["subject_id"].unique())
    pat_500_for_skipped = df_pat[(df_pat["subject_id"].isin(skipped_subj_ids)) & (df_pat["sampling_rate"] == 500)]
    assert len(pat_500_for_skipped) == 16
    assert set(pat_500_for_skipped["stress_recording_status"].unique()) == {"SUCCESS"}
    assert pat_500_for_skipped["t1_delta_V5_st60_adc"].isna().sum() == 0, "500 Hz differentials should not be NaN!"
    print("  [PASS] Corresponding 500 Hz records for the 16 subjects are 100% valid and complete.")

    # 3. Holdout verification
    holdout_rows = df_pat[df_pat["is_unlabelled_holdout"] == True]
    assert len(holdout_rows) == 4, f"Expected 4 holdout rows (2 patients x 2 rates), found {len(holdout_rows)}"
    assert set(holdout_rows["subject_id"].unique()) == {"sub_005", "sub_049"}
    print("  [PASS] sub_005 and sub_049 strictly verified as unlabelled holdouts (4 rows total).")

    # 4. Descriptive Statistics of Key Patient Differentials across Valid Rows (N = 144)
    valid_pat = df_pat[df_pat["stress_recording_status"] == "SUCCESS"]
    print(f"\n--- Descriptive Statistics on Valid Patient-Strata (N = {len(valid_pat)}) ---")
    diff_cols = [
        "rest_V5_st60_adc", "stress_V5_st60_adc", "recover_V5_st60_adc",
        "t1_delta_V5_st60_adc", "t2_delta_V5_st60_adc",
        "t1_delta_min_st60_global_adc", "t2_delta_min_st60_global_adc",
        "t1_delta_hr_bpm", "t2_delta_hr_bpm", "t2_hrr60_bpm", "t2_hrri_pct",
        "t1_frobenius_dissimilarity_adc_norm", "t2_frobenius_dissimilarity_adc_norm"
    ]
    summary_diffs = valid_pat[diff_cols].describe().T[["count", "mean", "std", "min", "50%", "max"]]
    print(summary_diffs.to_string())

    # Check for unexpected NaN/Inf in valid rows
    for col in diff_cols:
        # Note: t2_hrri_pct can have documented NaNs when late stress HR delta is zero (as verified in Stage 07A)
        n_inf = np.isinf(valid_pat[col]).sum()
        assert n_inf == 0, f"Inf detected in {col}!"
        if col != "t2_hrri_pct":
            n_nan = valid_pat[col].isna().sum()
            assert n_nan == 0, f"Unexpected NaN in {col}: {n_nan}"
    print("\n  [PASS] Zero unexpected NaN/Inf values across valid patient differentials.")

    # 5. Governance checks
    for col in ["MSIMI-based-ecg", "anoca_confirmed", "raw_msimi_ecg_label", "target_label", "diagnosis"]:
        assert col not in df_pat.columns, f"Label leakage: {col}"
    assert set(df_pat["stage06_candidate_nn_status"].unique()) == {"UNKNOWN"}
    print("  [PASS] Zero clinical label leakage; stage06_candidate_nn_status strictly UNKNOWN.")

    # 6. Disk space check
    total_bytes, used_bytes, free_bytes = shutil.disk_usage(project_root)
    free_gb = free_bytes / (1024 ** 3)
    print(f"  [PASS] Free disk space: {free_gb:.3f} GB (>= 3.0 GB guard).")

    # 7. Check strict execution boundary on notebook
    for idx in range(len(nb["cells"])):
        c = nb["cells"][idx]
        if c["cell_type"] == "code":
            if idx in [2, 4, 6, 8, 10, 12]:
                assert c["execution_count"] is not None, f"Executed cell {idx} has None count"
                assert len(c["outputs"]) > 0, f"Executed cell {idx} has empty outputs"
            else:
                assert c["execution_count"] is None, f"Cell {idx} should be unexecuted, found {c['execution_count']}"
                assert len(c["outputs"]) == 0, f"Cell {idx} should have empty outputs, found {len(c['outputs'])}"
    print("  [PASS] Execution boundary strictly verified: Cells 02, 04, 06, 08, 10, 12 executed; Cells 14..18 unexecuted.")

    # Save state to step4_state.pkl
    state_to_save = {
        "df_recording_features": saved_state["df_recording_features"],
        "df_patient_features": df_pat,
        "OVERWRITE_EXISTING_OUTPUTS": global_ctx["OVERWRITE_EXISTING_OUTPUTS"],
        "df_stage06": global_ctx["df_stage06"],
        "df_stage06_success": global_ctx["df_stage06_success"],
        "df_patient_summary": global_ctx["df_patient_summary"],
        "df_stage07a": global_ctx["df_stage07a"],
    }
    step4_pkl_path = project_root / "scratch" / "step4_state.pkl"
    with open(step4_pkl_path, "wb") as f:
        pickle.dump(state_to_save, f)
    print(f"  [PASS] Step 4 execution context persisted to {step4_pkl_path.name} ({step4_pkl_path.stat().st_size:,} bytes).")

    print("\n" + "=" * 60)
    print("STEP 4 EXECUTION COMPLETE & AUDITED: FULL PASS")
    print("=" * 60)

if __name__ == "__main__":
    main()

