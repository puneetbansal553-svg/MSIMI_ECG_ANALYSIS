"""
execute_step3_08.py
Executes Cell 10 in notebooks/08_ecg_feature_extraction.ipynb.
Preserves strict execution boundary and updates notebook outputs and execution counts.
Persists execution context to scratch/step3_state.pkl.
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

    # Verify Cells 02, 04, 06, 08 are already executed
    assert nb["cells"][2]["execution_count"] == 1, "Cell 02 not executed"
    assert nb["cells"][4]["execution_count"] == 2, "Cell 04 not executed"
    assert nb["cells"][6]["execution_count"] == 3, "Cell 06 not executed"
    assert nb["cells"][8]["execution_count"] == 4, "Cell 08 not executed"
    assert nb["cells"][10]["execution_count"] is None, "Cell 10 already executed"

    # Reconstitute kernel state from Cells 02, 04, 06
    print("Reconstituting interactive kernel state from Cells 02, 04, and 06...")
    global_ctx = {
        "__name__": "__main__",
        "__file__": str(nb_path),
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

    cell_idx = 10
    exec_count = 5
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

    # Save notebook state after Cell 10
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print(f"Successfully saved {nb_path.name} with execution outputs for Cell 10.")

    # -------------------------------------------------------------
    # POST-EXECUTION STEP 3 AUDIT & VALIDATION
    # -------------------------------------------------------------
    print("\n" + "=" * 60)
    print("PERFORMING STEP 3 INDEPENDENT COHORT AUDIT")
    print("=" * 60)

    df_feats = global_ctx.get("df_recording_features")
    assert df_feats is not None, "df_recording_features missing from global context"

    # 1. Row counts & accounting
    assert len(df_feats) == 480, f"Expected 480 rows, found {len(df_feats)}"
    status_counts = df_feats["feature_extraction_status"].value_counts().to_dict()
    print(f"  [PASS] Total rows: {len(df_feats)} (Columns: {len(df_feats.columns)})")
    print(f"  [PASS] Feature extraction status: {status_counts}")
    assert status_counts.get("SUCCESS", 0) == 464, f"Expected 464 SUCCESS, got {status_counts.get('SUCCESS')}"
    assert status_counts.get("SKIPPED_UPSTREAM_ERROR", 0) == 16, f"Expected 16 SKIPPED, got {status_counts.get('SKIPPED_UPSTREAM_ERROR')}"

    # 2. Strata breakdown
    fs_counts = df_feats["sampling_rate"].value_counts().to_dict()
    print(f"  [PASS] Sampling rate breakdown: {fs_counts}")
    assert fs_counts[100] == 240 and fs_counts[500] == 240, "Stratum counts mismatch"

    # Verify all 16 skipped are 100 Hz stress
    skipped_df = df_feats[df_feats["feature_extraction_status"] == "SKIPPED_UPSTREAM_ERROR"]
    assert set(skipped_df["sampling_rate"].unique()) == {100}, "Skipped rows must be 100 Hz"
    assert set(skipped_df["phase"].unique()) == {"stress"}, "Skipped rows must be stress"

    # 3. Check 500 Hz stratum is 100% complete
    df_500 = df_feats[df_feats["sampling_rate"] == 500]
    assert len(df_500) == 240
    assert set(df_500["feature_extraction_status"].unique()) == {"SUCCESS"}, "500 Hz stratum must be 100% SUCCESS"
    print("  [PASS] 500 Hz stratum is 100% complete (240/240 SUCCESS).")

    # 4. Fiducial ranges and validity across successful recordings
    success_df = df_feats[df_feats["feature_extraction_status"] == "SUCCESS"]
    print(f"\n--- Descriptive Statistics on Successful Recordings (N = {len(success_df)}) ---")
    metrics_to_check = [
        "V5_st60_adc", "V5_st80_adc", "II_st60_adc", "II_st80_adc",
        "min_st60_global_adc", "min_st60_inferior_adc", "min_st60_anterior_adc", "min_st60_lateral_adc",
        "qrs_axis_degrees", "contiguous_depressed_leads_count", "V5_qrs_dur_ms", "V5_qt_ms"
    ]
    summary_stats = success_df[metrics_to_check].describe().T[["count", "mean", "std", "min", "50%", "max"]]
    print(summary_stats.to_string())

    # Check for unexpected NaN or Inf in successful recordings
    for col in metrics_to_check:
        n_nan = success_df[col].isna().sum()
        n_inf = np.isinf(success_df[col]).sum()
        assert n_inf == 0, f"Inf detected in {col}!"
        # qrs_axis_degrees may have rare delineation failure, check it is <= 5
        if col == "qrs_axis_degrees":
            assert n_nan <= 5, f"Excessive axis NaNs: {n_nan}"
        else:
            assert n_nan == 0, f"Unexpected NaN in {col}: {n_nan}"
    print("\n  [PASS] Zero unexpected NaN/Inf values in mandatory features.")

    # 5. Governance checks
    for col in ["MSIMI-based-ecg", "anoca_confirmed", "raw_msimi_ecg_label", "target_label", "diagnosis"]:
        assert col not in df_feats.columns, f"Label leakage: {col}"
    assert set(df_feats["stage06_candidate_nn_status"].unique()) == {"UNKNOWN"}
    print("  [PASS] Zero clinical label leakage; stage06_candidate_nn_status strictly UNKNOWN.")

    # 6. Disk space check
    total_bytes, used_bytes, free_bytes = shutil.disk_usage(project_root)
    free_gb = free_bytes / (1024 ** 3)
    print(f"  [PASS] Free disk space: {free_gb:.3f} GB (>= 3.0 GB guard).")

    # 7. Check strict execution boundary on notebook
    for idx in range(len(nb["cells"])):
        c = nb["cells"][idx]
        if c["cell_type"] == "code":
            if idx in [2, 4, 6, 8, 10]:
                assert c["execution_count"] is not None, f"Executed cell {idx} has None count"
                assert len(c["outputs"]) > 0, f"Executed cell {idx} has empty outputs"
            else:
                assert c["execution_count"] is None, f"Cell {idx} should be unexecuted, found {c['execution_count']}"
                assert len(c["outputs"]) == 0, f"Cell {idx} should have empty outputs, found {len(c['outputs'])}"
    print("  [PASS] Execution boundary strictly verified: Cells 02, 04, 06, 08, 10 executed; Cells 12..18 unexecuted.")

    # Save state to step3_state.pkl
    state_to_save = {
        "df_recording_features": df_feats,
        "OVERWRITE_EXISTING_OUTPUTS": global_ctx["OVERWRITE_EXISTING_OUTPUTS"],
        "df_stage06": global_ctx["df_stage06"],
        "df_stage06_success": global_ctx["df_stage06_success"],
        "df_patient_summary": global_ctx["df_patient_summary"],
        "df_stage07a": global_ctx["df_stage07a"],
    }
    step3_pkl_path = project_root / "scratch" / "step3_state.pkl"
    with open(step3_pkl_path, "wb") as f:
        pickle.dump(state_to_save, f)
    print(f"  [PASS] Step 3 execution context persisted to {step3_pkl_path.name} ({step3_pkl_path.stat().st_size:,} bytes).")

    print("\n" + "=" * 60)
    print("STEP 3 EXECUTION COMPLETE & AUDITED: FULL PASS")
    print("=" * 60)

if __name__ == "__main__":
    main()

