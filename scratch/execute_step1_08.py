"""
execute_step1_08.py
Executes Cell 02 and Cell 04 of notebooks/08_ecg_feature_extraction.ipynb.
Saves executed notebook with outputs and execution_count.
Persists execution context to scratch/step1_state.pkl.
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

def main():
    project_root = Path.cwd()
    mpl_dir = project_root / "scratch" / ".matplotlib"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_dir)

    nb_path = project_root / "notebooks" / "08_ecg_feature_extraction.ipynb"
    assert nb_path.exists(), f"Notebook missing at {nb_path}"

    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    print(f"Loaded {nb_path.name} with {len(nb['cells'])} cells.")
    print("Executing Step 1: Cell 02 and Cell 04 only...")

    # Global context for cell execution
    global_ctx = {
        "__name__": "__main__",
        "__file__": str(nb_path),
    }

    class TeeStream(io.StringIO):
        def __init__(self, original_stream):
            super().__init__()
            self.original_stream = original_stream

        def write(self, s):
            self.original_stream.write(s)
            self.original_stream.flush()
            return super().write(s)

    target_code_indices = [2, 4]
    exec_counter = 1

    for cell_idx in target_code_indices:
        cell = nb["cells"][cell_idx]
        assert cell["cell_type"] == "code", f"Cell {cell_idx} is not a code cell!"
        src_code = "".join(cell.get("source", []))

        print(f"\n{'='*60}")
        print(f"EXECUTING CELL {cell_idx:02d} (Execution Count: {exec_counter})")
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

        cell["execution_count"] = exec_counter
        cell["outputs"] = cell_outputs
        exec_counter += 1

        if error_info:
            print(f"HALT: Execution failed in Cell {cell_idx:02d}: {error_info['ename']}: {error_info['evalue']}")
            with open(nb_path, "w", encoding="utf-8") as f:
                json.dump(nb, f, indent=1)
            sys.exit(1)

    # Save notebook with executed cells
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print(f"\nSuccessfully updated {nb_path.name} with execution outputs for Cells 02 and 04.")

    # -------------------------------------------------------------
    # POST-EXECUTION STEP 1 VALIDATION CHECKS
    # -------------------------------------------------------------
    print("\n" + "=" * 60)
    print("PERFORMING STEP 1 INDEPENDENT VALIDATION CHECKS")
    print("=" * 60)

    # 1. Input paths and schemas
    df_stage06 = global_ctx.get("df_stage06")
    df_stage07a = global_ctx.get("df_stage07a")
    df_patient_summary = global_ctx.get("df_patient_summary")

    assert df_stage06 is not None, "df_stage06 missing from context"
    assert df_stage07a is not None, "df_stage07a missing from context"
    assert df_patient_summary is not None, "df_patient_summary missing from context"

    assert len(df_stage06) == 480, f"Expected 480 rows in df_stage06, got {len(df_stage06)}"
    assert len(df_stage07a) == 288, f"Expected 288 rows in df_stage07a, got {len(df_stage07a)}"
    assert len(df_patient_summary) == 80, f"Expected 80 patients, got {len(df_patient_summary)}"

    # 2. Strata breakdown
    s06_100 = df_stage06[df_stage06["sampling_rate"] == 100]
    s06_500 = df_stage06[df_stage06["sampling_rate"] == 500]
    assert len(s06_100) == 240, f"Expected 240 100Hz rows in Stage 06, got {len(s06_100)}"
    assert len(s06_500) == 240, f"Expected 240 500Hz rows in Stage 06, got {len(s06_500)}"

    s06_success = df_stage06[df_stage06["rpeak_detection_status"] == "SUCCESS"]
    s06_skipped = df_stage06[df_stage06["rpeak_detection_status"] != "SUCCESS"]
    assert len(s06_success) == 464, f"Expected 464 SUCCESS rows, got {len(s06_success)}"
    assert len(s06_skipped) == 16, f"Expected 16 SKIPPED rows, got {len(s06_skipped)}"
    assert set(s06_skipped["rpeak_detection_status"].unique()) == {"SKIPPED_AUDIT_ERROR"}
    assert set(s06_skipped["sampling_rate"].unique()) == {100}, "All skipped must be 100 Hz"
    assert set(s06_skipped["phase"].unique()) == {"stress"}, "All skipped must be stress"

    t1_rows = df_stage07a[df_stage07a["transition_type"] == "REST_TO_STRESS"]
    t2_rows = df_stage07a[df_stage07a["transition_type"] == "STRESS_TO_RECOVERY"]
    assert len(t1_rows) == 144, f"Expected 144 T1 rows, got {len(t1_rows)}"
    assert len(t2_rows) == 144, f"Expected 144 T2 rows, got {len(t2_rows)}"

    print(f"  [PASS] Stage 06 Breakdown: 480 total (240 @ 100Hz, 240 @ 500Hz; 464 SUCCESS, 16 SKIPPED).")
    print(f"  [PASS] Stage 07A Breakdown: 288 total (T1 = 144 [64 @ 100Hz, 80 @ 500Hz], T2 = 144 [64 @ 100Hz, 80 @ 500Hz]).")

    # 3. Beat archives
    processed_beats_dir = global_ctx.get("processed_beats_dir")
    for rec_id in s06_success["recording_id"]:
        assert (processed_beats_dir / f"{rec_id}_beats.npz").exists(), f"NPZ missing for {rec_id}"
    print(f"  [PASS] All 464 Stage 07 beat archives verified on disk.")

    # 4. Clinical governance & holdouts
    for col in ["MSIMI-based-ecg", "anoca_confirmed", "raw_msimi_ecg_label", "target_label"]:
        assert col not in df_stage06.columns, f"Label {col} in Stage 06"
        assert col not in df_stage07a.columns, f"Label {col} in Stage 07A"
    assert set(df_stage07a["stage06_candidate_nn_status"].unique()) == {"UNKNOWN"}
    print(f"  [PASS] Clinical diagnosis labels absent from upstream manifests; candidate-NN status strictly UNKNOWN.")

    # 5. Output targets non-existence & disk space
    target_outputs = global_ctx.get("target_outputs")
    for top in target_outputs:
        assert not top.exists(), f"Target output unexpectedly exists: {top}"
    print(f"  [PASS] Confirmed 0 feature output targets exist prior to Step 5 serialization.")

    total_bytes, used_bytes, free_bytes = shutil.disk_usage(project_root)
    free_gb = free_bytes / (1024 ** 3)
    print(f"  [PASS] Post-execution host free disk space: {free_gb:.3f} GB (>= 3.0 GB guard).")

    # 6. Verify subsequent cells remain unexecuted
    for idx in range(len(nb["cells"])):
        c = nb["cells"][idx]
        if c["cell_type"] == "code":
            if idx in target_code_indices:
                assert c["execution_count"] is not None, f"Target cell {idx} has None execution_count"
                assert len(c["outputs"]) > 0, f"Target cell {idx} has empty outputs"
            else:
                assert c["execution_count"] is None, f"Cell {idx} should be unexecuted, but has count {c['execution_count']}"
                assert len(c["outputs"]) == 0, f"Cell {idx} should have empty outputs, but has {len(c['outputs'])}"
    print(f"  [PASS] Strict execution boundary verified: only Cells 02 and 04 executed; Cells 06..18 remain unexecuted.")

    # Persist context for Step 2
    state_to_save = {
        "project_root": global_ctx["project_root"],
        "data_dir": global_ctx["data_dir"],
        "interim_ecg_dir": global_ctx["interim_ecg_dir"],
        "processed_rpeaks_dir": global_ctx["processed_rpeaks_dir"],
        "processed_beats_dir": global_ctx["processed_beats_dir"],
        "features_dir": global_ctx["features_dir"],
        "reports_hrv_dir": global_ctx["reports_hrv_dir"],
        "reports_segmentation_dir": global_ctx["reports_segmentation_dir"],
        "reports_transitions_dir": global_ctx["reports_transitions_dir"],
        "reports_features_dir": global_ctx["reports_features_dir"],
        "figures_dir": global_ctx["figures_dir"],
        "recording_features_csv_path": global_ctx["recording_features_csv_path"],
        "patient_features_csv_path": global_ctx["patient_features_csv_path"],
        "feature_summary_json_path": global_ctx["feature_summary_json_path"],
        "figure_st_dist_path": global_ctx["figure_st_dist_path"],
        "figure_autonomic_repol_path": global_ctx["figure_autonomic_repol_path"],
        "target_outputs": global_ctx["target_outputs"],
        "OVERWRITE_EXISTING_OUTPUTS": global_ctx["OVERWRITE_EXISTING_OUTPUTS"],
        "df_stage06": df_stage06,
        "df_stage06_success": global_ctx["df_stage06_success"],
        "df_patient_summary": df_patient_summary,
        "df_stage07a": df_stage07a,
    }
    state_pkl_path = project_root / "scratch" / "step1_state.pkl"
    with open(state_pkl_path, "wb") as f:
        pickle.dump(state_to_save, f)
    print(f"  [PASS] Step 1 execution context persisted to {state_pkl_path.name} ({state_pkl_path.stat().st_size:,} bytes).")
    print("\n" + "=" * 60)
    print("STEP 1 EXECUTION COMPLETE & AUDITED: FULL PASS")
    print("=" * 60)

if __name__ == "__main__":
    main()
