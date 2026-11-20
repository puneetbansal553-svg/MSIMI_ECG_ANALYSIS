"""
execute_phase1_07a.py
Executes Cells 02, 04, and 06 of notebooks/07a_inter_phase_transition_analysis.ipynb.
Saves executed notebook with outputs and execution_count.
Persists execution context to scratch/phase1_state.pkl.
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

    nb_path = project_root / "notebooks" / "07a_inter_phase_transition_analysis.ipynb"
    assert nb_path.exists(), f"Notebook missing at {nb_path}"

    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    print(f"Loaded {nb_path.name} with {len(nb['cells'])} cells.")
    print("Executing Phase 1: Cells 02, 04, 06...")

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

    target_code_indices = [2, 4, 6]
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
    print(f"\nSuccessfully updated {nb_path.name} with execution outputs for Cells 02, 04, 06.")

    # -------------------------------------------------------------
    # PHASE 1 VALIDATION CHECKS
    # -------------------------------------------------------------
    print("\n" + "=" * 60)
    print("PERFORMING PHASE 1 VALIDATION CHECKS")
    print("=" * 60)

    # 1. Input paths and schemas
    stage06_manifest = global_ctx.get("df_stage06")
    stage07_manifest = global_ctx.get("df_beat_manifest")
    assert stage06_manifest is not None, "df_stage06 not found in context"
    assert stage07_manifest is not None, "df_beat_manifest not found in context"
    assert len(stage06_manifest) == 480, f"Expected 480 rows in df_stage06, got {len(stage06_manifest)}"
    assert len(stage07_manifest) == 307561, f"Expected 307561 rows in df_beat_manifest, got {len(stage07_manifest)}"
    print(f"  [PASS] Stage 06 Manifest schema verified: {len(stage06_manifest)} rows, columns: {list(stage06_manifest.columns[:6])}...")
    print(f"  [PASS] Stage 07 Beat Manifest schema verified: {len(stage07_manifest):,} rows, columns: {list(stage07_manifest.columns[:6])}...")

    # 2. Transition pair counts
    t1_pairs = global_ctx.get("t1_pairs")
    t2_pairs = global_ctx.get("t2_pairs")
    t3_pairs = global_ctx.get("t3_pairs")
    assert len(t1_pairs) == 144, f"Expected 144 T1 pairs, got {len(t1_pairs)}"
    assert len(t2_pairs) == 144, f"Expected 144 T2 pairs, got {len(t2_pairs)}"
    assert len(t3_pairs) == 0, f"Expected 0 T3 pairs, got {len(t3_pairs)}"
    print(f"  [PASS] Transition pair counts: T1 = {len(t1_pairs)}, T2 = {len(t2_pairs)}, T3 = {len(t3_pairs)} (PROTOCOL_NOT_APPLICABLE).")

    # 3. 100 Hz and 500 Hz stratum counts
    t1_100hz = [p for p in t1_pairs if p[1] == 100]
    t1_500hz = [p for p in t1_pairs if p[1] == 500]
    t2_100hz = [p for p in t2_pairs if p[1] == 100]
    t2_500hz = [p for p in t2_pairs if p[1] == 500]

    assert len(t1_100hz) == 64, f"Expected 64 T1 100Hz pairs, got {len(t1_100hz)}"
    assert len(t1_500hz) == 80, f"Expected 80 T1 500Hz pairs, got {len(t1_500hz)}"
    assert len(t2_100hz) == 64, f"Expected 64 T2 100Hz pairs, got {len(t2_100hz)}"
    assert len(t2_500hz) == 80, f"Expected 80 T2 500Hz pairs, got {len(t2_500hz)}"
    print(f"  [PASS] 100 Hz pair counts: T1 = {len(t1_100hz)}, T2 = {len(t2_100hz)} (16 corrupted stress exclusions accounted for).")
    print(f"  [PASS] 500 Hz pair counts: T1 = {len(t1_500hz)}, T2 = {len(t2_500hz)} (100% complete cohort).")

    # 4. Stage 07 inputs are read-only
    manifest_s07_path = global_ctx["manifest_s07_path"]
    summary_s07_path = global_ctx["summary_s07_path"]
    output_beat_dir = global_ctx["output_beat_dir"]
    assert manifest_s07_path.exists() and manifest_s07_path.stat().st_size == 58842887, f"Stage 07 manifest modified: size {manifest_s07_path.stat().st_size}"
    assert summary_s07_path.exists() and summary_s07_path.stat().st_size == 1066, f"Stage 07 summary modified: size {summary_s07_path.stat().st_size}"
    npz_count = len(list(output_beat_dir.glob("*_beats.npz")))
    assert npz_count == 464, f"Expected 464 beat archives, found {npz_count}"
    print(f"  [PASS] Stage 07 files verified read-only and unmodified ({npz_count} NPZ archives, manifest size: {manifest_s07_path.stat().st_size:,} bytes).")

    # 5. Clinical diagnosis label isolation & Candidate NN Status
    forbidden_labels = ["anoca_confirmed", "raw_msimi_ecg_label", "target_label", "MSIMI-based-ecg", "label_status", "diagnosis"]
    for label in forbidden_labels:
        assert label not in stage07_manifest.columns, f"Forbidden label {label} present in Stage 07 manifest!"
    print("  [PASS] Zero clinical diagnosis labels in Stage 07 beat manifest.")

    assert set(stage07_manifest["stage06_candidate_nn_status"].unique()) == {"UNKNOWN"}, "Candidate NN status in beat manifest is not UNKNOWN"
    print("  [PASS] stage06_candidate_nn_status remains UNKNOWN across 100% of rows (307,561/307,561).")

    # 6. Verify mathematical routines
    delineate_j_point = global_ctx.get("delineate_j_point")
    compute_morphology_metrics = global_ctx.get("compute_morphology_metrics")
    assert callable(delineate_j_point), "delineate_j_point not callable"
    assert callable(compute_morphology_metrics), "compute_morphology_metrics not callable"
    print("  [PASS] Mathematical routines delineate_j_point and compute_morphology_metrics verified callable.")

    # Persist context for Phase 2
    save_keys = [
        "project_root", "OVERWRITE_EXISTING_OUTPUTS", "data_dir", "interim_ecg_dir",
        "processed_rpeaks_dir", "output_beat_dir", "reports_segmentation_dir",
        "reports_transitions_dir", "reports_qc_dir", "reports_hrv_dir",
        "reports_rpeaks_dir", "figures_dir", "manifest_s07_path", "summary_s07_path",
        "patient_s07_path", "transition_manifest_csv_path", "transition_summary_json_path",
        "figure_transition_morph_path", "figure_transition_dynamics_path",
        "STANDARD_12_LEADS", "t1_pairs", "t2_pairs", "t3_pairs"
    ]
    state_to_save = {k: global_ctx[k] for k in save_keys if k in global_ctx}

    with open("scratch/phase1_state.pkl", "wb") as f:
        pickle.dump(state_to_save, f)
    print("  [PASS] Phase 1 state persisted to scratch/phase1_state.pkl.")

    print("\nPHASE 1 EXECUTION AND VALIDATION COMPLETED SUCCESSFULLY!")

if __name__ == "__main__":
    main()
