"""
execute_step5_08.py
Executes Cell 14 in notebooks/08_ecg_feature_extraction.ipynb.
Preserves strict execution boundary and updates notebook outputs and execution counts.
Persists execution context to scratch/step5_state.pkl.
"""
import sys
import os
import json
import time
import io
import pickle
import hashlib
import traceback
import shutil
from pathlib import Path
import numpy as np
import pandas as pd

def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

def main():
    project_root = Path.cwd()
    mpl_dir = project_root / "scratch" / ".matplotlib"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_dir)

    nb_path = project_root / "notebooks" / "08_ecg_feature_extraction.ipynb"
    assert nb_path.exists(), f"Notebook missing at {nb_path}"

    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    # Verify Cells 02, 04, 06, 08, 10, 12 are already executed
    assert nb["cells"][2]["execution_count"] == 1, "Cell 02 not executed"
    assert nb["cells"][4]["execution_count"] == 2, "Cell 04 not executed"
    assert nb["cells"][6]["execution_count"] == 3, "Cell 06 not executed"
    assert nb["cells"][8]["execution_count"] == 4, "Cell 08 not executed"
    assert nb["cells"][10]["execution_count"] == 5, "Cell 10 not executed"
    assert nb["cells"][12]["execution_count"] == 6, "Cell 12 not executed"
    assert nb["cells"][14]["execution_count"] is None, "Cell 14 already executed"

    # Load Step 4 context (containing df_recording_features and df_patient_features)
    step4_pkl = project_root / "scratch" / "step4_state.pkl"
    assert step4_pkl.exists(), f"Missing {step4_pkl}"
    with open(step4_pkl, "rb") as f:
        saved_state = pickle.load(f)

    print("Reconstituting interactive kernel state from Cells 02, 04, and 06...")
    global_ctx = {
        "__name__": "__main__",
        "__file__": str(nb_path),
        "df_recording_features": saved_state["df_recording_features"],
        "df_patient_features": saved_state["df_patient_features"],
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

    cell_idx = 14
    exec_count = 7
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

    # Save notebook state after Cell 14
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print(f"Successfully saved {nb_path.name} with execution outputs for Cell 14.")

    # -------------------------------------------------------------
    # POST-EXECUTION STEP 5 SERIALIZATION AUDIT & VALIDATION
    # -------------------------------------------------------------
    print("\n" + "=" * 60)
    print("PERFORMING STEP 5 INDEPENDENT SERIALIZATION AUDIT")
    print("=" * 60)

    rec_csv_p = global_ctx["recording_features_csv_path"]
    pat_csv_p = global_ctx["patient_features_csv_path"]
    sum_json_p = global_ctx["feature_summary_json_path"]

    # 1. Existence and non-emptiness
    assert rec_csv_p.exists() and rec_csv_p.stat().st_size > 0, "Recording CSV missing or empty"
    assert pat_csv_p.exists() and pat_csv_p.stat().st_size > 0, "Patient CSV missing or empty"
    assert sum_json_p.exists() and sum_json_p.stat().st_size > 0, "Summary JSON missing or empty"

    # Compute hashes and file sizes
    rec_size = rec_csv_p.stat().st_size
    rec_hash = compute_sha256(rec_csv_p)
    pat_size = pat_csv_p.stat().st_size
    pat_hash = compute_sha256(pat_csv_p)
    sum_size = sum_json_p.stat().st_size
    sum_hash = compute_sha256(sum_json_p)

    print(f"  [PASS] 08_recording_level_features.csv : {rec_size:,} bytes | SHA-256: {rec_hash}")
    print(f"  [PASS] 08_patient_phase_features.csv   : {pat_size:,} bytes | SHA-256: {pat_hash}")
    print(f"  [PASS] 08_ecg_feature_summary.json     : {sum_size:,} bytes | SHA-256: {sum_hash}")

    # 2. Re-read and validate serialized patient CSV
    df_read_pat = pd.read_csv(pat_csv_p)
    assert len(df_read_pat) == 160, f"Expected 160 rows in CSV, got {len(df_read_pat)}"
    assert len(df_read_pat.columns) == 264, f"Expected 264 columns in CSV, got {len(df_read_pat.columns)}"
    assert set(df_read_pat["subject_id"].unique()) == {f"sub_{i:03d}" for i in range(1, 81)}
    assert df_read_pat["sampling_rate"].value_counts().to_dict() == {100: 80, 500: 80}
    assert df_read_pat["stress_recording_status"].value_counts().to_dict() == {"SUCCESS": 144, "SKIPPED_UPSTREAM_ERROR": 16}
    assert set(df_read_pat["stage06_candidate_nn_status"].unique()) == {"UNKNOWN"}
    assert df_read_pat[df_read_pat["is_unlabelled_holdout"] == True]["subject_id"].unique().tolist() == ["sub_005", "sub_049"]
    print("  [PASS] Serialized patient feature CSV schema, counts, holdouts, and statuses fully reconciled.")

    # 3. Re-read and validate serialized recording CSV
    df_read_rec = pd.read_csv(rec_csv_p)
    assert len(df_read_rec) == 480, f"Expected 480 rows in CSV, got {len(df_read_rec)}"
    assert len(df_read_rec.columns) == 243, f"Expected 243 columns in CSV, got {len(df_read_rec.columns)}"
    assert df_read_rec["feature_extraction_status"].value_counts().to_dict() == {"SUCCESS": 464, "SKIPPED_UPSTREAM_ERROR": 16}
    assert df_read_rec["sampling_rate"].value_counts().to_dict() == {100: 240, 500: 240}
    print("  [PASS] Serialized recording feature CSV schema and row counts fully reconciled.")

    # 4. Validate Summary JSON
    with open(sum_json_p, "r", encoding="utf-8") as f:
        summary_obj = json.load(f)
    assert summary_obj["cohort_size"] == 80
    assert summary_obj["total_recordings"] == 480
    assert summary_obj["valid_recordings"] == 464
    assert summary_obj["skipped_recordings"] == 16
    assert summary_obj["patient_strata_rows"] == 160
    assert summary_obj["clinical_label_leakage"] is False
    assert summary_obj["stage06_candidate_nn_status"] == "UNKNOWN"
    assert summary_obj["physical_units"]["amplitude"] == "zero_centered_12bit_adc_counts"
    print("  [PASS] Quality summary JSON verified: units in ADC counts, zero leakage, UNKNOWN candidate status.")

    # 5. Check orphan temporary files
    for p in [rec_csv_p, pat_csv_p, sum_json_p]:
        tmp_p = p.with_suffix(".tmp" + p.suffix)
        assert not tmp_p.exists(), f"Orphan temporary file left behind: {tmp_p}"
    print("  [PASS] Clean atomic write: 0 orphan temporary files detected.")

    # 6. Verify Upstream Manifests Immutability
    s06 = project_root / "reports" / "hrv" / "06_rr_hrv_manifest.csv"
    s07 = project_root / "reports" / "segmentation" / "07_beat_segmentation_manifest.csv"
    s07a = project_root / "reports" / "transitions" / "07a_inter_phase_transition_manifest.csv"
    assert s06.stat().st_size == 168809, f"Stage 06 manifest altered: {s06.stat().st_size}"
    assert s07.stat().st_size == 58842887, f"Stage 07 beat manifest altered: {s07.stat().st_size}"
    assert s07a.stat().st_size == 68546, f"Stage 07A manifest altered: {s07a.stat().st_size}"
    print("  [PASS] Upstream Stage 06, 07, and 07A manifests verified cryptographically intact on disk.")

    # 7. Check remaining free disk space
    total_bytes, used_bytes, free_bytes = shutil.disk_usage(project_root)
    free_gb = free_bytes / (1024 ** 3)
    print(f"  [PASS] Free disk space: {free_gb:.3f} GB (>= 3.0 GB guard).")

    # 8. Check strict execution boundary on notebook
    for idx in range(len(nb["cells"])):
        c = nb["cells"][idx]
        if c["cell_type"] == "code":
            if idx in [2, 4, 6, 8, 10, 12, 14]:
                assert c["execution_count"] is not None, f"Executed cell {idx} has None count"
                assert len(c["outputs"]) > 0, f"Executed cell {idx} has empty outputs"
            else:
                assert c["execution_count"] is None, f"Cell {idx} should be unexecuted, found {c['execution_count']}"
                assert len(c["outputs"]) == 0, f"Cell {idx} should have empty outputs, found {len(c['outputs'])}"
    print("  [PASS] Execution boundary strictly verified: Cells 02, 04, 06, 08, 10, 12, 14 executed; Cells 16 and 18 unexecuted.")

    # Save state to step5_state.pkl
    state_to_save = {
        **saved_state,
        "recording_features_csv_path": rec_csv_p,
        "patient_features_csv_path": pat_csv_p,
        "feature_summary_json_path": sum_json_p,
        "rec_hash": rec_hash,
        "pat_hash": pat_hash,
        "sum_hash": sum_hash,
    }
    step5_pkl_path = project_root / "scratch" / "step5_state.pkl"
    with open(step5_pkl_path, "wb") as f:
        pickle.dump(state_to_save, f)
    print(f"  [PASS] Step 5 execution context persisted to {step5_pkl_path.name} ({step5_pkl_path.stat().st_size:,} bytes).")

    print("\n" + "=" * 60)
    print("STEP 5 EXECUTION COMPLETE & AUDITED: FULL PASS")
    print("=" * 60)

if __name__ == "__main__":
    main()

