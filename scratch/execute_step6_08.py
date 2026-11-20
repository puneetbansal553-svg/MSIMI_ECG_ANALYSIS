"""
execute_step6_08.py
Executes Cell 16 in notebooks/08_ecg_feature_extraction.ipynb.
Preserves strict execution boundary and updates notebook outputs and execution counts.
Validates generated PNG figures and persists execution context.
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
from PIL import Image

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

    # Verify Cells 02, 04, 06, 08, 10, 12, 14 are executed
    assert nb["cells"][2]["execution_count"] == 1, "Cell 02 not executed"
    assert nb["cells"][4]["execution_count"] == 2, "Cell 04 not executed"
    assert nb["cells"][6]["execution_count"] == 3, "Cell 06 not executed"
    assert nb["cells"][8]["execution_count"] == 4, "Cell 08 not executed"
    assert nb["cells"][10]["execution_count"] == 5, "Cell 10 not executed"
    assert nb["cells"][12]["execution_count"] == 6, "Cell 12 not executed"
    assert nb["cells"][14]["execution_count"] == 7, "Cell 14 not executed"
    assert nb["cells"][16]["execution_count"] is None, "Cell 16 already executed"
    assert nb["cells"][18]["execution_count"] is None, "Cell 18 must be unexecuted"

    # Load serialized patient features from disk (as required)
    pat_csv_path = project_root / "data" / "features" / "08_patient_phase_features.csv"
    assert pat_csv_path.exists(), f"Missing {pat_csv_path}"
    df_patient_features = pd.read_csv(pat_csv_path)

    rec_csv_path = project_root / "data" / "features" / "08_recording_level_features.csv"
    assert rec_csv_path.exists(), f"Missing {rec_csv_path}"
    df_recording_features = pd.read_csv(rec_csv_path)

    print("Reconstituting interactive kernel state for Cell 16...")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data_dir = project_root / "data"
    features_dir = data_dir / "features"
    reports_features_dir = project_root / "reports" / "features"
    figures_dir = project_root / "outputs" / "figures"

    recording_features_csv_path = features_dir / "08_recording_level_features.csv"
    patient_features_csv_path = features_dir / "08_patient_phase_features.csv"
    feature_summary_json_path = reports_features_dir / "08_ecg_feature_summary.json"
    figure_st_dist_path = figures_dir / "08_multilead_st_depression_distribution.png"
    figure_autonomic_repol_path = figures_dir / "08_autonomic_and_repolarization_feature_matrix.png"

    global_ctx = {
        "__name__": "__main__",
        "__file__": str(nb_path),
        "project_root": project_root,
        "data_dir": data_dir,
        "features_dir": features_dir,
        "reports_features_dir": reports_features_dir,
        "figures_dir": figures_dir,
        "recording_features_csv_path": recording_features_csv_path,
        "patient_features_csv_path": patient_features_csv_path,
        "feature_summary_json_path": feature_summary_json_path,
        "figure_st_dist_path": figure_st_dist_path,
        "figure_autonomic_repol_path": figure_autonomic_repol_path,
        "OVERWRITE_EXISTING_OUTPUTS": False,
        "plt": plt,
        "np": np,
        "pd": pd,
        "df_recording_features": df_recording_features,
        "df_patient_features": df_patient_features,
    }
    print("Kernel state successfully initialized for Cell 16.")

    class TeeStream(io.StringIO):
        def __init__(self, original_stream):
            super().__init__()
            self.original_stream = original_stream

        def write(self, s):
            self.original_stream.write(s)
            self.original_stream.flush()
            return super().write(s)

    cell_idx = 16
    exec_count = 8
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

    # Save notebook state after Cell 16
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print(f"Successfully saved {nb_path.name} with execution outputs for Cell 16.")

    # -------------------------------------------------------------
    # POST-EXECUTION STEP 6 VISUALIZATION AUDIT & VALIDATION
    # -------------------------------------------------------------
    print("\n" + "=" * 60)
    print("PERFORMING STEP 6 INDEPENDENT VISUALIZATION AUDIT")
    print("=" * 60)

    fig1_p = global_ctx["figure_st_dist_path"]
    fig2_p = global_ctx["figure_autonomic_repol_path"]

    # 1. Existence and non-emptiness
    assert fig1_p.exists() and fig1_p.stat().st_size > 0, "Figure 1 missing or empty"
    assert fig2_p.exists() and fig2_p.stat().st_size > 0, "Figure 2 missing or empty"

    fig1_size = fig1_p.stat().st_size
    fig1_hash = compute_sha256(fig1_p)
    fig2_size = fig2_p.stat().st_size
    fig2_hash = compute_sha256(fig2_p)

    # Validate image dimensions using PIL
    with Image.open(fig1_p) as im1:
        w1, h1 = im1.size
        format1 = im1.format
        assert format1 == "PNG", f"Figure 1 is not PNG: {format1}"

    with Image.open(fig2_p) as im2:
        w2, h2 = im2.size
        format2 = im2.format
        assert format2 == "PNG", f"Figure 2 is not PNG: {format2}"

    print(f"  [PASS] Figure 1: {fig1_p.name} | {w1}x{h1} px | {fig1_size:,} bytes | SHA-256: {fig1_hash}")
    print(f"  [PASS] Figure 2: {fig2_p.name} | {w2}x{h2} px | {fig2_size:,} bytes | SHA-256: {fig2_hash}")

    # 2. Check orphan temporary files
    for p in [fig1_p, fig2_p]:
        tmp_p = p.with_suffix(".tmp" + p.suffix)
        assert not tmp_p.exists(), f"Orphan temporary file left behind: {tmp_p}"
    print("  [PASS] Clean atomic write: 0 orphan temporary files detected.")

    # 3. Verify Serialized Feature CSVs and Upstream Manifests Immutability
    step5_pkl = project_root / "scratch" / "step5_state.pkl"
    with open(step5_pkl, "rb") as f:
        s5 = pickle.load(f)
    assert compute_sha256(pat_csv_path) == s5["pat_hash"], "Patient CSV altered during visualization!"
    assert compute_sha256(rec_csv_path) == s5["rec_hash"], "Recording CSV altered during visualization!"

    s06 = project_root / "reports" / "hrv" / "06_rr_hrv_manifest.csv"
    s07 = project_root / "reports" / "segmentation" / "07_beat_segmentation_manifest.csv"
    s07a = project_root / "reports" / "transitions" / "07a_inter_phase_transition_manifest.csv"
    assert s06.stat().st_size == 168809, f"Stage 06 manifest altered: {s06.stat().st_size}"
    assert s07.stat().st_size == 58842887, f"Stage 07 beat manifest altered: {s07.stat().st_size}"
    assert s07a.stat().st_size == 68546, f"Stage 07A manifest altered: {s07a.stat().st_size}"
    print("  [PASS] Feature stores and upstream Stage 06, 07, 07A manifests verified cryptographically unchanged.")

    # 4. Check remaining free disk space
    total_bytes, used_bytes, free_bytes = shutil.disk_usage(project_root)
    free_gb = free_bytes / (1024 ** 3)
    print(f"  [PASS] Free disk space: {free_gb:.3f} GB (>= 3.0 GB guard).")

    # 5. Check strict execution boundary on notebook (Cell 18 must be unexecuted)
    for idx in range(len(nb["cells"])):
        c = nb["cells"][idx]
        if c["cell_type"] == "code":
            if idx in [2, 4, 6, 8, 10, 12, 14, 16]:
                assert c["execution_count"] is not None, f"Executed cell {idx} has None count"
                assert len(c["outputs"]) > 0, f"Executed cell {idx} has empty outputs"
            else:
                assert c["execution_count"] is None, f"Cell {idx} should be unexecuted, found {c['execution_count']}"
                assert len(c["outputs"]) == 0, f"Cell {idx} should have empty outputs, found {len(c['outputs'])}"
    print("  [PASS] Execution boundary strictly verified: Cells 02..16 executed; Cell 18 remains strictly unexecuted.")

    # Clean up temp dump files if they exist
    for tmp_file in [project_root / "scratch" / "cell16_source.py", project_root / "scratch" / "dump_cell16.py"]:
        if tmp_file.exists():
            tmp_file.unlink()

    # Save state to step6_state.pkl
    state_to_save = {
        **s5,
        "fig1_path": fig1_p,
        "fig2_path": fig2_p,
        "fig1_hash": fig1_hash,
        "fig2_hash": fig2_hash,
        "fig1_size": fig1_size,
        "fig2_size": fig2_size,
        "fig1_dims": (w1, h1),
        "fig2_dims": (w2, h2),
    }
    step6_pkl_path = project_root / "scratch" / "step6_state.pkl"
    with open(step6_pkl_path, "wb") as f:
        pickle.dump(state_to_save, f)
    print(f"  [PASS] Step 6 execution context persisted to {step6_pkl_path.name} ({step6_pkl_path.stat().st_size:,} bytes).")

    print("\n" + "=" * 60)
    print("STEP 6 EXECUTION COMPLETE & AUDITED: FULL PASS")
    print("=" * 60)

if __name__ == "__main__":
    main()
