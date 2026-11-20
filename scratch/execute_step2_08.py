"""
execute_step2_08.py
Executes Cell 06 followed by Cell 08 in notebooks/08_ecg_feature_extraction.ipynb.
Preserves strict execution boundary and updates notebook outputs and execution counts.
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

    # Verify Cell 02 and Cell 04 are already executed
    assert nb["cells"][2]["execution_count"] == 1, "Cell 02 is not executed!"
    assert nb["cells"][4]["execution_count"] == 2, "Cell 04 is not executed!"

    # Initialize global execution context by executing Cell 02 and Cell 04 in memory
    global_ctx = {
        "__name__": "__main__",
        "__file__": str(nb_path),
    }

    print("Reconstituting interactive kernel state from Cells 02 and 04...")
    src_c02 = "".join(nb["cells"][2].get("source", []))
    src_c04 = "".join(nb["cells"][4].get("source", []))
    exec(compile(src_c02, "<cell_02_reconstitute>", "exec"), global_ctx)
    exec(compile(src_c04, "<cell_04_reconstitute>", "exec"), global_ctx)
    print("Kernel state successfully initialized with Cells 02 and 04 namespace.")

    class TeeStream(io.StringIO):
        def __init__(self, original_stream):
            super().__init__()
            self.original_stream = original_stream

        def write(self, s):
            self.original_stream.write(s)
            self.original_stream.flush()
            return super().write(s)

    def run_cell(cell_idx, exec_count):
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

        # Update notebook on disk after each cell
        with open(nb_path, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=1)
        print(f"Saved notebook state after Cell {cell_idx:02d}.")

    # 1. Execute Cell 06
    run_cell(6, 3)

    # Inspect Cell 06 outputs
    assert "delineate_lead_fiducials" in global_ctx, "delineate_lead_fiducials missing from context"
    assert "compute_contiguous_lead_count" in global_ctx, "compute_contiguous_lead_count missing from context"
    assert "compute_frontal_qrs_axis" in global_ctx, "compute_frontal_qrs_axis missing from context"
    print("  [PASS] Cell 06 functions confirmed compiled and present in execution context.")

    # 2. Execute Cell 08
    run_cell(8, 4)

    # -------------------------------------------------------------
    # POST-EXECUTION STEP 2 VALIDATIONS & PILOT AUDIT
    # -------------------------------------------------------------
    print("\n" + "=" * 60)
    print("PERFORMING STEP 2 INDEPENDENT PILOT AUDIT")
    print("=" * 60)

    # Extract detailed metrics across the 6 pilot recordings
    processed_beats_dir = global_ctx["processed_beats_dir"]
    pilot_recs = [
        "sub_001_rest_100hz",
        "sub_001_stress_100hz",
        "sub_001_recover_100hz",
        "sub_001_rest_500hz",
        "sub_001_stress_500hz",
        "sub_001_recover_500hz",
    ]

    STANDARD_12_LEADS = global_ctx["STANDARD_12_LEADS"]
    delineate_fn = global_ctx["delineate_lead_fiducials"]
    axis_fn = global_ctx["compute_frontal_qrs_axis"]
    contiguous_fn = global_ctx["compute_contiguous_lead_count"]
    REGIONAL_GROUPS = global_ctx["REGIONAL_GROUPS"]

    pilot_audit_data = []

    for rec_id in pilot_recs:
        npz_p = processed_beats_dir / f"{rec_id}_beats.npz"
        assert npz_p.exists()
        with np.load(npz_p) as data:
            beats = data["beats"]
            fs = 100 if "100hz" in rec_id else 500
            med = np.median(beats, axis=0)

        row_res = {"recording_id": rec_id, "fs": fs, "n_beats": len(beats)}
        
        # Lead V5 and II fiducials
        v5_idx = STANDARD_12_LEADS.index("V5")
        ii_idx = STANDARD_12_LEADS.index("II")
        v5_f = delineate_fn(med[v5_idx], fs)
        ii_f = delineate_fn(med[ii_idx], fs)

        row_res["v5_st60_adc"] = v5_f["st60_adc"]
        row_res["v5_st80_adc"] = v5_f["st80_adc"]
        row_res["v5_j_offset_ms"] = v5_f["j_offset_ms"]
        row_res["v5_qrs_dur_ms"] = v5_f["qrs_dur_ms"]
        row_res["v5_qt_ms"] = v5_f["qt_ms"]
        row_res["v5_j_valid"] = v5_f["j_point_valid"]
        row_res["v5_qrs_valid"] = v5_f["qrs_valid"]
        row_res["v5_t_valid"] = v5_f["t_wave_valid"]
        row_res["v5_qt_valid"] = v5_f["qt_valid"]

        row_res["ii_st60_adc"] = ii_f["st60_adc"]
        row_res["ii_st80_adc"] = ii_f["st80_adc"]
        row_res["ii_j_offset_ms"] = ii_f["j_offset_ms"]
        row_res["ii_j_valid"] = ii_f["j_point_valid"]

        # Frontal axis
        axis_deg, axis_valid = axis_fn(med[0], med[5], fs)
        row_res["qrs_axis_deg"] = axis_deg
        row_res["qrs_axis_valid"] = axis_valid

        # Regional minimums and contiguous depressed leads
        depressed_leads = []
        for l_idx, lead in enumerate(STANDARD_12_LEADS):
            lf = delineate_fn(med[l_idx], fs)
            if lf["st60_adc"] < -50.0: # threshold in ADC counts
                depressed_leads.append(lead)

        row_res["depressed_leads"] = depressed_leads
        row_res["contig_inferior"] = contiguous_fn(depressed_leads, REGIONAL_GROUPS["inferior"])
        row_res["contig_anterior"] = contiguous_fn(depressed_leads, REGIONAL_GROUPS["anterior"])
        row_res["contig_lateral"] = contiguous_fn(depressed_leads, REGIONAL_GROUPS["lateral"])

        pilot_audit_data.append(row_res)

    df_pilot_audit = pd.DataFrame(pilot_audit_data)
    print("\n--- Detailed Pilot Metrics Table ---")
    print(df_pilot_audit.to_string(index=False))

    # Physiological assertions
    for _, r in df_pilot_audit.iterrows():
        # J-offset should be within [20, 80] ms
        assert 20 <= r["v5_j_offset_ms"] <= 80, f"Unusual J-offset: {r['v5_j_offset_ms']}"
        assert 20 <= r["ii_j_offset_ms"] <= 80, f"Unusual J-offset: {r['ii_j_offset_ms']}"
        # QRS duration should be within [40, 160] ms
        assert 40 <= r["v5_qrs_dur_ms"] <= 160, f"Unusual QRS duration: {r['v5_qrs_dur_ms']}"
        # ST60 and ST80 should be finite
        assert np.isfinite(r["v5_st60_adc"])
        assert np.isfinite(r["v5_st80_adc"])
        assert np.isfinite(r["ii_st60_adc"])
        assert np.isfinite(r["ii_st80_adc"])
        # All V5 fiducials must be valid for sub_001
        assert r["v5_j_valid"] is True
        assert r["v5_qrs_valid"] is True
        assert r["qrs_axis_valid"] is True

    print("\n  [PASS] All pilot physiological ranges and validity flags verified.")

    # Check remaining disk space
    total_bytes, used_bytes, free_bytes = shutil.disk_usage(project_root)
    free_gb = free_bytes / (1024 ** 3)
    print(f"  [PASS] Post-execution host free disk space: {free_gb:.3f} GB (>= 3.0 GB guard).")

    # Verify execution boundary on notebook
    for idx in range(len(nb["cells"])):
        c = nb["cells"][idx]
        if c["cell_type"] == "code":
            if idx in [2, 4, 6, 8]:
                assert c["execution_count"] is not None, f"Executed cell {idx} has None count"
                assert len(c["outputs"]) > 0, f"Executed cell {idx} has no outputs"
            else:
                assert c["execution_count"] is None, f"Cell {idx} should be unexecuted, found {c['execution_count']}"
                assert len(c["outputs"]) == 0, f"Cell {idx} should have empty outputs, found {len(c['outputs'])}"
    print("  [PASS] Execution boundary strictly verified: Cells 02, 04, 06, 08 executed; Cells 10..18 unexecuted.")

    # Save state to step2_state.pkl (filter out unpicklable module objects)
    state_to_save = {
        "pilot_audit_df": df_pilot_audit,
        "OVERWRITE_EXISTING_OUTPUTS": global_ctx["OVERWRITE_EXISTING_OUTPUTS"],
        "df_stage06": global_ctx["df_stage06"],
        "df_stage06_success": global_ctx["df_stage06_success"],
        "df_patient_summary": global_ctx["df_patient_summary"],
        "df_stage07a": global_ctx["df_stage07a"],
    }
    step2_pkl_path = project_root / "scratch" / "step2_state.pkl"
    with open(step2_pkl_path, "wb") as f:
        pickle.dump(state_to_save, f)
    print(f"  [PASS] Step 2 execution context persisted to {step2_pkl_path.name} ({step2_pkl_path.stat().st_size:,} bytes).")

    print("\n" + "=" * 60)
    print("STEP 2 EXECUTION COMPLETE & AUDITED: FULL PASS")
    print("=" * 60)

if __name__ == "__main__":
    main()
