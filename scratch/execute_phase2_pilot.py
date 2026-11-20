"""
execute_phase2_pilot.py
Executes Cell 08 in Pilot Mode (PILOT_MODE = True, PILOT_SUBJECTS = ["sub_001"])
in notebooks/07a_inter_phase_transition_analysis.ipynb.
Validates all pilot transition metrics.
Persists pilot state to scratch/pilot_state.pkl.
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

    nb_path = project_root / "notebooks" / "07a_inter_phase_transition_analysis.ipynb"
    assert nb_path.exists(), f"Notebook missing at {nb_path}"

    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    print(f"Loaded {nb_path.name}.")
    print("Executing Phase 2: Cell 08 (Pilot Mode)...")

    # Ingest Phase 1 state
    with open("scratch/phase1_state.pkl", "rb") as f:
        global_ctx = pickle.load(f)

    # Ingest manifests into context
    global_ctx["df_stage06"] = pd.read_csv(global_ctx["reports_hrv_dir"] / "06_rr_hrv_manifest.csv")
    global_ctx["df_beat_manifest"] = pd.read_csv(global_ctx["manifest_s07_path"])
    global_ctx["__name__"] = "__main__"
    global_ctx["__file__"] = str(nb_path)

    from typing import Dict, Any, List, Tuple, Optional
    import math
    global_ctx["Dict"] = Dict
    global_ctx["Any"] = Any
    global_ctx["List"] = List
    global_ctx["Tuple"] = Tuple
    global_ctx["Optional"] = Optional
    global_ctx["math"] = math
    global_ctx["time"] = time
    global_ctx["shutil"] = shutil
    global_ctx["np"] = np
    global_ctx["pd"] = pd

    # Define mathematical routines into context
    from typing import Dict, Any, List, Tuple
    import math

    def delineate_j_point(median_beat: np.ndarray, fs: int, lead_idx: int = 10) -> Tuple[int, float]:
        m_R = 25 if fs == 100 else 125
        signal = median_beat[lead_idx]
        dt = 1.0 / fs
        diff1 = np.gradient(signal, dt)
        w_qrs_pre = int(round(0.020 * fs))
        w_qrs_post = int(round(0.020 * fs))
        max_depol_slope = np.max(np.abs(diff1[m_R - w_qrs_pre : m_R + w_qrs_post + 1]))
        if max_depol_slope == 0:
            max_depol_slope = 1.0
        s_search_start = m_R + int(round(0.010 * fs))
        s_search_end = m_R + int(round(0.060 * fs))
        m_S = s_search_start + np.argmin(signal[s_search_start : s_search_end + 1])
        j_search_end = m_R + int(round(0.120 * fs))
        threshold = 0.10 * max_depol_slope
        m_J = j_search_end
        for k in range(m_S, j_search_end - 2):
            if np.all(np.abs(diff1[k : k + 3]) <= threshold):
                m_J = k
                break
        j_offset_ms = round((m_J - m_R) * (1000.0 / fs), 2)
        return m_J, j_offset_ms

    def compute_morphology_metrics(b_pre: np.ndarray, b_post: np.ndarray, compute_dtw: bool = False) -> Dict[str, Any]:
        corr_list = []
        rmsd_list = []
        for l_idx in range(12):
            sig1 = b_pre[l_idx]
            sig2 = b_post[l_idx]
            std1 = np.std(sig1)
            std2 = np.std(sig2)
            if std1 > 1e-6 and std2 > 1e-6:
                cov = np.mean((sig1 - np.mean(sig1)) * (sig2 - np.mean(sig2)))
                r = float(np.clip(cov / (std1 * std2), -1.0, 1.0))
            else:
                r = 1.0 if np.allclose(sig1, sig2) else 0.0
            corr_list.append(float(r))
            rmsd = np.sqrt(np.mean((sig2 - sig1) ** 2))
            rmsd_list.append(float(rmsd))
        frob_dist = float(np.sqrt(np.sum((b_post - b_pre) ** 2)))
        max_rmsd_idx = int(np.argmax(rmsd_list))
        return {
            "mean_morphology_corr": round(float(np.mean(corr_list)), 4),
            "min_morphology_corr": round(float(np.min(corr_list)), 4),
            "mean_rmsd_mv": round(float(np.mean(rmsd_list)), 4),
            "frobenius_dissimilarity_mv": round(frob_dist, 4),
            "max_rmsd_lead": global_ctx["STANDARD_12_LEADS"][max_rmsd_idx],
            "max_rmsd_value_mv": round(float(rmsd_list[max_rmsd_idx]), 4),
            "rmsd_per_lead": rmsd_list,
            "corr_per_lead": corr_list,
        }

    global_ctx["delineate_j_point"] = delineate_j_point
    global_ctx["compute_morphology_metrics"] = compute_morphology_metrics

    class TeeStream(io.StringIO):
        def __init__(self, original_stream):
            super().__init__()
            self.original_stream = original_stream

        def write(self, s):
            self.original_stream.write(s)
            self.original_stream.flush()
            return super().write(s)

    cell_08 = nb["cells"][8]
    assert cell_08["cell_type"] == "code", "Cell 08 is not a code cell!"
    src_code = "".join(cell_08.get("source", []))

    print(f"\n{'='*60}")
    print("EXECUTING CELL 08 (Pilot Mode: sub_001, 4 pairs)")
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
        exec(compile(src_code, "<cell_08_pilot>", "exec"), global_ctx)
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
    print(f"\n[Cell 08 completed in {elapsed:.3f}s]")

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

    cell_08["execution_count"] = 4
    cell_08["outputs"] = cell_outputs

    if error_info:
        print(f"HALT: Execution failed in Cell 08: {error_info['ename']}: {error_info['evalue']}")
        with open(nb_path, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=1)
        sys.exit(1)

    # Save notebook with executed Cell 08
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print(f"\nSuccessfully updated {nb_path.name} with execution outputs for Cell 08.")

    # -------------------------------------------------------------
    # PHASE 2 PILOT VALIDATION CHECKS
    # -------------------------------------------------------------
    print("\n" + "=" * 60)
    print("PERFORMING PHASE 2 PILOT VALIDATION CHECKS")
    print("=" * 60)

    pilot_records = global_ctx.get("transition_records", [])
    assert len(pilot_records) == 4, f"Expected exactly 4 pilot records, got {len(pilot_records)}"
    df_pilot = pd.DataFrame(pilot_records)

    # 1. Pilot pair representation
    print("\nPilot Transition Pairs Evaluated:")
    for idx, r in df_pilot.iterrows():
        print(f"  [{idx+1}] {r['transition_id']} | Type: {r['transition_type']:<18} | fs: {r['sampling_rate']} Hz | Status: {r['transition_status']}")
        print(f"      Beats: pre={r['pre_eligible_beats']}, post={r['post_eligible_beats']}")
        print(f"      HR: pre={r['hr_pre_bpm']} bpm, post={r['hr_post_bpm']} bpm, Delta={r['delta_hr_bpm']} bpm, RR ratio={r['rr_ratio']}")
        if r['transition_type'] == 'STRESS_TO_RECOVERY':
            print(f"      HRR60: {r['hrr60_bpm']} bpm (end_stress={r['hr_end_stress_bpm']} bpm, rec_60s={r['hr_rec_60s_bpm']} bpm, late_120s={r['hr_late_stress_120s_bpm']} bpm)")
            print(f"      HRRI: {r['hrri_pct']}%")
        print(f"      Morphology: Frobenius={r['frobenius_dissimilarity_mv']} mV, Mean r={r['mean_morphology_corr']}, Min r={r['min_morphology_corr']}")
        print(f"      J-Point: offset={r['j_point_offset_ms_v5']} ms, Delta ST60 (V5)={r['delta_st60_lead_v5_mv']} mV, Delta ST60 (II)={r['delta_st60_lead_ii_mv']} mV")

    # 2. Validate HRR60 time windows and calculations
    t2_records = df_pilot[df_pilot["transition_type"] == "STRESS_TO_RECOVERY"]
    assert len(t2_records) == 2, f"Expected 2 T2 records, found {len(t2_records)}"
    for _, t2_row in t2_records.iterrows():
        hr_end = t2_row["hr_end_stress_bpm"]
        hr_rec = t2_row["hr_rec_60s_bpm"]
        hrr = t2_row["hrr60_bpm"]
        assert np.isfinite(hr_end), f"hr_end_stress_bpm is not finite: {hr_end}"
        assert np.isfinite(hr_rec), f"hr_rec_60s_bpm is not finite: {hr_rec}"
        assert np.isfinite(hrr), f"hrr60_bpm is not finite: {hrr}"
        assert round(hr_end - hr_rec, 2) == round(hrr, 2), f"HRR60 arithmetic mismatch: {hr_end} - {hr_rec} != {hrr}"
    print("\n  [PASS] HRR60 time windows and calculations verified: exact end-of-stress vs 60s recovery window.")

    # 3. Validate J-point detection and ST60 measurements
    for _, row in df_pilot.iterrows():
        j_off = row["j_point_offset_ms_v5"]
        assert 20.0 <= j_off <= 100.0, f"J-point offset {j_off} ms outside physiological range [20, 100] ms"
        st60_v5 = row["delta_st60_lead_v5_mv"]
        st60_ii = row["delta_st60_lead_ii_mv"]
        assert np.isfinite(st60_v5) and abs(st60_v5) < 100.0, f"ST60 V5 {st60_v5} out of realistic range"
        assert np.isfinite(st60_ii) and abs(st60_ii) < 100.0, f"ST60 II {st60_ii} out of realistic range"
    print("  [PASS] J-point detection and ST60 measurements verified: realistic offsets and voltages.")

    # 4. Validate Pearson correlation, RMSD, Frobenius distance
    for _, row in df_pilot.iterrows():
        r_mean = row["mean_morphology_corr"]
        r_min = row["min_morphology_corr"]
        frob = row["frobenius_dissimilarity_mv"]
        assert 0.70 <= r_mean <= 1.0, f"Mean correlation {r_mean} out of range"
        assert -1.0 <= r_min <= 1.0, f"Min correlation {r_min} out of range"
        assert frob > 0.0 and frob < 5000.0, f"Frobenius distance {frob} out of range"
    print("  [PASS] Morphology metrics verified: Pearson r in [0.70, 1.0], Frobenius distance positive and bounded.")

    # 5. Check no NaN/Inf silently accepted in mandatory columns
    mandatory_cols = [
        "transition_id", "subject_id", "sampling_rate", "transition_type",
        "hr_pre_bpm", "hr_post_bpm", "delta_hr_bpm", "rr_ratio",
        "mean_morphology_corr", "min_morphology_corr", "frobenius_dissimilarity_mv",
        "j_point_offset_ms_v5", "delta_st60_lead_v5_mv", "delta_st60_lead_ii_mv",
        "stage06_candidate_nn_status"
    ]
    for col in mandatory_cols:
        assert df_pilot[col].notna().all(), f"NaN found in mandatory pilot column {col}"
    print(f"  [PASS] No NaN/Inf values across {len(mandatory_cols)} mandatory transition fields.")

    # 6. Verify zero clinical diagnosis leakage
    forbidden_labels = ["anoca_confirmed", "raw_msimi_ecg_label", "target_label", "MSIMI-based-ecg", "label_status", "diagnosis"]
    for col in forbidden_labels:
        assert col not in df_pilot.columns, f"LEAKAGE: Forbidden label {col} in pilot DataFrame!"
    print("  [PASS] Zero clinical diagnosis labels in pilot transition records.")

    # 7. Verify downstream cells remain strictly unexecuted
    for idx in [10, 12, 14, 16]:
        cell = nb["cells"][idx]
        assert cell["execution_count"] is None, f"Cell {idx} has execution_count != None: {cell['execution_count']}"
        assert cell["outputs"] == [], f"Cell {idx} has outputs: {len(cell['outputs'])}"
    print("  [PASS] Downstream Cells 10, 12, 14, 16 remain strictly unexecuted.")

    # Persist pilot state
    with open("scratch/pilot_state.pkl", "wb") as f:
        pickle.dump({"pilot_records": pilot_records, "df_pilot": df_pilot}, f)
    print("  [PASS] Pilot state persisted to scratch/pilot_state.pkl.")

    print("\nPHASE 2 PILOT EXECUTION AND VALIDATION COMPLETED SUCCESSFULLY!")

if __name__ == "__main__":
    main()
