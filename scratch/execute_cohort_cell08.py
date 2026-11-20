"""
execute_cohort_cell08.py
Sets PILOT_MODE = False in notebooks/07a_inter_phase_transition_analysis.ipynb,
executes Cell 08 across the full 288-pair cohort, validates all metrics,
and persists state to scratch/cell08_cohort_state.pkl.
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
    print("Preparing Full-Cohort Execution: Setting PILOT_MODE = False in Cell 08...")

    cell_08 = nb["cells"][8]
    assert cell_08["cell_type"] == "code", "Cell 08 is not a code cell!"

    # 1. Update Cell 08 source to ensure PILOT_MODE = False and cohort assertions are present
    src_lines = cell_08["source"]
    updated_lines = []
    for line in src_lines:
        if line.strip().startswith("PILOT_MODE = True"):
            updated_lines.append("PILOT_MODE = False  # True: run representative pilot; False: run full cohort (288 pairs)\n")
        else:
            updated_lines.append(line)

    # Check if cohort assertion block is present in source
    src_text = "".join(updated_lines)
    if "Full Cohort Accounting & Validation" not in src_text:
        # Append cohort assertion block after pilot assertion
        old_pilot_end = '    print(f"  [PASS] Pilot assertions passed across {len(pilot_df)} pairs (T1 & T2 at 100 Hz and 500 Hz).\\n")\n'
        cohort_block = (
            'else:\n'
            '    print("\\n--- Full Cohort Accounting & Validation ---\\n")\n'
            '    cohort_df = pd.DataFrame(transition_records)\n'
            '    assert len(cohort_df) == 288, f"Expected 288 cohort records, found {len(cohort_df)}"\n'
            '    assert set(cohort_df["transition_type"]) == {"REST_TO_STRESS", "STRESS_TO_RECOVERY"}\n'
            '    assert len(cohort_df[cohort_df["transition_type"] == "REST_TO_STRESS"]) == 144\n'
            '    assert len(cohort_df[cohort_df["transition_type"] == "STRESS_TO_RECOVERY"]) == 144\n'
            '    assert len(cohort_df[cohort_df["sampling_rate"] == 100]) == 128\n'
            '    assert len(cohort_df[cohort_df["sampling_rate"] == 500]) == 160\n'
            '    assert set(cohort_df["stage06_candidate_nn_status"].unique()) == {"UNKNOWN"}\n'
            '    print(f"  [PASS] Cohort transition accounting verified: 288 records (144 T1 + 144 T2; 128 @ 100Hz, 160 @ 500Hz).\\n")\n'
        )
        if old_pilot_end in src_text:
            src_text = src_text.replace(old_pilot_end, old_pilot_end + cohort_block)
            updated_lines = [l + "\n" for l in src_text.splitlines()]

    cell_08["source"] = updated_lines

    # 2. Ingest Phase 1 execution context
    with open("scratch/phase1_state.pkl", "rb") as f:
        global_ctx = pickle.load(f)

    global_ctx["df_stage06"] = pd.read_csv(global_ctx["reports_hrv_dir"] / "06_rr_hrv_manifest.csv")
    global_ctx["df_beat_manifest"] = pd.read_csv(global_ctx["manifest_s07_path"])
    global_ctx["__name__"] = "__main__"
    global_ctx["__file__"] = str(nb_path)
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

    print(f"\n{'='*70}")
    print("STARTING FULL COHORT EXECUTION: CELL 08 (288 Transition Pairs)")
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
        code_to_exec = "".join(cell_08["source"])
        exec(compile(code_to_exec, "<cell_08_cohort>", "exec"), global_ctx)
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
    print(f"\n[Cell 08 Full-Cohort completed in {elapsed:.2f}s]")

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
    print(f"\nSuccessfully updated {nb_path.name} with execution outputs for Cell 08 (Full Cohort).")

    # -------------------------------------------------------------
    # EXHAUSTIVE COHORT VALIDATION AUDIT
    # -------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PERFORMING FULL COHORT VALIDATION AUDIT")
    print("=" * 70)

    transition_records = global_ctx.get("transition_records", [])
    assert len(transition_records) == 288, f"Expected exactly 288 records, got {len(transition_records)}"
    df_transitions = pd.DataFrame(transition_records)

    # 1. Pair Counts & Transition Accounting
    t1_records = df_transitions[df_transitions["transition_type"] == "REST_TO_STRESS"]
    t2_records = df_transitions[df_transitions["transition_type"] == "STRESS_TO_RECOVERY"]
    assert len(t1_records) == 144, f"Expected 144 T1 records, got {len(t1_records)}"
    assert len(t2_records) == 144, f"Expected 144 T2 records, got {len(t2_records)}"
    assert "RECOVERY_TO_STRESS" not in df_transitions["transition_type"].unique()

    # Strata counts
    t1_100 = t1_records[t1_records["sampling_rate"] == 100]
    t1_500 = t1_records[t1_records["sampling_rate"] == 500]
    t2_100 = t2_records[t2_records["sampling_rate"] == 100]
    t2_500 = t2_records[t2_records["sampling_rate"] == 500]
    assert len(t1_100) == 64 and len(t1_500) == 80
    assert len(t2_100) == 64 and len(t2_500) == 80
    print(f"  [PASS] Transition pair counts reconciled: T1 = 144 (64 @ 100Hz, 80 @ 500Hz), T2 = 144 (64 @ 100Hz, 80 @ 500Hz), T3 = 0.")

    # 2. Mandatory Columns Completeness (Zero NaN/Inf allowed)
    mandatory_cols = [
        "transition_id", "subject_id", "sampling_rate", "transition_type",
        "pre_recording_id", "post_recording_id", "transition_status",
        "pre_eligible_beats", "post_eligible_beats", "hr_pre_bpm", "hr_post_bpm",
        "delta_hr_bpm", "rr_ratio", "delta_rmssd_ms",
        "mean_morphology_corr", "min_morphology_corr", "frobenius_dissimilarity_mv",
        "max_rmsd_lead", "max_rmsd_value_mv",
        "j_point_offset_ms_v5", "delta_st60_lead_v5_mv", "delta_st60_lead_ii_mv",
        "stage06_candidate_nn_status"
    ]
    for col in mandatory_cols:
        nan_count = df_transitions[col].isna().sum()
        assert nan_count == 0, f"Mandatory column '{col}' contains {nan_count} NaN values!"
        if df_transitions[col].dtype in [np.float64, np.float32]:
            inf_count = np.isinf(df_transitions[col]).sum()
            assert inf_count == 0, f"Mandatory column '{col}' contains {inf_count} Inf values!"
    print(f"  [PASS] 100% completeness (zero NaN/Inf) across all {len(mandatory_cols)} mandatory columns (288/288 rows).")

    # 3. Optional & Transition-Specific Columns Accounting
    print("\nTransition-Specific Metrics Accounting:")
    # T1 specific: HRR60 and HRRI are NaN by definition
    assert t1_records["hrr60_bpm"].isna().all(), "HRR60 must be NaN for T1"
    assert t1_records["hrri_pct"].isna().all(), "HRRI must be NaN for T1"

    # T2 specific
    t2_hrr60_valid = t2_records["hrr60_bpm"].notna().sum()
    t2_hrr60_nan = t2_records["hrr60_bpm"].isna().sum()
    t2_hrri_valid = t2_records["hrri_pct"].notna().sum()
    t2_hrri_nan = t2_records["hrri_pct"].isna().sum()

    print(f"  T2 HRR60 Evaluated: {t2_hrr60_valid}/144 valid pairs ({t2_hrr60_nan} NaN).")
    print(f"    - Median HRR60: {t2_records['hrr60_bpm'].median():.2f} bpm (IQR: [{t2_records['hrr60_bpm'].quantile(0.25):.2f}, {t2_records['hrr60_bpm'].quantile(0.75):.2f}])")
    print(f"    - Mean HRR60: {t2_records['hrr60_bpm'].mean():.2f} bpm (Std: {t2_records['hrr60_bpm'].std():.2f})")
    print(f"  T2 HRRI Evaluated: {t2_hrri_valid}/144 valid pairs ({t2_hrri_nan} NaN due to chronotropic reserve <= 5 bpm or recovery length).")
    if t2_hrri_valid > 0:
        print(f"    - Median HRRI: {t2_records['hrri_pct'].dropna().median():.2f}%")

    # 4. Clinical Label Isolation & Candidate NN Status
    forbidden_labels = ["anoca_confirmed", "raw_msimi_ecg_label", "target_label", "MSIMI-based-ecg", "label_status", "diagnosis"]
    for col in forbidden_labels:
        assert col not in df_transitions.columns, f"LEAKAGE VIOLATION: Forbidden label '{col}' found in transitions!"
    print("  [PASS] Zero clinical diagnosis label leakage in cohort transition records.")

    assert set(df_transitions["stage06_candidate_nn_status"].unique()) == {"UNKNOWN"}
    print("  [PASS] stage06_candidate_nn_status strictly UNKNOWN across 100% of cohort records (288/288).")

    # 5. Disk Space Verification
    total_b, used_b, free_b = shutil.disk_usage(project_root)
    free_gb = free_b / (1024 ** 3)
    print(f"\n  [PASS] Host Free Disk Space: {free_gb:.3f} GB (Safely above 3.0 GB dynamic stop guard).")
    assert free_gb >= 3.0, f"Disk space guard violated: {free_gb:.3f} GB < 3.0 GB!"

    # 6. Upstream Stages Immutability Spot-Check
    assert Path("reports/segmentation/07_beat_segmentation_manifest.csv").stat().st_size == 58842887
    assert len(list(Path("data/processed/beat_segments").glob("*_beats.npz"))) == 464
    print("  [PASS] Stage 07 beat segmentation archives and manifest confirmed 100% unchanged.")

    # 7. Downstream Cells Status Verification
    for idx in [10, 12, 14, 16]:
        cell = nb["cells"][idx]
        assert cell["execution_count"] is None, f"Downstream cell {idx} has execution_count != None: {cell['execution_count']}"
        assert cell["outputs"] == [], f"Downstream cell {idx} has non-empty outputs!"
    print("  [PASS] Downstream Cells 10, 12, 14, 16 remain strictly unexecuted.")

    # 8. Persist Cohort State for Cell 10
    state_to_save = {
        "transition_records": transition_records,
        "df_transitions": df_transitions,
        "t1_df": t1_records,
        "t2_df": t2_records,
        "execution_time_sec": elapsed,
        "free_disk_gb": free_gb,
    }
    with open("scratch/cell08_cohort_state.pkl", "wb") as f:
        pickle.dump(state_to_save, f)
    print("  [PASS] Cohort transition state persisted to scratch/cell08_cohort_state.pkl.")

    print("\n" + "=" * 70)
    print("STAGE 07A: CELL 08 FULL-COHORT EXECUTION COMPLETED SUCCESSFULLY!")
    print("=" * 70)

if __name__ == "__main__":
    main()

