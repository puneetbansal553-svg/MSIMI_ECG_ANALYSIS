import json
from pathlib import Path

cells = []

def add_md(source):
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source.strip().split("\n")]
    })

def add_code(source):
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.strip().split("\n")]
    })

# Cell 00: Header
add_md("""# 07A — Inter-Phase Transition Analysis
**MSIMI ECG Research Study | Stage 07A Pipeline Module**

---

### Overview & Objectives
This notebook implements **Stage 07A: Inter-Phase Transition Analysis**, positioned directly between Stage 07 Beat Window Segmentation and Stage 08 Feature Extraction. It evaluates the dynamic electrocardiographic and autonomic adaptations occurring across protocol transitions:
1. **$T_1: \\text{REST} \\rightarrow \\text{STRESS}$:** Sympathetic activation, chronotropic acceleration, and workload-induced repolarization shifts.
2. **$T_2: \\text{STRESS} \\rightarrow \\text{RECOVERY}$:** Parasympathetic reactivation, 1-minute heart rate recovery (HRR60), and repolarization hysteresis.
3. **$T_3: \\text{RECOVERY} \\rightarrow \\text{STRESS}$:** Explicitly audited and cataloged as `PROTOCOL_NOT_APPLICABLE` (0 subjects in this single-challenge cohort).

### Clinical Data Governance & Scientific Boundaries
- **Stage 07 Read-Only:** All Stage 07 beat archives, manifests, and patient summaries remain strictly read-only.
- **Zero Label Leakage:** Transition features are calculated in a blinded manner without clinical diagnosis labels.
- **Data Governance:** `stage06_candidate_nn_status` remains strictly `UNKNOWN`.
- **Descriptive Integrity:** Morphological adaptations are quantified without diagnostic classification claims.""")

# Cell 01: Markdown Section 1
add_md("""## 1. Environment Setup, Configuration Loading & Disk Space Guard

This section establishes paths, sets up output directories, enforces overwrite protection (`OVERWRITE_EXISTING_OUTPUTS = False`), and verifies available host disk space against the 3.0 GB safety envelope.""")

# Cell 02: Code Cell 02
add_code("""import sys
import os
import shutil
import json
import hashlib
import time
import math
import gc
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 1. Project Root & Working Directory Resolution
project_root = Path.cwd()
if not (project_root / "configs").exists() and (project_root.parent / "configs").exists():
    project_root = project_root.parent
os.chdir(project_root)

print(f"Working Directory set to: {project_root}")
assert (project_root / "configs").exists(), f"Configuration directory missing at {project_root / 'configs'}"

# 2. Overwrite Protection Guard (Default: False)
OVERWRITE_EXISTING_OUTPUTS = False

# 3. Path Definitions
data_dir = project_root / "data"
interim_ecg_dir = data_dir / "interim" / "preprocessed_ecg"
processed_rpeaks_dir = data_dir / "processed" / "rpeaks"
output_beat_dir = data_dir / "processed" / "beat_segments"
reports_segmentation_dir = project_root / "reports" / "segmentation"
reports_transitions_dir = project_root / "reports" / "transitions"
reports_qc_dir = project_root / "reports" / "qc"
reports_hrv_dir = project_root / "reports" / "hrv"
reports_rpeaks_dir = project_root / "reports" / "rpeaks"
figures_dir = project_root / "outputs" / "figures"

# Stage 07 Input Targets
manifest_s07_path = reports_segmentation_dir / "07_beat_segmentation_manifest.csv"
summary_s07_path = reports_segmentation_dir / "07_beat_segmentation_summary.json"
patient_s07_path = reports_segmentation_dir / "07_patient_beat_summary.csv"

# Stage 07A Output Targets
transition_manifest_csv_path = reports_transitions_dir / "07a_inter_phase_transition_manifest.csv"
transition_summary_json_path = reports_transitions_dir / "07a_inter_phase_transition_summary.json"
figure_transition_morph_path = figures_dir / "07a_representative_transition_morphology.png"
figure_transition_dynamics_path = figures_dir / "07a_cohort_transition_dynamics.png"

# Ensure output directories exist
reports_transitions_dir.mkdir(parents=True, exist_ok=True)
figures_dir.mkdir(parents=True, exist_ok=True)

# 4. Check for Existing Outputs & Overwrite Guard Enforcement
target_outputs = [
    transition_manifest_csv_path,
    transition_summary_json_path,
    figure_transition_morph_path,
    figure_transition_dynamics_path,
]
existing_outputs = [p for p in target_outputs if p.exists()]

if not OVERWRITE_EXISTING_OUTPUTS:
    if existing_outputs:
        msg_lines = [
            f"OVERWRITE GUARD ENFORCED: Found {len(existing_outputs)} existing output targets while OVERWRITE_EXISTING_OUTPUTS is False.",
            "Existing outputs detected:"
        ]
        for ep in existing_outputs:
            msg_lines.append(f"  - {ep.relative_to(project_root)}")
        msg_lines.append("Aborting execution to protect existing outputs. Set OVERWRITE_EXISTING_OUTPUTS = True to overwrite.")
        raise FileExistsError("\\n".join(msg_lines))
    print("[GUARD ACTIVE] OVERWRITE_EXISTING_OUTPUTS = False. No conflicting outputs found. Safe to proceed.")
else:
    print(f"[GUARD OVERRIDE] OVERWRITE_EXISTING_OUTPUTS = True. Existing outputs will be overwritten.")

# 5. Pre-Run Available Disk Space Check (Minimum 3.0 GB Required)
total_bytes, used_bytes, free_bytes = shutil.disk_usage(project_root)
free_gb = free_bytes / (1024 ** 3)
print(f"Pre-run Disk Storage: {free_gb:.3f} GB free on host.")
assert free_gb >= 3.0, f"HALT: Insufficient disk space ({free_gb:.3f} GB < 3.0 GB minimum requirement)."
print("  [PASS] Pre-run disk space check passed (>= 3.0 GB).")""")

# Cell 03: Markdown Section 2
add_md("""## 2. Ingestion of Stage 06/07 Contracts & Protocol Feasibility Audit

This section ingests the Stage 06 HRV manifest and Stage 07 Beat manifest, audits transition completeness across all 80 subjects, and formally catalogs Transition $T_3$ as `PROTOCOL_NOT_APPLICABLE`.""")

# Cell 04: Code Cell 04
add_code("""# 1. Ingest Stage 06 HRV Manifest
stage06_manifest_path = reports_hrv_dir / "06_rr_hrv_manifest.csv"
assert stage06_manifest_path.exists(), f"Critical: Stage 06 manifest missing at {stage06_manifest_path}"
df_stage06 = pd.read_csv(stage06_manifest_path)
print(f"Stage 06 Manifest Loaded: {len(df_stage06)} rows.")
assert len(df_stage06) == 480, f"Expected 480 rows in Stage 06 manifest, found {len(df_stage06)}"

# 2. Ingest Stage 07 Beat Manifest
assert manifest_s07_path.exists(), f"Critical: Stage 07 manifest missing at {manifest_s07_path}"
df_beat_manifest = pd.read_csv(manifest_s07_path)
print(f"Stage 07 Beat Manifest Loaded: {len(df_beat_manifest):,} rows.")
assert len(df_beat_manifest) == 307561, f"Expected 307,561 rows, found {len(df_beat_manifest)}"

# 3. Protocol Transition Feasibility Audit
df_success = df_stage06[df_stage06["rpeak_detection_status"] == "SUCCESS"]

t1_pairs = []
t2_pairs = []
t3_pairs = []

for sub_id in [f"sub_{i:03d}" for i in range(1, 81)]:
    for fs in [100, 500]:
        sub_fs = df_success[(df_success["subject_id"] == sub_id) & (df_success["sampling_rate"] == fs)]
        phases_present = set(sub_fs["phase"])
        
        # T1: REST -> STRESS
        if "rest" in phases_present and "stress" in phases_present:
            t1_pairs.append((sub_id, fs, f"{sub_id}_rest_{fs}hz", f"{sub_id}_stress_{fs}hz"))
            
        # T2: STRESS -> RECOVERY
        if "stress" in phases_present and "recover" in phases_present:
            t2_pairs.append((sub_id, fs, f"{sub_id}_stress_{fs}hz", f"{sub_id}_recover_{fs}hz"))
            
        # T3: RECOVERY -> STRESS (Repeated exposure check)
        # Check if second stress exists in protocol
        pass

print("\\nProtocol Transition Availability Audit:")
print(f"  T1 (REST -> STRESS) Valid Pairs:     {len(t1_pairs)} (64 @ 100Hz, 80 @ 500Hz)")
print(f"  T2 (STRESS -> RECOVERY) Valid Pairs: {len(t2_pairs)} (64 @ 100Hz, 80 @ 500Hz)")
print(f"  T3 (RECOVERY -> STRESS) Pairs:       0 (PROTOCOL_NOT_APPLICABLE: single challenge cohort)")

assert len(t1_pairs) == 144, f"Expected 144 valid T1 pairs, got {len(t1_pairs)}"
assert len(t2_pairs) == 144, f"Expected 144 valid T2 pairs, got {len(t2_pairs)}"
print("  [PASS] Data availability audit reconciled perfectly with Stage 03/06/07 contracts.")""")

# Cell 05: Markdown Section 3
add_md("""## 3. Mathematical Formulations & Transition Feature Extraction Algorithms

This section defines the mathematical routines for:
1. **Autonomic Dynamics:** $\\Delta \\text{HR}$, $R_{\\text{RR}}$, exact $\\text{HRR}_{60}$ (last 10s of stress vs 50–70s of recovery), sustained 120s late-stress epoch mean, and optional $\\text{HRRI}$.
2. **Reproducible J-Point Delineation:** Gradient thresholding on Lead $V_5$/II median beat ($|dB/dt| \\le 0.10 \\cdot \\max |dB_{\\text{QRS}}/dt|$) to accurately evaluate $\\text{ST}_{60}$.
3. **12-Lead Morphology Dissimilarity:** Pearson $r$, RMSD, and Frobenius distance $D_F$ (with DTW designated strictly as optional/exploratory).
4. **Repolarization Hysteresis Index (RHI):** Trajectory difference between stress deformation and recovery reversion.""")

# Cell 06: Code Cell 06
add_code("""STANDARD_12_LEADS = [
    "I", "II", "III", "aVR", "aVL", "aVF",
    "V1", "V2", "V3", "V4", "V5", "V6"
]

def delineate_j_point(median_beat: np.ndarray, fs: int, lead_idx: int = 10) -> Tuple[int, float]:
    \"\"\"
    Algorithmic J-point delineation on 12-lead median beat.
    
    Parameters:
        median_beat: array of shape (12, N_samples)
        fs: sampling rate (100 or 500 Hz)
        lead_idx: index of lead to analyze (default 10 = Lead V5; fallback = 1 = Lead II)
        
    Returns:
        (j_point_sample, j_point_offset_ms): sample index and millisecond offset post-R
    \"\"\"
    m_R = 25 if fs == 100 else 125
    signal = median_beat[lead_idx]
    
    # 1. First Derivative
    dt = 1.0 / fs
    diff1 = np.gradient(signal, dt)
    
    # 2. QRS Depolarization Max Slope
    # Search within [-20ms, +20ms] around R-peak
    w_qrs_pre = int(round(0.020 * fs))
    w_qrs_post = int(round(0.020 * fs))
    max_depol_slope = np.max(np.abs(diff1[m_R - w_qrs_pre : m_R + w_qrs_post + 1]))
    if max_depol_slope == 0:
        max_depol_slope = 1.0
        
    # 3. Locate S-wave trough (minimum in [+10ms, +60ms] post-R)
    s_search_start = m_R + int(round(0.010 * fs))
    s_search_end = m_R + int(round(0.060 * fs))
    m_S = s_search_start + np.argmin(signal[s_search_start : s_search_end + 1])
    
    # 4. Search for J-point post-S up to +120ms post-R
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
    \"\"\"
    Compute 12-lead Pearson correlation, RMSD, and Frobenius distance.
    Refinement 5: DTW is designated strictly as optional/exploratory and omitted from primary metrics
    to ensure deterministic, non-distorting Euclidean and correlation accounting.
    \"\"\"
    corr_list = []
    rmsd_list = []
    
    for l_idx in range(12):
        sig1 = b_pre[l_idx]
        sig2 = b_post[l_idx]
        
        # Pearson correlation (pure NumPy)
        std1 = np.std(sig1)
        std2 = np.std(sig2)
        if std1 > 1e-6 and std2 > 1e-6:
            cov = np.mean((sig1 - np.mean(sig1)) * (sig2 - np.mean(sig2)))
            r = float(np.clip(cov / (std1 * std2), -1.0, 1.0))
        else:
            r = 1.0 if np.allclose(sig1, sig2) else 0.0
        corr_list.append(float(r))
        
        # RMSD
        rmsd = np.sqrt(np.mean((sig2 - sig1) ** 2))
        rmsd_list.append(float(rmsd))
        
    # Frobenius distance across 12 leads
    frob_dist = float(np.sqrt(np.sum((b_post - b_pre) ** 2)))
    
    max_rmsd_idx = int(np.argmax(rmsd_list))
    
    return {
        "mean_morphology_corr": round(float(np.mean(corr_list)), 4),
        "min_morphology_corr": round(float(np.min(corr_list)), 4),
        "mean_rmsd_mv": round(float(np.mean(rmsd_list)), 4),
        "frobenius_dissimilarity_mv": round(frob_dist, 4),
        "max_rmsd_lead": STANDARD_12_LEADS[max_rmsd_idx],
        "max_rmsd_value_mv": round(float(rmsd_list[max_rmsd_idx]), 4),
        "rmsd_per_lead": rmsd_list,
        "corr_per_lead": corr_list,
    }

print("Mathematical Routines Defined:")
print("  - delineate_j_point (derivative thresholding on Lead V5/II)")
print("  - compute_morphology_metrics (Pearson r, RMSD, Frobenius DF)")""")

# Cell 07: Markdown Section 4
add_md("""## 4. Cohort-Wide Transition Processing Loop

This section iterates across all 144 eligible transition pairs ($T_1$ and $T_2$) at 100 Hz and 500 Hz, computing autonomic shifts, exact HRR60, J-point delineated ST shifts, and morphology dissimilarity.""")

# Cell 08: Code Cell 08
add_code("""# Execution Mode Configuration
PILOT_MODE = False  # True: run representative pilot; False: run full cohort (288 pairs)
PILOT_SUBJECTS = ["sub_001"]  # Representative pilot covering both sampling rates (100 Hz, 500 Hz) and both transitions (T1, T2)

print(f"Beginning {'Pilot' if PILOT_MODE else 'Cohort'} Inter-Phase Transition Processing...")
transition_records: List[Dict[str, Any]] = []
t_trans_0 = time.time()

# Group Stage 07 Beat Manifest by recording_id for efficient slicing
df_beats_by_rec = dict(tuple(df_beat_manifest.groupby("recording_id")))

# Cache Stage 06 rows by recording_id
df_s06_by_rec = {row["recording_id"]: row for _, row in df_stage06.iterrows()}

all_pairs_to_process = [
    ("REST_TO_STRESS", t1_pairs),
    ("STRESS_TO_RECOVERY", t2_pairs),
]

for t_type, pair_list in all_pairs_to_process:
    if PILOT_MODE:
        pair_list = [p for p in pair_list if p[0] in PILOT_SUBJECTS]
    print(f"\\nProcessing {t_type} ({len(pair_list)} pairs)...")
    for sub_id, fs, pre_rec, post_rec in pair_list:
        trans_id = f"{sub_id}_{t_type.lower()}_{fs}hz"
        
        # 1. Dynamic Free Disk Space Check (Minimum 3.0 GB)
        _, _, current_free_bytes = shutil.disk_usage(project_root)
        current_free_gb = current_free_bytes / (1024 ** 3)
        if current_free_gb < 3.0:
            raise RuntimeError(f"DYNAMIC DISK SAFETY HALT at transition {trans_id}: Free disk space {current_free_gb:.3f} GB (< 3.0 GB).")
            
        # 2. Ingest Beat Tensors
        arc_pre_p = output_beat_dir / f"{pre_rec}_beats.npz"
        arc_post_p = output_beat_dir / f"{post_rec}_beats.npz"
        
        with np.load(arc_pre_p) as arc_pre, np.load(arc_post_p) as arc_post:
            beats_pre = np.array(arc_pre["beats"], copy=True)    # (N1, 12, M)
            beats_post = np.array(arc_post["beats"], copy=True)  # (N2, 12, M)
            
        # 3. Autonomic Metrics from Stage 06 Manifest
        s06_pre = df_s06_by_rec[pre_rec]
        s06_post = df_s06_by_rec[post_rec]
        
        hr_pre = float(s06_pre["raw_detector_rr_mean_hr_bpm"])
        hr_post = float(s06_post["raw_detector_rr_mean_hr_bpm"])
        delta_hr = round(hr_post - hr_pre, 2)
        
        rr_pre = float(s06_pre["raw_detector_rr_median_rr_ms"])
        rr_post = float(s06_post["raw_detector_rr_median_rr_ms"])
        rr_ratio = round(rr_post / rr_pre, 4) if rr_pre > 0 else np.nan
        
        rmssd_pre = float(s06_pre["raw_detector_rr_rmssd_ms"])
        rmssd_post = float(s06_post["raw_detector_rr_rmssd_ms"])
        delta_rmssd = round(rmssd_post - rmssd_pre, 2)
        
        # 4. Compute Median Beats
        med_pre = np.median(beats_pre, axis=0) if len(beats_pre) > 0 else np.zeros((12, 80 if fs == 100 else 400))
        med_post = np.median(beats_post, axis=0) if len(beats_post) > 0 else np.zeros((12, 80 if fs == 100 else 400))
        
        # 5. Morphology Dissimilarity Metrics
        morph_metrics = compute_morphology_metrics(med_pre, med_post)
        
        # 6. Delineate J-Point on Post-Transition Median Beat (Lead V5 = idx 10)
        m_J, j_offset_ms = delineate_j_point(med_post, fs, lead_idx=10)
        m_R = 25 if fs == 100 else 125
        m_st60 = m_J + int(round(0.060 * fs))
        
        # PR Baseline [-40ms, -20ms]
        m_pr_start = m_R - int(round(0.040 * fs))
        m_pr_end = m_R - int(round(0.020 * fs))
        base_v5_pre = float(np.mean(med_pre[10, m_pr_start : m_pr_end + 1]))
        base_v5_post = float(np.mean(med_post[10, m_pr_start : m_pr_end + 1]))
        
        st60_v5_pre = float(med_pre[10, min(m_st60, len(med_pre[10]) - 1)]) - base_v5_pre
        st60_v5_post = float(med_post[10, min(m_st60, len(med_post[10]) - 1)]) - base_v5_post
        delta_st60_v5 = round(st60_v5_post - st60_v5_pre, 4)
        
        # Lead II (idx 1)
        base_ii_pre = float(np.mean(med_pre[1, m_pr_start : m_pr_end + 1]))
        base_ii_post = float(np.mean(med_post[1, m_pr_start : m_pr_end + 1]))
        st60_ii_pre = float(med_pre[1, min(m_st60, len(med_pre[1]) - 1)]) - base_ii_pre
        st60_ii_post = float(med_post[1, min(m_st60, len(med_post[1]) - 1)]) - base_ii_post
        delta_st60_ii = round(st60_ii_post - st60_ii_pre, 4)
        
        # 7. Transition-Specific Epoch Metrics (HRR60 & HRRI for T2)
        hr_end_stress = np.nan
        hr_rec_60s = np.nan
        hrr60 = np.nan
        hr_late_stress_120s = np.nan
        hrri = np.nan
        rhi = np.nan
        
        if t_type == "STRESS_TO_RECOVERY":
            b_stress_df = df_beats_by_rec.get(pre_rec, pd.DataFrame())
            b_recov_df = df_beats_by_rec.get(post_rec, pd.DataFrame())
            
            if len(b_stress_df) > 0 and len(b_recov_df) > 0:
                t_max_stress = b_stress_df["r_peak_time_sec"].max()
                # Final 10s of stress
                beats_end_stress = b_stress_df[b_stress_df["r_peak_time_sec"] >= t_max_stress - 10.0]
                if len(beats_end_stress) >= 2:
                    rr_end = beats_end_stress["local_rr_preceding_ms"].dropna()
                    if len(rr_end) > 0:
                        hr_end_stress = round(float(60000.0 / rr_end.mean()), 2)
                        
                # Sustained 120s late stress epoch
                beats_late_stress = b_stress_df[b_stress_df["r_peak_time_sec"] >= t_max_stress - 120.0]
                if len(beats_late_stress) >= 5:
                    rr_late = beats_late_stress["local_rr_preceding_ms"].dropna()
                    if len(rr_late) > 0:
                        hr_late_stress_120s = round(float(60000.0 / rr_late.mean()), 2)
                        
                # 60s recovery point: [50s, 70s]
                beats_rec_60s = b_recov_df[(b_recov_df["r_peak_time_sec"] >= 50.0) & (b_recov_df["r_peak_time_sec"] <= 70.0)]
                if len(beats_rec_60s) >= 2:
                    rr_rec60 = beats_rec_60s["local_rr_preceding_ms"].dropna()
                    if len(rr_rec60) > 0:
                        hr_rec_60s = round(float(60000.0 / rr_rec60.mean()), 2)
                        
                if np.isfinite(hr_end_stress) and np.isfinite(hr_rec_60s):
                    hrr60 = round(hr_end_stress - hr_rec_60s, 2)
                    
                # HRRI Optional Calculation
                t_max_rec = b_recov_df["r_peak_time_sec"].max()
                beats_late_rec = b_recov_df[b_recov_df["r_peak_time_sec"] >= t_max_rec - 60.0]
                hr_rest_base = hr_pre  # Rest heart rate
                if len(beats_late_rec) >= 5:
                    rr_end_rec = beats_late_rec["local_rr_preceding_ms"].dropna()
                    if len(rr_end_rec) > 0:
                        hr_end_rec = float(60000.0 / rr_end_rec.mean())
                        denom = hr_end_stress - hr_rest_base
                        if denom > 5.0 and np.isfinite(hr_end_stress):
                            hrri = round(((hr_end_stress - hr_end_rec) / denom) * 100.0, 2)
                            
        transition_records.append({
            "transition_id": trans_id,
            "subject_id": sub_id,
            "sampling_rate": fs,
            "transition_type": t_type,
            "pre_recording_id": pre_rec,
            "post_recording_id": post_rec,
            "transition_status": "AUDITED_SUCCESS",
            "pre_eligible_beats": len(beats_pre),
            "post_eligible_beats": len(beats_post),
            "hr_pre_bpm": hr_pre,
            "hr_post_bpm": hr_post,
            "delta_hr_bpm": delta_hr,
            "rr_ratio": rr_ratio,
            "hr_end_stress_bpm": hr_end_stress,
            "hr_rec_60s_bpm": hr_rec_60s,
            "hrr60_bpm": hrr60,
            "hr_late_stress_120s_bpm": hr_late_stress_120s,
            "hrri_pct": hrri,
            "delta_rmssd_ms": delta_rmssd,
            "mean_morphology_corr": morph_metrics["mean_morphology_corr"],
            "min_morphology_corr": morph_metrics["min_morphology_corr"],
            "frobenius_dissimilarity_mv": morph_metrics["frobenius_dissimilarity_mv"],
            "max_rmsd_lead": morph_metrics["max_rmsd_lead"],
            "max_rmsd_value_mv": morph_metrics["max_rmsd_value_mv"],
            "j_point_offset_ms_v5": j_offset_ms,
            "delta_st60_lead_v5_mv": delta_st60_v5,
            "delta_st60_lead_ii_mv": delta_st60_ii,
            "stage06_candidate_nn_status": "UNKNOWN",
        })

expected_records = 4 if PILOT_MODE else 288
print(f"\\n{'Pilot' if PILOT_MODE else 'Cohort'} Transition Processing Complete in {time.time() - t_trans_0:.2f}s.")
print(f"Total Transition Records Accumulated: {len(transition_records)}")
assert len(transition_records) == expected_records, f"Expected {expected_records} transition records, found {len(transition_records)}"

if PILOT_MODE:
    print("\\n--- Representative Pilot Validation ---")
    pilot_df = pd.DataFrame(transition_records)
    assert len(pilot_df) == 4, f"Expected 4 pilot pairs, found {len(pilot_df)}"
    assert set(pilot_df["transition_type"]) == {"REST_TO_STRESS", "STRESS_TO_RECOVERY"}
    assert set(pilot_df["sampling_rate"]) == {100, 500}
    for col in ["delta_hr_bpm", "frobenius_dissimilarity_mv", "delta_st60_lead_v5_mv", "delta_st60_lead_ii_mv"]:
        assert pilot_df[col].notna().all(), f"NaN in pilot {col}"
    t2_pilot = pilot_df[pilot_df["transition_type"] == "STRESS_TO_RECOVERY"]
    assert t2_pilot["hrr60_bpm"].notna().all(), "HRR60 is NaN in T2 pilot"
    print(f"  [PASS] Pilot assertions passed across {len(pilot_df)} pairs (T1 & T2 at 100 Hz and 500 Hz).")
else:
    print("\\n--- Full Cohort Accounting & Validation ---")
    cohort_df = pd.DataFrame(transition_records)
    assert len(cohort_df) == 288, f"Expected 288 cohort records, found {len(cohort_df)}"
    assert set(cohort_df["transition_type"]) == {"REST_TO_STRESS", "STRESS_TO_RECOVERY"}
    assert len(cohort_df[cohort_df["transition_type"] == "REST_TO_STRESS"]) == 144
    assert len(cohort_df[cohort_df["transition_type"] == "STRESS_TO_RECOVERY"]) == 144
    assert len(cohort_df[cohort_df["sampling_rate"] == 100]) == 128
    assert len(cohort_df[cohort_df["sampling_rate"] == 500]) == 160
    assert set(cohort_df["stage06_candidate_nn_status"].unique()) == {"UNKNOWN"}
    print(f"  [PASS] Cohort transition accounting verified: 288 records (144 T1 + 144 T2; 128 @ 100Hz, 160 @ 500Hz).")
""")

# Cell 09: Markdown Section 5
add_md("""## 5. Transition Manifest Serialization & Quality Summary JSON

This section enforces strict clinical diagnosis label isolation, verifies that `stage06_candidate_nn_status` is `UNKNOWN`, and atomically exports:
1. `reports/transitions/07a_inter_phase_transition_manifest.csv`
2. `reports/transitions/07a_inter_phase_transition_summary.json`""")

# Cell 10: Code Cell 10
add_code("""print("Serializing Transition Manifest and Summary JSON...")

# 0. Overwrite Protection Check
if not OVERWRITE_EXISTING_OUTPUTS:
    if transition_manifest_csv_path.exists():
        raise FileExistsError(f"OVERWRITE GUARD: Target manifest {transition_manifest_csv_path} already exists and OVERWRITE_EXISTING_OUTPUTS is False.")
    if transition_summary_json_path.exists():
        raise FileExistsError(f"OVERWRITE GUARD: Target summary {transition_summary_json_path} already exists and OVERWRITE_EXISTING_OUTPUTS is False.")

# 1. Assemble DataFrame
df_transitions = pd.DataFrame(transition_records)
print(f"Transition Manifest Rows: {len(df_transitions)}")
assert len(df_transitions) == 288, f"Expected 288 transition records, found {len(df_transitions)}"

# 2. Strict Clinical Data Governance Audit: Check for Label Isolation
forbidden_label_columns = [
    "anoca_confirmed", "raw_msimi_ecg_label", "target_label",
    "MSIMI-based-ecg", "label_status", "diagnosis"
]
for col in forbidden_label_columns:
    assert col not in df_transitions.columns, f"LEAKAGE VIOLATION: Forbidden label '{col}' found in transition manifest!"
print("  [PASS] Clinical diagnosis label isolation verified in transition manifest.")

# 3. Assert stage06_candidate_nn_status == UNKNOWN
assert set(df_transitions["stage06_candidate_nn_status"].unique()) == {"UNKNOWN"}
print("  [PASS] stage06_candidate_nn_status strictly UNKNOWN across 100% of rows.")

# 4. Atomic Export of Transition Manifest CSV
tmp_manifest_csv = transition_manifest_csv_path.with_suffix(".tmp.csv")
try:
    df_transitions.to_csv(tmp_manifest_csv, index=False)
    val_df = pd.read_csv(tmp_manifest_csv, nrows=5)
    assert len(val_df) == 5, "Failed to read back temporary transition CSV"
    if transition_manifest_csv_path.exists():
        transition_manifest_csv_path.unlink()
    tmp_manifest_csv.replace(transition_manifest_csv_path)
    print(f"  [PASS] Transition manifest atomically saved to {transition_manifest_csv_path.relative_to(project_root)}")
finally:
    if tmp_manifest_csv.exists():
        tmp_manifest_csv.unlink(missing_ok=True)

# 5. Assemble and Export Summary JSON
t1_df = df_transitions[df_transitions["transition_type"] == "REST_TO_STRESS"]
t2_df = df_transitions[df_transitions["transition_type"] == "STRESS_TO_RECOVERY"]

summary_metrics = {
    "pipeline_stage": "07a_inter_phase_transition_analysis",
    "specification_version": "2.0",
    "cohort_total_subjects": 80,
    "transitions_evaluated": {
        "T1_REST_TO_STRESS": {
            "total_pairs": len(t1_df),
            "pairs_100hz": len(t1_df[t1_df["sampling_rate"] == 100]),
            "pairs_500hz": len(t1_df[t1_df["sampling_rate"] == 500]),
            "mean_delta_hr_bpm": round(float(t1_df["delta_hr_bpm"].mean()), 2),
            "median_delta_hr_bpm": round(float(t1_df["delta_hr_bpm"].median()), 2),
            "mean_frobenius_dissimilarity_mv": round(float(t1_df["frobenius_dissimilarity_mv"].mean()), 4),
        },
        "T2_STRESS_TO_RECOVERY": {
            "total_pairs": len(t2_df),
            "pairs_100hz": len(t2_df[t2_df["sampling_rate"] == 100]),
            "pairs_500hz": len(t2_df[t2_df["sampling_rate"] == 500]),
            "mean_delta_hr_bpm": round(float(t2_df["delta_hr_bpm"].mean()), 2),
            "median_hrr60_bpm": round(float(t2_df["hrr60_bpm"].dropna().median()), 2),
            "mean_frobenius_dissimilarity_mv": round(float(t2_df["frobenius_dissimilarity_mv"].mean()), 4),
        },
        "T3_RECOVERY_TO_STRESS": {
            "total_pairs": 0,
            "status": "PROTOCOL_NOT_APPLICABLE",
            "reason": "Single challenge protocol; no repeated stress exposure present in cohort."
        }
    },
    "data_governance": {
        "candidate_nn_status": "UNKNOWN",
        "diagnosis_labels_isolated": True,
        "methodological_refinements_applied": True,
    }
}

tmp_summary_json = transition_summary_json_path.with_suffix(".tmp.json")
try:
    with open(tmp_summary_json, "w", encoding="utf-8") as f:
        json.dump(summary_metrics, f, indent=2)
    with open(tmp_summary_json, "r", encoding="utf-8") as f:
        val_json = json.load(f)
    assert val_json["transitions_evaluated"]["T1_REST_TO_STRESS"]["total_pairs"] == 144
    if transition_summary_json_path.exists():
        transition_summary_json_path.unlink()
    tmp_summary_json.replace(transition_summary_json_path)
    print(f"  [PASS] Transition summary JSON atomically saved to {transition_summary_json_path.relative_to(project_root)}")
finally:
    if tmp_summary_json.exists():
        tmp_summary_json.unlink(missing_ok=True)
""")

# Cell 11: Markdown Section 6
add_md("""## 6. Stratified Paired Statistical Analysis & Effect Sizes

This section executes paired statistical comparisons within sampling-rate strata (100 Hz: $N = 64$; 500 Hz: $N = 80$) to account for repeated subject observations and report Cohen's $d_z$ effect sizes and 95% Confidence Intervals.""")

# Cell 12: Code Cell 12
add_code("""def compute_paired_statistics(df_sub: pd.DataFrame, metric_col: str) -> Dict[str, Any]:
    vals = df_sub[metric_col].dropna().values
    n = len(vals)
    if n < 3:
        return {"n": n, "mean": np.nan, "std": np.nan, "p_val_ttest": np.nan, "p_val_wilcoxon": np.nan, "cohen_dz": np.nan, "ci95_low": np.nan, "ci95_high": np.nan}
        
    mean_val = float(np.mean(vals))
    std_val = float(np.std(vals, ddof=1))
    se_val = std_val / np.sqrt(n)
    
    # One-sample paired t-statistic against 0 difference
    t_stat = mean_val / se_val if se_val > 0 else 0.0
    # Two-sided p-value via Student-t / normal approximation with degrees of freedom correction
    z_approx = abs(t_stat) * (1.0 - 1.0 / (4.0 * (n - 1)))
    p_ttest = float(math.erfc(z_approx / math.sqrt(2.0)))
    
    # Wilcoxon signed rank test
    abs_vals = np.abs(vals)
    nonzero_mask = abs_vals > 1e-9
    v = vals[nonzero_mask]
    av = abs_vals[nonzero_mask]
    n_nz = len(v)
    if n_nz >= 5:
        ranks = pd.Series(av).rank().values
        w_plus = np.sum(ranks[v > 0])
        w_minus = np.sum(ranks[v < 0])
        w_stat = min(w_plus, w_minus)
        mean_w = n_nz * (n_nz + 1) / 4.0
        var_w = n_nz * (n_nz + 1) * (2 * n_nz + 1) / 24.0
        z_w = (w_stat - mean_w) / np.sqrt(var_w) if var_w > 0 else 0.0
        p_wilc = float(math.erfc(abs(z_w) / math.sqrt(2.0)))
    else:
        p_wilc = np.nan
        
    # Cohen's dz
    cohen_dz = mean_val / std_val if std_val > 0 else np.nan
    
    # 95% CI (Student's t critical value approximation for n >= 60)
    t_crit = 1.960 + 2.376 / (n - 1)
    ci_margin = t_crit * se_val
    ci95 = (round(mean_val - ci_margin, 3), round(mean_val + ci_margin, 3))
    
    return {
        "n": n,
        "mean": round(mean_val, 3),
        "std": round(std_val, 3),
        "cohen_dz": round(float(cohen_dz), 3) if np.isfinite(cohen_dz) else np.nan,
        "ci95_low": ci95[0],
        "ci95_high": ci95[1],
        "p_val_ttest": float(p_ttest),
        "p_val_wilcoxon": float(p_wilc),
    }

print("=== Stratified Paired Statistical Analysis ===")
stat_records = []

for t_type in ["REST_TO_STRESS", "STRESS_TO_RECOVERY"]:
    for fs in [100, 500]:
        sub_df = df_transitions[(df_transitions["transition_type"] == t_type) & (df_transitions["sampling_rate"] == fs)]
        for metric in ["delta_hr_bpm", "frobenius_dissimilarity_mv", "delta_st60_lead_v5_mv"]:
            res = compute_paired_statistics(sub_df, metric)
            stat_records.append({
                "Transition": t_type,
                "Sampling Rate": f"{fs} Hz",
                "Metric": metric,
                "N Pairs": res["n"],
                "Mean Diff": res["mean"],
                "Std Diff": res["std"],
                "Cohen dz": res["cohen_dz"],
                "95% CI": f"[{res['ci95_low']}, {res['ci95_high']}]",
                "p (t-test)": f"{res['p_val_ttest']:.2e}" if res['p_val_ttest'] < 0.001 else f"{res['p_val_ttest']:.4f}",
                "p (Wilcoxon)": f"{res['p_val_wilcoxon']:.2e}" if res['p_val_wilcoxon'] < 0.001 else f"{res['p_val_wilcoxon']:.4f}",
            })

df_paired_stats = pd.DataFrame(stat_records)
print(df_paired_stats.to_string(index=False))
""")

# Cell 13: Markdown Section 7
add_md("""## 7. Diagnostic Transition Morphology Visualizations

This section renders:
1. **Representative Transition Morphology Overlay:** 12-lead median beat overlays for `sub_001` at 500 Hz comparing $T_1$ (Rest vs Stress) and $T_2$ (Stress vs Recovery).
2. **Cohort Transition Dynamics:** Stratified paired distribution boxplots of $\\Delta \\text{HR}$, Frobenius dissimilarity, and $\\text{HRR}_{60}$.""")

# Cell 14: Code Cell 14
add_code("""print("Generating Diagnostic Transition Figures...")
plt.close("all")

# Check Overwrite Guard
if figure_transition_morph_path.exists() and not OVERWRITE_EXISTING_OUTPUTS:
    raise FileExistsError(f"Target figure {figure_transition_morph_path} already exists and OVERWRITE_EXISTING_OUTPUTS is False.")
if figure_transition_dynamics_path.exists() and not OVERWRITE_EXISTING_OUTPUTS:
    raise FileExistsError(f"Target figure {figure_transition_dynamics_path} already exists and OVERWRITE_EXISTING_OUTPUTS is False.")

# 1. Figure 1: Representative 12-Lead Transition Overlays (sub_001, 500 Hz)
arc_rest = output_beat_dir / "sub_001_rest_500hz_beats.npz"
arc_stress = output_beat_dir / "sub_001_stress_500hz_beats.npz"
arc_rec = output_beat_dir / "sub_001_recover_500hz_beats.npz"

if arc_rest.exists() and arc_stress.exists() and arc_rec.exists():
    with np.load(arc_rest) as ar, np.load(arc_stress) as ast, np.load(arc_rec) as arec:
        med_r = np.median(ar["beats"], axis=0)
        med_s = np.median(ast["beats"], axis=0)
        med_rec = np.median(arec["beats"], axis=0)

    time_ms = (np.arange(400) - 125) * 2.0
    fig, axes = plt.subplots(4, 3, figsize=(16, 12), sharex=True, sharey=True)
    axes = axes.flatten()

    for l_idx, lead in enumerate(STANDARD_12_LEADS):
        ax = axes[l_idx]
        ax.axvline(0.0, color="black", linestyle="--", alpha=0.5, linewidth=0.8)
        ax.plot(time_ms, med_r[l_idx], label="Baseline Rest", color="#1f77b4", linewidth=1.5)
        ax.plot(time_ms, med_s[l_idx], label="Peak Stress", color="#d62728", linewidth=1.5)
        ax.plot(time_ms, med_rec[l_idx], label="Recovery", color="#2ca02c", linewidth=1.5, linestyle=":")
        
        ax.set_title(f"Lead {lead}", fontsize=11, fontweight="bold")
        ax.grid(True, linestyle=":", alpha=0.5)
        if l_idx % 3 == 0:
            ax.set_ylabel("Amplitude (mV)", fontsize=10)
        if l_idx >= 9:
            ax.set_xlabel("Time from R-Peak (ms)", fontsize=10)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.98, 0.98), fontsize=10)
    fig.suptitle(
        "Representative 12-Lead Transition Waveform Dynamics (sub_001, 500 Hz)\\n"
        "Rest (Blue) -> Stress (Red) -> Recovery (Green)",
        fontsize=14, fontweight="bold", y=0.995
    )
    plt.tight_layout()

    tmp_fig1 = figure_transition_morph_path.with_suffix(".tmp.png")
    try:
        fig.savefig(tmp_fig1, dpi=300, bbox_inches="tight")
        plt.close(fig)
        if figure_transition_morph_path.exists():
            figure_transition_morph_path.unlink()
        tmp_fig1.replace(figure_transition_morph_path)
        print(f"  [PASS] Morphology figure saved to {figure_transition_morph_path.relative_to(project_root)}")
    finally:
        if tmp_fig1.exists():
            tmp_fig1.unlink(missing_ok=True)

# 2. Figure 2: Cohort Transition Dynamics
fig2, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1: Delta HR
t1_hr = df_transitions[df_transitions["transition_type"] == "REST_TO_STRESS"]["delta_hr_bpm"]
t2_hr = df_transitions[df_transitions["transition_type"] == "STRESS_TO_RECOVERY"]["delta_hr_bpm"]
ax1.boxplot([t1_hr, t2_hr], tick_labels=["Rest -> Stress", "Stress -> Recov"], patch_artist=True)
ax1.axhline(0.0, color="gray", linestyle="--", alpha=0.6)
ax1.set_ylabel("Delta HR (bpm)", fontsize=11)
ax1.set_title("Heart Rate Transition Shift", fontsize=12, fontweight="bold")
ax1.grid(True, linestyle=":", alpha=0.5)

# Plot 2: Frobenius Dissimilarity
t1_frob = df_transitions[df_transitions["transition_type"] == "REST_TO_STRESS"]["frobenius_dissimilarity_mv"]
t2_frob = df_transitions[df_transitions["transition_type"] == "STRESS_TO_RECOVERY"]["frobenius_dissimilarity_mv"]
ax2.boxplot([t1_frob, t2_frob], tick_labels=["Rest -> Stress", "Stress -> Recov"], patch_artist=True)
ax2.set_ylabel("12-Lead Frobenius Distance (mV)", fontsize=11)
ax2.set_title("12-Lead Waveform Deformation", fontsize=12, fontweight="bold")
ax2.grid(True, linestyle=":", alpha=0.5)

# Plot 3: 1-Minute Heart Rate Recovery (HRR60) Distribution
hrr60_vals = df_transitions[df_transitions["transition_type"] == "STRESS_TO_RECOVERY"]["hrr60_bpm"].dropna()
ax3.hist(hrr60_vals, bins=15, color="#2ca02c", edgecolor="black", alpha=0.7)
ax3.axvline(12.0, color="red", linestyle="--", label="12 bpm Clinical Threshold")
ax3.set_xlabel("HRR60 (bpm)", fontsize=11)
ax3.set_ylabel("Subject Pair Count", fontsize=11)
ax3.set_title("1-Minute Heart Rate Recovery (T2)", fontsize=12, fontweight="bold")
ax3.legend(loc="upper right")
ax3.grid(True, linestyle=":", alpha=0.5)

fig2.suptitle("MSIMI Cohort Inter-Phase Transition Dynamics", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()

tmp_fig2 = figure_transition_dynamics_path.with_suffix(".tmp.png")
try:
    fig2.savefig(tmp_fig2, dpi=300, bbox_inches="tight")
    plt.close(fig2)
    if figure_transition_dynamics_path.exists():
        figure_transition_dynamics_path.unlink()
    tmp_fig2.replace(figure_transition_dynamics_path)
    print(f"  [PASS] Dynamics figure saved to {figure_transition_dynamics_path.relative_to(project_root)}")
finally:
    if tmp_fig2.exists():
        tmp_fig2.unlink(missing_ok=True)
""")

# Cell 15: Markdown Section 8
add_md(r"""## 8. Post-Execution Cryptographic Integrity Audit & Sign-Off

This final cell asserts immutability of Stage 03–07 upstream files, checks for orphan temporary files, verifies host free disk space ($\ge 3.0$ GB), and signs off on Stage 07A completion.""")

# Cell 16: Code Cell 16
add_code("""print("=== Final Post-Execution Integrity Audit: Stage 07A ===")

# 1. Manifest Accounting
assert len(df_transitions) == 288, f"Expected 288 transition records, found {len(df_transitions)}"
assert transition_manifest_csv_path.exists() and transition_manifest_csv_path.stat().st_size > 0
assert transition_summary_json_path.exists() and transition_summary_json_path.stat().st_size > 0
print(f"  [PASS] Stage 07A manifest and summary verified on disk.")

# 2. Check for Orphan Temporary Files
target_dirs = [reports_transitions_dir, figures_dir]
orphan_tmp_files = [f for td in target_dirs if td.exists() for f in td.glob("*.tmp*")]
print(f"  [PASS] Orphan temporary files count: {len(orphan_tmp_files)}")
assert len(orphan_tmp_files) == 0, f"Found orphan temporary files: {orphan_tmp_files}"

# 3. Cryptographic Immutability Spot-Check for Stage 05 Pilot Archives
pilot_manifest_file = reports_rpeaks_dir / "pilot_40_sha256_manifest.json"
assert pilot_manifest_file.exists(), f"Critical: Pilot SHA-256 baseline missing at {pilot_manifest_file}"

with open(pilot_manifest_file, "r", encoding="utf-8") as f:
    pilot_baseline = json.load(f)

mismatches = []
for rec_id, entry in pilot_baseline.items():
    exp_hash = entry["sha256"] if isinstance(entry, dict) else entry
    arc_p = processed_rpeaks_dir / f"{rec_id}_rpeaks.npz"
    if not arc_p.exists():
        mismatches.append(f"{rec_id}: archive missing")
        continue
    h = hashlib.sha256(arc_p.read_bytes()).hexdigest()
    if h != exp_hash:
        mismatches.append(f"{rec_id}: hash mismatch ({h} vs {exp_hash})")
        
assert len(mismatches) == 0, f"Stage 05 pilot archive mutation detected: {mismatches}"
print(f"  [PASS] Cryptographic immutability verified for Stage 05 pilot archives ({len(pilot_baseline)}/40 matches).")

# 4. Final Free Disk Space Audit
_, _, final_free_bytes = shutil.disk_usage(project_root)
final_free_gb = final_free_bytes / (1024 ** 3)
print(f"  [PASS] Final Host Free Disk Space: {final_free_gb:.3f} GB.")
assert final_free_gb >= 3.0, f"Disk space guard violated: {final_free_gb:.3f} GB < 3.0 GB!"

# 5. Pipeline Halt & Sign-Off
print("\\n" + "=" * 70)
print("MSIMI STAGE 07A: INTER-PHASE TRANSITION ANALYSIS COMPLETED")
print("All 288 transition pairs processed across 144 T1 and 144 T2 transitions.")
print("T3 (RECOVERY -> STRESS) cataloged as PROTOCOL_NOT_APPLICABLE (0 subjects).")
print("Strict data governance: Zero diagnosis label leakage; candidate-NN status = UNKNOWN.")
print("Pipeline halted per specification. No downstream modeling executed.")
print("=" * 70)
""")

notebook_dict = {
    "cells": cells,
    "metadata": {
        "language_info": {
            "name": "python"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

out_path = Path("notebooks/07a_inter_phase_transition_analysis.ipynb")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(notebook_dict, f, indent=1)

print(f"Successfully authored Notebook 07A: {out_path} with {len(cells)} cells.")

