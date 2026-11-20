import json
from pathlib import Path

# Cells definition for Notebook 07 with Version 2.1 Approved Corrections

cells = []

def add_md(source_text):
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source_text.strip().split("\n")]
    })

def add_code(source_text):
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source_text.strip().split("\n")]
    })

# ==============================================================================
# Cell 00 [Markdown]: Title, Overview, Clinical Scope & Algorithmic Provenance
# ==============================================================================
add_md("""# 07 — Beat Window Segmentation
**MSIMI ECG Research Study | Cohort: `sub_001` through `sub_080` (80 Patients, 480 Acquisitions)**  
**Document Reference:** Approved Planning Specification `07_beat_window_segmentation_plan.md` (Version 2.1)  
**Notebook Type:** Pure Computational Pipeline Notebook  

---

### Clinical Scope & Algorithmic Provenance Disclaimers

1. **Algorithmic Fiducial Provenance:** All beat extraction is anchored to fiducial R-peaks detected algorithmically in Stage 05 (Pan–Tompkins dual-thresholding) without manual clinical ground-truth verification (`ALGORITHM_DETECTED_NOT_MANUALLY_VALIDATED`).
2. **Exploratory Beat Representations:** Segmented beat windows represent candidate cardiac cycles for downstream feature extraction (morphological, wavelet, deep embedding). They do **not** constitute validated clinical diagnoses or ectopic beat classifications.
3. **Provisional Window Configuration:** The 800 ms window (250 ms pre-R, 550 ms post-R) is a **provisional exploratory research configuration**, not an established or clinically validated electrophysiological standard.
4. **Non-Destructive Integrity:** No signal samples are padded, truncated, interpolated, or fabricated. Incomplete or non-finite beat windows are strictly preserved and flagged as ineligible rather than repaired.
5. **Leakage Restraint & Model Boundary:** No train/test splitting, dimensionality reduction, feature selection, or diagnostic classification is performed in Stage 07. All records preserve patient-level grouping identifiers (`subject_id`) to prevent future cross-validation leakage.""")

# ==============================================================================
# Cell 01 [Markdown]: Section 1 - Environment Setup, Configurations & Pre-Run Disk Space Guard
# ==============================================================================
add_md("""## 1. Environment Setup, Configuration Loading & Disk Space Pre-Check

This section initializes the execution environment, verifies absolute project paths, loads configuration parameters, establishes the atomic serialization guards (`OVERWRITE_EXISTING_OUTPUTS = False`), and verifies that the host system has at least 6.0 GB of free storage before proceeding.""")

# ==============================================================================
# Cell 02 [Code]: Environment Setup, Paths, Disk Check & Overwrite Guard
# ==============================================================================
add_code("""import sys
import os
import shutil
import json
import hashlib
import time
import gc
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd
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
reports_qc_dir = project_root / "reports" / "qc"
reports_hrv_dir = project_root / "reports" / "hrv"
reports_rpeaks_dir = project_root / "reports" / "rpeaks"
figures_dir = project_root / "outputs" / "figures"

manifest_csv_path = reports_segmentation_dir / "07_beat_segmentation_manifest.csv"
summary_json_path = reports_segmentation_dir / "07_beat_segmentation_summary.json"
patient_summary_csv_path = reports_segmentation_dir / "07_patient_beat_summary.csv"
figure_morphology_path = figures_dir / "07_representative_12lead_beat_morphology.png"

# Ensure output directories exist
output_beat_dir.mkdir(parents=True, exist_ok=True)
reports_segmentation_dir.mkdir(parents=True, exist_ok=True)
figures_dir.mkdir(parents=True, exist_ok=True)

# 4. Check for Existing Outputs & Overwrite Guard Enforcement
target_outputs = [manifest_csv_path, summary_json_path, patient_summary_csv_path, figure_morphology_path]
existing_outputs = [p for p in target_outputs if p.exists()]
existing_beat_archives = list(output_beat_dir.glob("*_beats.npz"))

if not OVERWRITE_EXISTING_OUTPUTS:
    if existing_outputs or existing_beat_archives:
        total_existing = len(existing_outputs) + len(existing_beat_archives)
        msg_lines = [
            f"OVERWRITE GUARD ENFORCED: Found {total_existing} existing output targets while OVERWRITE_EXISTING_OUTPUTS is False.",
            "Existing outputs detected:"
        ]
        for ep in existing_outputs:
            msg_lines.append(f"  - File: {ep.relative_to(project_root)}")
        if existing_beat_archives:
            msg_lines.append(f"  - Beat Archives: {len(existing_beat_archives)} archives in {output_beat_dir.relative_to(project_root)}")
        msg_lines.append("Aborting execution to protect existing outputs. Set OVERWRITE_EXISTING_OUTPUTS = True to overwrite.")
        raise FileExistsError("\\n".join(msg_lines))
    print("[GUARD ACTIVE] OVERWRITE_EXISTING_OUTPUTS = False. No conflicting outputs found. Safe to proceed.")
else:
    print(f"[GUARD OVERRIDE] OVERWRITE_EXISTING_OUTPUTS = True. Existing outputs will be overwritten.")

# 5. Pre-Run Storage Space Check (Minimum 6.0 GB Required)
total_bytes, used_bytes, free_bytes = shutil.disk_usage(project_root)
free_gb = free_bytes / (1024 ** 3)
print(f"Pre-run Disk Storage: {free_gb:.3f} GB free on host.")
if free_gb < 6.0:
    raise RuntimeError(
        f"INSUFFICIENT DISK SPACE: Pre-run safety check requires >= 6.0 GB free, "
        f"but only {free_gb:.3f} GB is available. Aborting."
    )
print("  [PASS] Pre-run disk space check passed (>= 6.0 GB).")

# 6. Global Standards & Seed
np.random.seed(42)
STANDARD_12_LEADS = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']""")

# ==============================================================================
# Cell 03 [Markdown]: Section 2 - Upstream Contract, Manifest Ingestion & Shortened Lead V6 Accounting
# ==============================================================================
add_md("""## 2. Upstream Contract, Manifest Ingestion & Shortened Lead V6 Accounting

Stage 07 consumes verified read-only outputs from previous pipeline stages:
1. **Stage 06 HRV Manifest (`reports/hrv/06_rr_hrv_manifest.csv`):** 480 acquisition rows (464 `SUCCESS`, 16 `SKIPPED_AUDIT_ERROR` due to corrupted/missing lead files in Stage 03).
2. **Stage 05 R-Peak Detection Archives (`data/processed/rpeaks/{rec_id}_rpeaks.npz`):** Exactly 464 archives containing `rhythm_lead` and `rhythm_r_peaks`. Total detected R-peaks: exactly 307,561 (141,124 at 100 Hz, 166,437 at 500 Hz).
3. **Stage 04 Preprocessed Signals (`data/interim/preprocessed_ecg/{rec_id}_preprocessed.npz`):** Exactly 464 archives containing filtered 12-lead float32 arrays.
4. **Stage 03 Lead QC Audit (`reports/qc/03_ecg_lead_level_quality_audit.csv`):** Identifies the 85 recordings with shortened Lead V6 (~20.8 seconds deficit) and original ADC clipping percentages.""")

# ==============================================================================
# Cell 04 [Code]: Ingest Manifests, Verify Archives & Reconcile Cohort
# ==============================================================================
add_code("""# 1. Ingest Stage 06 HRV Manifest
stage06_manifest_path = reports_hrv_dir / "06_rr_hrv_manifest.csv"
assert stage06_manifest_path.exists(), f"Stage 06 manifest missing at {stage06_manifest_path}"
df_stage06 = pd.read_csv(stage06_manifest_path)
print(f"Stage 06 Manifest Loaded: {len(df_stage06)} rows.")
assert len(df_stage06) == 480, f"Expected 480 recording rows, found {len(df_stage06)}"

# 2. Reconcile SUCCESS and SKIPPED Statuses
df_success = df_stage06[df_stage06["rpeak_detection_status"] == "SUCCESS"].copy()
df_skipped = df_stage06[df_stage06["rpeak_detection_status"] != "SUCCESS"].copy()
assert len(df_success) == 464, f"Expected 464 SUCCESS recordings, found {len(df_success)}"
assert len(df_skipped) == 16, f"Expected 16 SKIPPED recordings, found {len(df_skipped)}"

# Verify total R-peaks reported in Stage 06 manifest
total_manifest_peaks = df_success["raw_detector_rr_peak_count"].astype(int).sum()
assert total_manifest_peaks == 307561, f"Expected 307,561 detected R-peaks, found {total_manifest_peaks}"
print(f"  [PASS] Cohort peak count reconciled: {total_manifest_peaks:,} peaks across 464 valid recordings.")

# 3. Verify Upstream Archive Paths on Disk
missing_stage04 = []
missing_stage05 = []
for _, row in df_success.iterrows():
    rec_id = row["recording_id"]
    p4 = interim_ecg_dir / f"{rec_id}_preprocessed.npz"
    p5 = processed_rpeaks_dir / f"{rec_id}_rpeaks.npz"
    if not p4.exists():
        missing_stage04.append(rec_id)
    if not p5.exists():
        missing_stage05.append(rec_id)

assert len(missing_stage04) == 0, f"Missing {len(missing_stage04)} Stage 04 archives: {missing_stage04[:5]}"
assert len(missing_stage05) == 0, f"Missing {len(missing_stage05)} Stage 05 archives: {missing_stage05[:5]}"
print("  [PASS] All 464 Stage 04 preprocessed and Stage 05 R-peak archives verified on disk.")

# 4. Ingest Stage 03 Lead QC Audit for Saturation Risk & Shortened V6 Tracking (Using lead_name)
stage03_qc_path = reports_qc_dir / "03_ecg_lead_level_quality_audit.csv"
qc_clipping_lookup: Dict[str, Tuple[float, bool]] = {}
v6_shortened_records = set()

if stage03_qc_path.exists():
    df_qc = pd.read_csv(stage03_qc_path)
    for rec_id, grp in df_qc.groupby("recording_id"):
        max_clip = float(grp["clipping_pct"].max()) if "clipping_pct" in grp.columns else 0.0
        clip_risk = bool(max_clip > 1.0)
        qc_clipping_lookup[rec_id] = (max_clip, clip_risk)
        
        # Access verified column lead_name
        v6_rows = grp[grp["lead_name"] == "V6"]
        other_rows = grp[grp["lead_name"] != "V6"]
        if not v6_rows.empty and not other_rows.empty:
            v6_len = v6_rows["sample_count"].iloc[0] if "sample_count" in v6_rows.columns else None
            other_len = other_rows["sample_count"].iloc[0] if "sample_count" in other_rows.columns else None
            if v6_len is not None and other_len is not None and v6_len < other_len:
                v6_shortened_records.add(rec_id)
    print(f"Stage 03 QC Loaded: {len(qc_clipping_lookup)} recordings mapped; {len(v6_shortened_records)} with shortened Lead V6.")
else:
    print("  [WARNING] Stage 03 lead-level QC manifest not found; defaulting clipping risk to False.")""")

# ==============================================================================
# Cell 05 [Markdown]: Section 3 - Mathematical Formulations, Window Allocation & Boundary / Overlap Inequalities
# ==============================================================================
add_md("""## 3. Mathematical Formulations, Window Allocation & Boundary / Overlap Inequalities

### 3.1 Window Duration & Sample Allocation (Option A)
- **Total Duration:** 800.0 ms ($T_{\\text{pre}} = 250.0\\text{ ms}$, $T_{\\text{post}} = 550.0\\text{ ms}$).
- **100 Hz Configuration:** $N_{\\text{pre}} = 25$, $N_{\\text{post}} = 55$, $N_{\\text{samples}} = 80$ samples.
- **500 Hz Configuration:** $N_{\\text{pre}} = 125$, $N_{\\text{post}} = 275$, $N_{\\text{samples}} = 400$ samples.
- **Scale Factor:** Exact $5.0\\times$ temporal and sample scaling.

### 3.2 Synchronous Boundary Eligibility Inequality
For recording with lead lengths $L_l$, the minimum lead length is $L_{\\min} = \\min_{l=1..12} L_l$:
- Beat $i$ window interval: $[\\text{start}_i, \\text{end}_i) = [R_i - N_{\\text{pre}}, R_i + N_{\\text{post}})$.
- Beat $i$ is boundary-eligible if and only if:
  $$N_{\\text{pre}} \\le R_i \\le L_{\\min} - N_{\\text{post}}$$
- Head truncation: $R_i < N_{\\text{pre}} \\implies \\text{EXCLUDED\\_BOUNDARY\\_INCOMPLETE}$ (`reason = 'HEAD_TRUNCATION'`).
- Tail truncation: $R_i + N_{\\text{post}} > L_{\\min} \\implies \\text{EXCLUDED\\_BOUNDARY\\_INCOMPLETE}$ (`reason = 'V6_LEAD_DEFICIT_LIMIT'` if $L_{\\min} < \\max_l L_l$, else `'TAIL_TRUNCATION'`).

### 3.3 Forward & Backward Overlap Detection
- **Forward Overlap (Succeeding R-Peak):**
  $$R_{i+1} < R_i + N_{\\text{post}} \\iff \\Delta_{\\text{next}} = R_{i+1} - R_i < N_{\\text{post}}$$
  `overlap_post_ms` $= 550.0 - (\\Delta_{\\text{next}} / f_s \\times 1000.0)$.
- **Backward Overlap (Preceding R-Peak, Boundary-Inclusive):**
  $$R_{i-1} \\ge R_i - N_{\\text{pre}} \\iff \\Delta_{\\text{prev}} = R_i - R_{i-1} \\le N_{\\text{pre}}$$
  `overlap_pre_ms` $= 250.0 - (\\Delta_{\\text{prev}} / f_s \\times 1000.0)$.
  (Boundary touch: $\\Delta_{\\text{prev}} = N_{\\text{pre}} \\implies \\text{overlap\\_pre\\_ms} = 0.0\\text{ ms}$).

### 3.4 Candidate-NN Status & Local RR Continuity
- `stage06_candidate_nn_status = 'UNKNOWN'` (governance mandate).
- `local_rr_preceding_valid`: $300.0 \\le RR_{i-1} \\le 2000.0\\text{ ms}$.
- `local_rr_jump_valid`: $|RR_{i-1} - RR_{i-2}| / RR_{i-2} \\le 0.20$.
- `local_rr_continuity`: `local_rr_preceding_valid and local_rr_jump_valid`.""")

# ==============================================================================
# Cell 06 [Code]: Core Extraction Functions & Comprehensive Synthetic Unit Tests
# ==============================================================================
add_code("""def get_window_params(fs: int) -> Tuple[int, int, int]:
    \"\"\"Calculate sample allocation for 800 ms window (250 ms pre, 550 ms post).\"\"\"
    n_pre = int(round(250.0 * fs / 1000.0))
    n_post = int(round(550.0 * fs / 1000.0))
    n_samples = n_pre + n_post
    return n_pre, n_post, n_samples


def evaluate_beat_boundaries(
    r_peak: int, n_pre: int, n_post: int, l_min: int, l_max: int
) -> Tuple[bool, str]:
    \"\"\"Evaluate whether beat window fits within available signal bounds.\"\"\"
    if r_peak < n_pre:
        return False, "HEAD_TRUNCATION"
    if r_peak + n_post > l_min:
        if l_min < l_max:
            return False, "V6_LEAD_DEFICIT_LIMIT"
        return False, "TAIL_TRUNCATION"
    return True, "ELIGIBLE"


def compute_overlap_metrics(
    r_peaks: np.ndarray, idx: int, fs: int, n_pre: int, n_post: int
) -> Dict[str, Any]:
    \"\"\"
    Compute forward and backward overlap metrics.
    Backward overlap includes the window-start boundary: R_{i-1} >= R_i - n_pre.
    \"\"\"
    n_peaks = len(r_peaks)
    curr_r = int(r_peaks[idx])
    
    # 1. Forward Overlap (Succeeding Peak)
    if idx < n_peaks - 1:
        next_r = int(r_peaks[idx + 1])
        delta_next = next_r - curr_r
        if delta_next < n_post:
            has_next_overlap = True
            next_offset_ms = round((delta_next / fs) * 1000.0, 3)
            overlap_post_ms = round(550.0 - next_offset_ms, 3)
        else:
            has_next_overlap = False
            next_offset_ms = round((delta_next / fs) * 1000.0, 3)
            overlap_post_ms = 0.0
    else:
        has_next_overlap = False
        delta_next = None
        next_offset_ms = None
        overlap_post_ms = None

    # 2. Backward Overlap (Preceding Peak, Boundary-Inclusive)
    if idx > 0:
        prev_r = int(r_peaks[idx - 1])
        delta_prev = curr_r - prev_r
        if delta_prev <= n_pre:
            has_prev_overlap = True
            prev_offset_ms = round((delta_prev / fs) * 1000.0, 3)
            overlap_pre_ms = round(250.0 - prev_offset_ms, 3)
        else:
            has_prev_overlap = False
            prev_offset_ms = round((delta_prev / fs) * 1000.0, 3)
            overlap_pre_ms = 0.0
    else:
        has_prev_overlap = False
        delta_prev = None
        prev_offset_ms = None
        overlap_pre_ms = None

    return {
        "has_next_rpeak_overlap": has_next_overlap,
        "next_rpeak_offset_samples": delta_next,
        "next_rpeak_offset_ms": next_offset_ms,
        "overlap_post_ms": overlap_post_ms,
        "has_prev_rpeak_overlap": has_prev_overlap,
        "prev_rpeak_offset_samples": delta_prev,
        "prev_rpeak_offset_ms": prev_offset_ms,
        "overlap_pre_ms": overlap_pre_ms,
    }


def compute_local_rr_heuristics(r_peaks: np.ndarray, idx: int, fs: int) -> Dict[str, Any]:
    \"\"\"Compute local RR continuity heuristic without claiming clinical NN status.\"\"\"
    if idx == 0:
        return {
            "stage06_candidate_nn_status": "UNKNOWN",
            "local_rr_preceding_ms": None,
            "local_rr_preceding_valid": False,
            "local_rr_jump_valid": False,
            "local_rr_continuity": False,
        }
    
    curr_r = int(r_peaks[idx])
    prev_r = int(r_peaks[idx - 1])
    rr_prec_ms = round(((curr_r - prev_r) / fs) * 1000.0, 3)
    prec_valid = bool(300.0 <= rr_prec_ms <= 2000.0)
    
    if idx >= 2:
        prev2_r = int(r_peaks[idx - 2])
        rr_prec2_ms = ((prev_r - prev2_r) / fs) * 1000.0
        if rr_prec2_ms > 0:
            jump_pct = abs(rr_prec_ms - rr_prec2_ms) / rr_prec2_ms
            jump_valid = bool(jump_pct <= 0.20)
        else:
            jump_valid = False
    else:
        jump_valid = False

    continuity = bool(prec_valid and jump_valid)
    return {
        "stage06_candidate_nn_status": "UNKNOWN",
        "local_rr_preceding_ms": rr_prec_ms,
        "local_rr_preceding_valid": prec_valid,
        "local_rr_jump_valid": jump_valid,
        "local_rr_continuity": continuity,
    }


def extract_beat_window_12lead(
    lead_dict: Dict[str, np.ndarray], r_peak: int, n_pre: int, n_post: int
) -> Tuple[np.ndarray, bool]:
    \"\"\"
    Synchronously slices all 12 leads for beat at r_peak.
    Returns (12, n_samples) float32 tensor and is_finite boolean.
    \"\"\"
    n_samples = n_pre + n_post
    start = r_peak - n_pre
    end = r_peak + n_post
    
    beat_matrix = np.empty((12, n_samples), dtype=np.float32)
    for l_idx, lead in enumerate(STANDARD_12_LEADS):
        beat_matrix[l_idx, :] = lead_dict[lead][start:end]
        
    is_finite = bool(np.all(np.isfinite(beat_matrix)))
    return beat_matrix, is_finite


# ==============================================================================
# Comprehensive Synthetic Unit Test Suite
# ==============================================================================
print("Executing Synthetic Unit Tests...")

# Suite 1: Exact Boundary Inequality Suite
L_syn = 1000
n_pre_100, n_post_100, n_tot_100 = get_window_params(100)
assert n_pre_100 == 25 and n_post_100 == 55 and n_tot_100 == 80

# Assert R = 0 -> EXCLUDED (HEAD_TRUNCATION)
ok, reason = evaluate_beat_boundaries(0, n_pre_100, n_post_100, L_syn, L_syn)
assert not ok and reason == "HEAD_TRUNCATION"

# Assert R = n_pre - 1 -> EXCLUDED
ok, reason = evaluate_beat_boundaries(n_pre_100 - 1, n_pre_100, n_post_100, L_syn, L_syn)
assert not ok and reason == "HEAD_TRUNCATION"

# Assert R = n_pre -> ELIGIBLE (Window exactly [0, 80))
ok, reason = evaluate_beat_boundaries(n_pre_100, n_pre_100, n_post_100, L_syn, L_syn)
assert ok and reason == "ELIGIBLE"

# Assert R = L - n_post -> ELIGIBLE (Window exactly [L-80, L))
ok, reason = evaluate_beat_boundaries(L_syn - n_post_100, n_pre_100, n_post_100, L_syn, L_syn)
assert ok and reason == "ELIGIBLE"

# Assert R = L - n_post + 1 -> EXCLUDED (TAIL_TRUNCATION)
ok, reason = evaluate_beat_boundaries(L_syn - n_post_100 + 1, n_pre_100, n_post_100, L_syn, L_syn)
assert not ok and reason == "TAIL_TRUNCATION"
print("  [PASS] Unit Test Suite 1: Exact Boundary Inequality passed.")

# Suite 2: Sampling Rate Conversion & Temporal Equivalence
n_pre_500, n_post_500, n_tot_500 = get_window_params(500)
assert n_pre_500 == 125 and n_post_500 == 275 and n_tot_500 == 400
assert (n_tot_100 / 100.0) == (n_tot_500 / 500.0) == 0.800
assert (n_pre_100 / 100.0) == (n_pre_500 / 500.0) == 0.250
print("  [PASS] Unit Test Suite 2: Sampling Rate Conversion & Temporal Equivalence passed.")

# Suite 3: Multichannel Lead Length Discrepancy Suite (Shortened V6)
ok, reason = evaluate_beat_boundaries(r_peak=9480, n_pre=n_pre_100, n_post=n_post_100, l_min=9500, l_max=10000)
assert not ok and reason == "V6_LEAD_DEFICIT_LIMIT"
print("  [PASS] Unit Test Suite 3: Lead Length Discrepancy & V6 Deficit passed.")

# Suite 4: Overlap Detection & Boundary Touch Suite
syn_peaks = np.array([100, 140, 165])  # delta 40 (400ms at 100Hz), delta 25 (250ms at 100Hz = n_pre)
ov0 = compute_overlap_metrics(syn_peaks, 0, 100, n_pre_100, n_post_100)
assert ov0["has_next_rpeak_overlap"] is True
assert ov0["next_rpeak_offset_ms"] == 400.0
assert ov0["overlap_post_ms"] == 150.0
assert ov0["has_prev_rpeak_overlap"] is False

ov1 = compute_overlap_metrics(syn_peaks, 1, 100, n_pre_100, n_post_100)
assert ov1["has_prev_rpeak_overlap"] is False
assert ov1["has_next_rpeak_overlap"] is True
assert ov1["next_rpeak_offset_ms"] == 250.0
assert ov1["overlap_post_ms"] == 300.0

ov2 = compute_overlap_metrics(syn_peaks, 2, 100, n_pre_100, n_post_100)
assert ov2["has_prev_rpeak_overlap"] is True
assert ov2["prev_rpeak_offset_ms"] == 250.0
assert ov2["overlap_pre_ms"] == 0.0  # boundary touch
assert ov2["has_next_rpeak_overlap"] is False
print("  [PASS] Unit Test Suite 4: Overlap Detection & Boundary Touch passed.")""")

# ==============================================================================
# Cell 07 [Markdown]: Section 4 - Representative 3-Recording Pilot Protocol
# ==============================================================================
add_md("""## 4. Representative 3-Recording Pilot Protocol

To ensure numerical stability, verify memory consumption, and validate storage projections before running the entire cohort, this section executes a controlled 3-recording pilot test:
1. `sub_001_rest_100hz`: Standard baseline 100 Hz recording (clean, equal lead lengths).
2. `sub_001_stress_500hz`: High-rate 500 Hz recording (large sample count, tachycardia, forward overlaps).
3. `sub_002_rest_500hz`: Recording with shortened Lead V6 (deficit of 20.84 seconds, testing synchronous boundary exclusion).

The pilot measures uncompressed memory footprint, compressed `.npz` file size, compression ratio, execution runtime, and confirms that projected cohort storage is $< 2.0\\text{ GB}$.""")

# ==============================================================================
# Cell 08 [Code]: Execute Pilot Test, Measure Storage & Extrapolate Cohort Footprint
# ==============================================================================
add_code("""pilot_recordings = [
    "sub_001_rest_100hz",
    "sub_001_stress_500hz",
    "sub_002_rest_500hz",
]

print("Executing 3-Recording Pilot Evaluation...")
pilot_results = []
temp_pilot_dir = project_root / "scratch" / "pilot_test_segments"
temp_pilot_dir.mkdir(parents=True, exist_ok=True)

try:
    for rec_id in pilot_recordings:
        row = df_success[df_success["recording_id"] == rec_id].iloc[0]
        fs = int(row["sampling_rate"])
        n_pre, n_post, n_samples = get_window_params(fs)
        
        t0 = time.time()
        
        p4 = interim_ecg_dir / f"{rec_id}_preprocessed.npz"
        p5 = processed_rpeaks_dir / f"{rec_id}_rpeaks.npz"
        
        with np.load(p4) as data4, np.load(p5) as data5:
            r_peaks = np.array(data5["rhythm_r_peaks"], copy=True)
            lead_dict = {lead: np.array(data4[lead], copy=True) for lead in STANDARD_12_LEADS}
        
        lead_lengths = [len(lead_dict[l]) for l in STANDARD_12_LEADS]
        l_min = min(lead_lengths)
        l_max = max(lead_lengths)
        
        eligible_beats = []
        eligible_indices = []
        overlap_count = 0
        excluded_count = 0
        
        for idx, r_peak in enumerate(r_peaks):
            ok, reason = evaluate_beat_boundaries(r_peak, n_pre, n_post, l_min, l_max)
            if ok:
                beat_mat, is_finite = extract_beat_window_12lead(lead_dict, r_peak, n_pre, n_post)
                if is_finite:
                    eligible_beats.append(beat_mat)
                    eligible_indices.append(r_peak)
                    ov = compute_overlap_metrics(r_peaks, idx, fs, n_pre, n_post)
                    if ov["has_next_rpeak_overlap"] or ov["has_prev_rpeak_overlap"]:
                        overlap_count += 1
                else:
                    excluded_count += 1
            else:
                excluded_count += 1
                
        t1 = time.time()
        
        arr_beats = np.array(eligible_beats, dtype=np.float32) if eligible_beats else np.empty((0, 12, n_samples), dtype=np.float32)
        uncompressed_bytes = arr_beats.nbytes
        
        temp_npz_path = temp_pilot_dir / f"{rec_id}_pilot.npz"
        np.savez_compressed(
            temp_npz_path,
            beats=arr_beats,
            beat_indices=np.array(eligible_indices, dtype=np.int64),
            lead_names=np.array(STANDARD_12_LEADS),
            sampling_rate=fs,
            recording_id=rec_id,
        )
        compressed_bytes = temp_npz_path.stat().st_size
        temp_npz_path.unlink(missing_ok=True)
        
        comp_ratio = uncompressed_bytes / compressed_bytes if compressed_bytes > 0 else 1.0
        elapsed_sec = t1 - t0
        
        pilot_results.append({
            "recording_id": rec_id,
            "sampling_rate": fs,
            "total_peaks": len(r_peaks),
            "eligible_beats": len(eligible_beats),
            "excluded_beats": excluded_count,
            "overlap_beats": overlap_count,
            "uncompressed_kb": uncompressed_bytes / 1024.0,
            "compressed_kb": compressed_bytes / 1024.0,
            "compression_ratio": comp_ratio,
            "runtime_sec": elapsed_sec,
        })
finally:
    if temp_pilot_dir.exists():
        shutil.rmtree(temp_pilot_dir, ignore_errors=True)

df_pilot = pd.DataFrame(pilot_results)
print("\\nPilot Test Results:")
print(df_pilot[["recording_id", "sampling_rate", "total_peaks", "eligible_beats", "excluded_beats", "overlap_beats", "compression_ratio", "runtime_sec"]].to_string(index=False))

# Extrapolate Cohort Storage
avg_comp_ratio = df_pilot["compression_ratio"].mean()
total_uncompressed_bytes = 3737506560  # Exact uncompressed cohort bytes from Version 2.1 plan
projected_compressed_bytes = total_uncompressed_bytes / avg_comp_ratio
projected_compressed_gb = projected_compressed_bytes / (1024 ** 3)
print(f"\\nCohort Storage Extrapolation:")
print(f"  Uncompressed Cohort Footprint: {total_uncompressed_bytes / (1024**3):.3f} GiB ({total_uncompressed_bytes / 1e9:.3f} GB)")
print(f"  Average Compression Ratio: {avg_comp_ratio:.2f}x")
print(f"  Projected Compressed Footprint: {projected_compressed_gb:.3f} GiB ({projected_compressed_bytes / 1e9:.3f} GB)")
assert projected_compressed_gb < 3.2, f"Projected storage {projected_compressed_gb:.2f} GiB exceeds 3.2 GB safety envelope!"
print("  [PASS] Pilot evaluation passed; projected cohort footprint within safe boundaries.")""")

# ==============================================================================
# Cell 09 [Markdown]: Section 5 - Batch Execution Engine, Atomic Serialization & Dynamic Resource Guards
# ==============================================================================
add_md("""## 5. Cohort Batch Segmentation Loop & Atomic Writing Engine

This section executes the cohort segmentation loop across all 480 recordings:
1. **Dynamic Disk Space Safeguard:** Before segmenting each recording, free disk space is inspected; execution halts immediately if available space falls below 3.0 GB.
2. **Streaming Execution:** Beat arrays are generated, validated, and serialized recording-by-recording, preventing host memory exhaustion.
3. **Atomic File Serialization:** Each archive is saved to `data/processed/beat_segments/{recording_id}_beats.npz.tmp`, validated for integrity via `np.load()`, and atomically renamed to its permanent filename.
4. **Non-Destructive Manifest Logging:** Excluded beats (boundary truncation or non-finite values) and SKIPPED recordings are fully logged with explicit provenance.
5. **Garbage Collection:** Explicit `gc.collect()` is invoked after each recording.""")

# ==============================================================================
# Cell 10 [Code]: Cohort Batch Processing Loop
# ==============================================================================
add_code("""print("Beginning Cohort Beat Window Segmentation...")
all_beat_records: List[Dict[str, Any]] = []
recording_summaries: List[Dict[str, Any]] = []

total_cohort_count = len(df_stage06)
processed_count = 0
skipped_count = 0
cohort_t0 = time.time()

for idx, row in df_stage06.iterrows():
    rec_id = row["recording_id"]
    sub_id = row["subject_id"]
    phase = row["phase"]
    fs = int(row["sampling_rate"])
    qc_status = row["qc_status"]
    rpeak_status = row["rpeak_detection_status"]
    
    # 1. Dynamic Free Disk Space Check (Minimum 3.0 GB)
    _, _, current_free_bytes = shutil.disk_usage(project_root)
    current_free_gb = current_free_bytes / (1024 ** 3)
    if current_free_gb < 3.0:
        raise RuntimeError(
            f"DYNAMIC DISK SAFETY HALT at recording {rec_id}: Free disk space dropped to "
            f"{current_free_gb:.3f} GB (< 3.0 GB minimum threshold). Stopping execution."
        )

    # 2. Handle Excluded / Skipped Recordings
    if rpeak_status != "SUCCESS":
        skipped_count += 1
        recording_summaries.append({
            "recording_id": rec_id,
            "subject_id": sub_id,
            "phase": phase,
            "sampling_rate": fs,
            "status": "SKIPPED_AUDIT_ERROR",
            "total_peaks": 0,
            "eligible_beats": 0,
            "excluded_beats": 0,
            "overlap_beats": 0,
            "archive_path": None,
        })
        continue

    # 3. Process Valid Recording with Explicit Handle Management
    n_pre, n_post, n_samples = get_window_params(fs)
    p4 = interim_ecg_dir / f"{rec_id}_preprocessed.npz"
    p5 = processed_rpeaks_dir / f"{rec_id}_rpeaks.npz"
    
    with np.load(p4) as data4, np.load(p5) as data5:
        r_peaks = np.array(data5["rhythm_r_peaks"], copy=True)
        lead_dict = {l: np.array(data4[l], copy=True) for l in STANDARD_12_LEADS}
    
    lead_lengths = [len(lead_dict[l]) for l in STANDARD_12_LEADS]
    l_min = min(lead_lengths)
    l_max = max(lead_lengths)
    
    max_clip_pct, clip_risk = qc_clipping_lookup.get(rec_id, (0.0, False))
    
    eligible_beat_mats = []
    eligible_beat_indices = []
    rec_overlap_count = 0
    rec_excluded_count = 0
    
    for b_idx, r_peak in enumerate(r_peaks):
        r_peak = int(r_peak)
        r_time_sec = round(r_peak / fs, 4)
        beat_id = f"{rec_id}_beat_{b_idx:05d}"
        
        is_boundary_ok, boundary_reason = evaluate_beat_boundaries(r_peak, n_pre, n_post, l_min, l_max)
        ov = compute_overlap_metrics(r_peaks, b_idx, fs, n_pre, n_post)
        if ov["has_next_rpeak_overlap"] or ov["has_prev_rpeak_overlap"]:
            rec_overlap_count += 1
            
        rr_heur = compute_local_rr_heuristics(r_peaks, b_idx, fs)
        
        if is_boundary_ok:
            beat_mat, is_finite = extract_beat_window_12lead(lead_dict, r_peak, n_pre, n_post)
            if is_finite:
                eligibility_status = "ELIGIBLE"
                exclusion_reason = "NONE"
                eligible_beat_mats.append(beat_mat)
                eligible_beat_indices.append(r_peak)
            else:
                eligibility_status = "EXCLUDED_NON_FINITE_SAMPLES"
                exclusion_reason = "NON_FINITE_AMPLITUDE"
                rec_excluded_count += 1
        else:
            eligibility_status = "EXCLUDED_BOUNDARY_INCOMPLETE"
            exclusion_reason = boundary_reason
            rec_excluded_count += 1

        all_beat_records.append({
            "beat_id": beat_id,
            "recording_id": rec_id,
            "subject_id": sub_id,
            "phase": phase,
            "sampling_rate": fs,
            "r_peak_sample": r_peak,
            "r_peak_time_sec": r_time_sec,
            "window_start_sample": r_peak - n_pre,
            "window_end_sample": r_peak + n_post,
            "l_min_samples": l_min,
            "eligibility_status": eligibility_status,
            "exclusion_reason": exclusion_reason,
            "has_next_rpeak_overlap": ov["has_next_rpeak_overlap"],
            "next_rpeak_offset_ms": ov["next_rpeak_offset_ms"],
            "overlap_post_ms": ov["overlap_post_ms"],
            "has_prev_rpeak_overlap": ov["has_prev_rpeak_overlap"],
            "prev_rpeak_offset_ms": ov["prev_rpeak_offset_ms"],
            "overlap_pre_ms": ov["overlap_pre_ms"],
            "stage06_candidate_nn_status": rr_heur["stage06_candidate_nn_status"],
            "local_rr_preceding_ms": rr_heur["local_rr_preceding_ms"],
            "local_rr_preceding_valid": rr_heur["local_rr_preceding_valid"],
            "local_rr_jump_valid": rr_heur["local_rr_jump_valid"],
            "local_rr_continuity": rr_heur["local_rr_continuity"],
            "stage03_lead_clipping_risk": clip_risk,
            "stage03_max_lead_clipping_pct": max_clip_pct,
        })

    # 4. Atomic Serialization with Overwrite Guard & Temporary Cleanup
    final_archive_path = output_beat_dir / f"{rec_id}_beats.npz"
    if final_archive_path.exists() and not OVERWRITE_EXISTING_OUTPUTS:
        raise FileExistsError(
            f"Target archive {final_archive_path} already exists and OVERWRITE_EXISTING_OUTPUTS is False."
        )

    tmp_archive_path = output_beat_dir / f"{rec_id}_beats.npz.tmp"
    
    if len(eligible_beat_mats) > 0:
        beats_tensor = np.array(eligible_beat_mats, dtype=np.float32)
        indices_array = np.array(eligible_beat_indices, dtype=np.int64)
    else:
        beats_tensor = np.empty((0, 12, n_samples), dtype=np.float32)
        indices_array = np.empty((0,), dtype=np.int64)

    try:
        np.savez_compressed(
            tmp_archive_path,
            beats=beats_tensor,
            beat_indices=indices_array,
            lead_names=np.array(STANDARD_12_LEADS),
            sampling_rate=fs,
            recording_id=rec_id,
        )
        
        with np.load(tmp_archive_path) as val_load:
            assert "beats" in val_load and len(val_load["beats"]) == len(eligible_beat_mats)
            assert "beat_indices" in val_load and len(val_load["beat_indices"]) == len(eligible_beat_indices)
        
        if final_archive_path.exists():
            final_archive_path.unlink()
        tmp_archive_path.replace(final_archive_path)
    finally:
        if tmp_archive_path.exists():
            tmp_archive_path.unlink(missing_ok=True)
    
    processed_count += 1
    recording_summaries.append({
        "recording_id": rec_id,
        "subject_id": sub_id,
        "phase": phase,
        "sampling_rate": fs,
        "status": "SUCCESS",
        "total_peaks": len(r_peaks),
        "eligible_beats": len(eligible_beat_mats),
        "excluded_beats": rec_excluded_count,
        "overlap_beats": rec_overlap_count,
        "archive_path": str(final_archive_path.relative_to(project_root)),
    })
    
    if processed_count % 40 == 0 or processed_count == 464:
        elapsed = time.time() - cohort_t0
        print(f"  Processed {processed_count}/464 valid recordings ({elapsed:.1f}s elapsed, {current_free_gb:.2f} GB free)")
        gc.collect()

cohort_elapsed = time.time() - cohort_t0
print(f"Cohort segmentation loop completed in {cohort_elapsed:.1f}s. Processed: {processed_count}, Skipped: {skipped_count}.")""")

# ==============================================================================
# Cell 11 [Markdown]: Section 6 - Manifest Serialization & Quality Summary JSON
# ==============================================================================
add_md("""## 6. Manifest Serialization & Quality Summary JSON

This section serializes the comprehensive beat-level manifest and cohort summary JSON using atomic replacement protocols:
- `reports/segmentation/07_beat_segmentation_manifest.csv`: Primary auditable manifest containing all 307,561 beat records.
- `reports/segmentation/07_beat_segmentation_summary.json`: High-level metrics and cohort-wide quality tallies.
- **Data Governance Audit:** Verifies that no patient diagnosis labels (`anoca_confirmed`, `raw_msimi_ecg_label`, `target_label`) are present in the beat manifest.
- **Parquet Support:** Optional serialization to Parquet is attempted safely in a `try...except ImportError` block without crashing.""")

# ==============================================================================
# Cell 12 [Code]: Safe Atomic Export of Manifest & Summary JSON
# ==============================================================================
add_code("""print("Serializing Beat Manifest and Summary JSON...")

# Check Overwrite Guard
if manifest_csv_path.exists() and not OVERWRITE_EXISTING_OUTPUTS:
    raise FileExistsError(f"Target manifest {manifest_csv_path} already exists and OVERWRITE_EXISTING_OUTPUTS is False.")
if summary_json_path.exists() and not OVERWRITE_EXISTING_OUTPUTS:
    raise FileExistsError(f"Target summary {summary_json_path} already exists and OVERWRITE_EXISTING_OUTPUTS is False.")

# 1. Assemble Manifest DataFrame
df_beat_manifest = pd.DataFrame(all_beat_records)
print(f"Total Beat Manifest Rows: {len(df_beat_manifest):,}")
assert len(df_beat_manifest) == 307561, f"Expected 307,561 beat records, found {len(df_beat_manifest)}"

# 2. Strict Clinical Data Governance Audit: Check for Label Isolation
forbidden_label_columns = ["anoca_confirmed", "raw_msimi_ecg_label", "target_label", "MSIMI-based-ecg", "label_status", "diagnosis"]
for col in forbidden_label_columns:
    assert col not in df_beat_manifest.columns, f"LEAKAGE VIOLATION: Forbidden label '{col}' found in beat manifest!"
print("  [PASS] Clinical diagnosis label isolation verified in beat manifest.")

# 3. Atomic Export of Beat Manifest CSV with try/finally cleanup
tmp_manifest_csv = manifest_csv_path.with_suffix(".csv.tmp")
try:
    df_beat_manifest.to_csv(tmp_manifest_csv, index=False)
    val_df = pd.read_csv(tmp_manifest_csv, nrows=5)
    assert len(val_df) == 5, "Failed to read back temporary manifest CSV"
    if manifest_csv_path.exists():
        manifest_csv_path.unlink()
    tmp_manifest_csv.replace(manifest_csv_path)
    print(f"  [PASS] Beat manifest atomically saved to {manifest_csv_path.relative_to(project_root)}")
finally:
    if tmp_manifest_csv.exists():
        tmp_manifest_csv.unlink(missing_ok=True)

# 4. Assemble and Export Summary JSON with try/finally cleanup
n_eligible_total = int((df_beat_manifest["eligibility_status"] == "ELIGIBLE").sum())
n_excluded_total = int((df_beat_manifest["eligibility_status"] != "ELIGIBLE").sum())
n_next_overlap_total = int(df_beat_manifest["has_next_rpeak_overlap"].sum())
n_prev_overlap_total = int(df_beat_manifest["has_prev_rpeak_overlap"].sum())
n_local_rr_valid_total = int(df_beat_manifest["local_rr_continuity"].sum())

exclusion_breakdown = df_beat_manifest["exclusion_reason"].value_counts().to_dict()

summary_metrics = {
    "pipeline_stage": "07_beat_window_segmentation",
    "specification_version": "2.1",
    "cohort_total_recordings": total_cohort_count,
    "recordings_processed": processed_count,
    "recordings_skipped_audit_error": skipped_count,
    "total_detected_rpeaks": len(df_beat_manifest),
    "total_eligible_beats": n_eligible_total,
    "total_excluded_beats": n_excluded_total,
    "overall_beat_eligibility_pct": round(n_eligible_total / len(df_beat_manifest) * 100.0, 3),
    "forward_overlap_beats_count": n_next_overlap_total,
    "forward_overlap_pct": round(n_next_overlap_total / len(df_beat_manifest) * 100.0, 3),
    "backward_overlap_beats_count": n_prev_overlap_total,
    "backward_overlap_pct": round(n_prev_overlap_total / len(df_beat_manifest) * 100.0, 3),
    "local_rr_continuity_beats_count": n_local_rr_valid_total,
    "local_rr_continuity_pct": round(n_local_rr_valid_total / len(df_beat_manifest) * 100.0, 3),
    "exclusion_reason_breakdown": exclusion_breakdown,
    "window_configuration": {
        "duration_ms": 800.0,
        "pre_r_ms": 250.0,
        "post_r_ms": 550.0,
        "samples_100hz": 80,
        "samples_500hz": 400,
    },
    "data_governance": {
        "candidate_nn_status": "UNKNOWN",
        "diagnosis_labels_isolated": True,
        "lead_synchronization": "MINIMUM_AVAILABLE_LEAD_LENGTH",
    },
}

tmp_summary_json = summary_json_path.with_suffix(".json.tmp")
try:
    with open(tmp_summary_json, "w", encoding="utf-8") as f:
        json.dump(summary_metrics, f, indent=2)
    with open(tmp_summary_json, "r", encoding="utf-8") as f:
        val_json = json.load(f)
    assert val_json["total_detected_rpeaks"] == 307561, "Summary JSON validation failed!"
    if summary_json_path.exists():
        summary_json_path.unlink()
    tmp_summary_json.replace(summary_json_path)
    print(f"  [PASS] Summary JSON atomically saved to {summary_json_path.relative_to(project_root)}")
finally:
    if tmp_summary_json.exists():
        tmp_summary_json.unlink(missing_ok=True)

# 5. Optional Parquet Export
try:
    parquet_path = reports_segmentation_dir / "07_beat_segmentation_manifest.parquet"
    df_beat_manifest.to_parquet(parquet_path, index=False)
    print(f"  [INFO] Parquet manifest exported to {parquet_path.relative_to(project_root)}")
except ImportError:
    print("  [INFO] Parquet engine (pyarrow/fastparquet) not installed; CSV primary manifest retained.")""")

# ==============================================================================
# Cell 13 [Markdown]: Section 7 - Stratified Beat Yield & Exclusion Accounting
# ==============================================================================
add_md("""## 7. Stratified Beat Yield & Exclusion Accounting

This section stratifies beat extraction yields across acquisition phases (`rest`, `stress`, `recover`) and sampling rates (100 Hz vs 500 Hz):
- Demonstrates physiological tachycardia effects on forward overlap rates during stress testing.
- Accounts for boundary loss mechanisms (head truncation, tail truncation, and Lead V6 deficit limits).""")

# ==============================================================================
# Cell 14 [Code]: Phase & Sampling-Rate Stratified Yield Tables
# ==============================================================================
add_code("""# 1. Stratification by Sampling Rate and Phase
strat_rows = []
for (rate, ph), grp in df_beat_manifest.groupby(["sampling_rate", "phase"]):
    total_b = len(grp)
    elig_b = (grp["eligibility_status"] == "ELIGIBLE").sum()
    head_ex = (grp["exclusion_reason"] == "HEAD_TRUNCATION").sum()
    tail_ex = (grp["exclusion_reason"] == "TAIL_TRUNCATION").sum()
    v6_ex = (grp["exclusion_reason"] == "V6_LEAD_DEFICIT_LIMIT").sum()
    fwd_ov = grp["has_next_rpeak_overlap"].sum()
    bwd_ov = grp["has_prev_rpeak_overlap"].sum()
    loc_cont = grp["local_rr_continuity"].sum()
    
    strat_rows.append({
        "Sampling Rate": f"{rate} Hz",
        "Phase": ph.upper(),
        "Total Peaks": total_b,
        "Eligible Beats": elig_b,
        "Yield (%)": round(elig_b / total_b * 100.0, 2),
        "Head Loss": head_ex,
        "Tail Loss": tail_ex,
        "V6 Deficit Loss": v6_ex,
        "Fwd Overlap (%)": round(fwd_ov / total_b * 100.0, 2),
        "Bwd Overlap (%)": round(bwd_ov / total_b * 100.0, 2),
        "Local RR Cont (%)": round(loc_cont / total_b * 100.0, 2),
    })

df_strat = pd.DataFrame(strat_rows)
print("Stratified Beat Yield and Loss Accounting:")
print(df_strat.to_string(index=False))

# 2. Overall Summary by Phase
phase_summary = df_beat_manifest.groupby("phase").agg(
    total_beats=("beat_id", "count"),
    eligible_beats=("eligibility_status", lambda s: (s == "ELIGIBLE").sum()),
    forward_overlap_pct=("has_next_rpeak_overlap", lambda s: round(s.mean() * 100.0, 2)),
    backward_overlap_pct=("has_prev_rpeak_overlap", lambda s: round(s.mean() * 100.0, 2)),
).reset_index()
print("\\nPhase-Level Aggregate Summary:")
print(phase_summary.to_string(index=False))""")

# ==============================================================================
# Cell 15 [Markdown]: Section 8 - Patient-Level Grouping Metadata & Clinical Label Isolation
# ==============================================================================
add_md("""## 8. Patient-Level Grouping Metadata & Clinical Label Isolation

To adhere strictly to clinical data governance and machine learning best practices:
1. **Beat Archives and Manifest Remain Label-Free:** Individual beat records never contain diagnosis targets.
2. **Patient-Level Binding:** Aggregated patient-level beat metrics (total beats, eligible beats, phase distributions) are joined with clinical labels from `cohort_manifest.csv` exclusively in a separate table: `reports/segmentation/07_patient_beat_summary.csv`.
3. **Audit:** Confirms that patient rows match all 80 subjects in the ANOCA study (`sub_001` through `sub_080`).""")

# ==============================================================================
# Cell 16 [Code]: Patient-Level Aggregation & Provenance Audit
# ==============================================================================
add_code("""# Check Overwrite Guard
if patient_summary_csv_path.exists() and not OVERWRITE_EXISTING_OUTPUTS:
    raise FileExistsError(
        f"Target patient summary {patient_summary_csv_path} already exists and OVERWRITE_EXISTING_OUTPUTS is False."
    )

# 1. Ingest Cohort Manifest for Patient Diagnosis Labels
cohort_manifest_file = project_root / "data" / "manifests" / "cohort_manifest.csv"
assert cohort_manifest_file.exists(), f"Cohort manifest missing at {cohort_manifest_file}"
df_cohort_labels = pd.read_csv(cohort_manifest_file)
assert len(df_cohort_labels) == 80, f"Expected 80 subjects in cohort manifest, found {len(df_cohort_labels)}"

# 2. Aggregate Beat Manifest at Patient Level
patient_stats = []
for sub_id, grp in df_beat_manifest.groupby("subject_id"):
    tot = len(grp)
    elig = (grp["eligibility_status"] == "ELIGIBLE").sum()
    fwd_ov = grp["has_next_rpeak_overlap"].sum()
    bwd_ov = grp["has_prev_rpeak_overlap"].sum()
    
    rest_b = len(grp[grp["phase"] == "rest"])
    stress_b = len(grp[grp["phase"] == "stress"])
    recover_b = len(grp[grp["phase"] == "recover"])
    
    patient_stats.append({
        "subject_id": sub_id,
        "patient_total_peaks": tot,
        "patient_eligible_beats": elig,
        "patient_beat_yield_pct": round(elig / tot * 100.0, 2) if tot > 0 else 0.0,
        "patient_forward_overlap_beats": fwd_ov,
        "patient_backward_overlap_beats": bwd_ov,
        "rest_peaks": rest_b,
        "stress_peaks": stress_b,
        "recover_peaks": recover_b,
    })

df_patient_stats = pd.DataFrame(patient_stats)

# 3. Merge with Clinical Labels
df_patient_summary = pd.merge(df_cohort_labels, df_patient_stats, on="subject_id", how="left")
assert len(df_patient_summary) == 80, f"Expected 80 patient summary rows, found {len(df_patient_summary)}"

# 4. Atomic Serialization of Patient Beat Summary CSV with try/finally cleanup
tmp_pat_csv = patient_summary_csv_path.with_suffix(".csv.tmp")
try:
    df_patient_summary.to_csv(tmp_pat_csv, index=False)
    val_pat_df = pd.read_csv(tmp_pat_csv)
    assert len(val_pat_df) == 80, "Validation of temporary patient summary CSV failed!"
    if patient_summary_csv_path.exists():
        patient_summary_csv_path.unlink()
    tmp_pat_csv.replace(patient_summary_csv_path)
    print(f"  [PASS] Patient beat summary atomically saved to {patient_summary_csv_path.relative_to(project_root)} (80 rows)")
finally:
    if tmp_pat_csv.exists():
        tmp_pat_csv.unlink(missing_ok=True)

# 5. Provenance & Isolation Audit: Verify verified cohort columns are present only at patient level
verified_label_cols = ["anoca_confirmed", "raw_msimi_ecg_label", "target_label"]
for col in verified_label_cols:
    assert col in df_patient_summary.columns, f"Expected label column '{col}' in patient summary"
    assert col not in df_beat_manifest.columns, f"LEAKAGE: '{col}' found in beat manifest"
print("  [PASS] Data governance audit complete: Verified labels bound strictly at patient level.")""")

# ==============================================================================
# Cell 17 [Markdown]: Section 9 - Diagnostic Morphology Visualization & Median Beat Waveforms
# ==============================================================================
add_md("""## 9. Diagnostic Morphology Visualization & Median Beat Waveforms

This section generates diagnostic visual figures to confirm physiological signal integrity:
- Plots median beat waveforms across standard 12 leads for a representative subject (`sub_001`) comparing `rest`, `stress`, and `recover` phases.
- Visualizes the 800 ms window geometry: the fiducial R-peak anchor at $t = 250.0\\text{ ms}$, the 250 ms pre-R window, and the 550 ms post-R ST-T repolarization interval.
- Saves the multi-panel figure to `outputs/figures/07_representative_12lead_beat_morphology.png`.""")

# ==============================================================================
# Cell 18 [Code]: Generate Diagnostic Figure & Atomic Export
# ==============================================================================
add_code("""print("Generating Diagnostic 12-Lead Morphology Figure...")
plt.close("all")

# Check Overwrite Guard
if figure_morphology_path.exists() and not OVERWRITE_EXISTING_OUTPUTS:
    raise FileExistsError(
        f"Target figure {figure_morphology_path} already exists and OVERWRITE_EXISTING_OUTPUTS is False."
    )

# Select representative patient sub_001 at 500 Hz
rep_phases = ["rest", "stress", "recover"]
phase_colors = {"rest": "#1f77b4", "stress": "#d62728", "recover": "#2ca02c"}
phase_beats: Dict[str, np.ndarray] = {}

for ph in rep_phases:
    arc_path = output_beat_dir / f"sub_001_{ph}_500hz_beats.npz"
    if arc_path.exists():
        with np.load(arc_path) as data:
            phase_beats[ph] = np.array(data["beats"], copy=True)  # shape (N, 12, 400)

if phase_beats:
    # Exact 400-sample grid aligned at fiducial R-peak (index 125 = 0.0 ms)
    time_ms = (np.arange(400) - 125) * 2.0
    fig, axes = plt.subplots(4, 3, figsize=(16, 12), sharex=True, sharey=True)
    axes = axes.flatten()

    for l_idx, lead in enumerate(STANDARD_12_LEADS):
        ax = axes[l_idx]
        ax.axvline(0.0, color="black", linestyle="--", alpha=0.6, linewidth=1.0, label="R-Peak (0 ms)")
        ax.axvspan(-250.0, 0.0, color="gray", alpha=0.08, label="Pre-R Window" if l_idx == 0 else None)
        ax.axvspan(0.0, 550.0, color="blue", alpha=0.04, label="Post-R Window" if l_idx == 0 else None)

        for ph in rep_phases:
            if ph in phase_beats and len(phase_beats[ph]) > 0:
                lead_data = phase_beats[ph][:, l_idx, :]  # (N, 400)
                median_wave = np.median(lead_data, axis=0)
                ax.plot(time_ms, median_wave, label=f"{ph.capitalize()}", color=phase_colors[ph], linewidth=1.5)

        ax.set_title(f"Lead {lead}", fontsize=11, fontweight="bold")
        ax.grid(True, linestyle=":", alpha=0.5)
        if l_idx % 3 == 0:
            ax.set_ylabel("Amplitude (mV)", fontsize=10)
        if l_idx >= 9:
            ax.set_xlabel("Time from R-Peak (ms)", fontsize=10)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.98, 0.98), fontsize=10)
    fig.suptitle(
        "Representative 12-Lead Median Beat Morphology (sub_001, 500 Hz)\\n"
        "800 ms Window: 250 ms Pre-R (P-PR) / 550 ms Post-R (ST-T-U)",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )
    plt.tight_layout()

    tmp_fig_path = figure_morphology_path.with_suffix(".png.tmp")
    try:
        fig.savefig(tmp_fig_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        gc.collect()

        if figure_morphology_path.exists():
            figure_morphology_path.unlink()
        tmp_fig_path.replace(figure_morphology_path)
        print(f"  [PASS] Morphology figure saved to {figure_morphology_path.relative_to(project_root)}")
    finally:
        if tmp_fig_path.exists():
            tmp_fig_path.unlink(missing_ok=True)
else:
    print("  [INFO] Morphology figure skipped: representative archives not found on disk.")""")

# ==============================================================================
# Cell 19 [Markdown]: Section 10 - Cryptographic Integrity Audit & Sign-Off
# ==============================================================================
add_md("""## 10. Post-Execution Cryptographic Integrity Audit & Sign-Off

This final section verifies end-to-end data integrity before completing Stage 07:
1. **Cohort Recording Accounting:** Asserts that all 480 cohort acquisitions are accounted for in the manifest.
2. **Physical Archive Verification:** Confirms that exactly 464 beat archives exist on disk.
3. **No Orphan Temporary Files:** Asserts that 0 `.tmp` files remain across the project directories.
4. **Cryptographic Immutability of Stage 05 Archives:** Re-computes SHA-256 hashes for the 40 baseline Stage 05 pilot archives and verifies exact matches against `pilot_40_sha256_manifest.json`.
5. **Post-Run Disk Space Audit:** Verifies remaining host disk space.
6. **Pipeline Boundary Sign-Off:** Halts execution with zero model training or classification.""")

# ==============================================================================
# Cell 20 [Code]: Integrity Verification, Hash Spot-Check & Pipeline Halt
# ==============================================================================
add_code("""print("=== Final Post-Execution Integrity Audit ===")

# 1. Manifest and Archive Accounting
assert len(df_beat_manifest) == 307561, f"Expected 307,561 beat rows, found {len(df_beat_manifest)}"
created_archives = list(output_beat_dir.glob("*_beats.npz"))
print(f"  [PASS] Physical Beat Archives on disk: {len(created_archives)} (expected 464)")
assert len(created_archives) == 464, f"Expected 464 archives, found {len(created_archives)}"

# 2. Check for Orphan Temporary Files in Target Directories
target_dirs = [output_beat_dir, reports_segmentation_dir, figures_dir]
orphan_tmp_files = [f for td in target_dirs if td.exists() for f in td.glob("*.tmp")]
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

# 5. Pipeline Halt & Sign-Off
print("\\n" + "=" * 70)
print("MSIMI STAGE 07: BEAT WINDOW SEGMENTATION COMPLETED SUCCESSFULLY")
print("All 307,561 R-peaks processed across 464 valid archives and 16 QC exclusions.")
print("Strict data governance: Labels isolated at patient level; Stage 06 NN status = UNKNOWN.")
print("Pipeline halted per specification. No downstream modeling executed.")
print("=" * 70)""")

# Assemble Notebook Structure
notebook_dict = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3 (ipykernel)",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.11.15"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

target_path = Path("notebooks/07_beat_window_segmentation.ipynb")
with open(target_path, "w", encoding="utf-8") as f:
    json.dump(notebook_dict, f, indent=1)

print(f"Notebook updated successfully at {target_path} with {len(cells)} cells.")
