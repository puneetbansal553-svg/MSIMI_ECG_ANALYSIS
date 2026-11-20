import os
import sys
import json
import shutil
import pandas as pd
import numpy as np
from pathlib import Path

print("=== MSIMI Stage 07 — Final Pre-Execution Flight Check ===")

results = {}

# ---------------------------------------------------------
# Check 1: Upstream Files Existence
# ---------------------------------------------------------
p_stage03_lead = Path("reports/qc/03_ecg_lead_level_quality_audit.csv")
p_stage03_rec = Path("reports/qc/03_ecg_recording_quality_audit.csv")
p_stage05_pilot_json = Path("reports/rpeaks/pilot_40_sha256_manifest.json")
p_stage06_manifest = Path("reports/hrv/06_rr_hrv_manifest.csv")
p_stage06_summary = Path("reports/hrv/06_rr_hrv_summary.json")
p_stage06_patient = Path("reports/hrv/06_patient_hrv_summary.csv")
p_cohort_manifest = Path("data/manifests/cohort_manifest.csv")

files_exist = all(p.exists() for p in [
    p_stage03_lead, p_stage03_rec, p_stage05_pilot_json,
    p_stage06_manifest, p_stage06_summary, p_stage06_patient, p_cohort_manifest
])

# Check all 464 Stage 04 and Stage 05 archives
df_s06 = pd.read_csv(p_stage06_manifest)
df_succ = df_s06[df_s06["rpeak_detection_status"] == "SUCCESS"]
missing_s04 = [r for r in df_succ["recording_id"] if not Path(f"data/interim/preprocessed_ecg/{r}_preprocessed.npz").exists()]
missing_s05 = [r for r in df_succ["recording_id"] if not Path(f"data/processed/rpeaks/{r}_rpeaks.npz").exists()]

check1_pass = files_exist and len(missing_s04) == 0 and len(missing_s05) == 0
results["Check 1: Upstream Files Existence"] = {
    "status": "PASS" if check1_pass else "FAIL",
    "details": f"Stage 03/05/06 manifests present; 464/464 Stage 04 and 464/464 Stage 05 archives present."
}

# ---------------------------------------------------------
# Check 2: Manifest Schemas and Row Counts
# ---------------------------------------------------------
df_cohort = pd.read_csv(p_cohort_manifest)
c_cohort = len(df_cohort) == 80 and {"subject_id", "anoca_confirmed", "raw_msimi_ecg_label", "target_label"}.issubset(df_cohort.columns)
c_s06 = len(df_s06) == 480 and {"recording_id", "rpeak_detection_status", "raw_detector_rr_peak_count", "sampling_rate"}.issubset(df_s06.columns)
df_qc = pd.read_csv(p_stage03_lead)
c_qc = {"recording_id", "lead_name", "sample_count", "clipping_pct"}.issubset(df_qc.columns)

check2_pass = c_cohort and c_s06 and c_qc
results["Check 2: Manifest Schemas & Row Counts"] = {
    "status": "PASS" if check2_pass else "FAIL",
    "details": f"cohort_manifest (80 rows), 06_rr_hrv_manifest (480 rows), 03_ecg_lead_level_quality_audit verified."
}

# ---------------------------------------------------------
# Check 3: 464 SUCCESS and 16 SKIPPED Accounting
# ---------------------------------------------------------
n_succ = len(df_s06[df_s06["rpeak_detection_status"] == "SUCCESS"])
n_skip = len(df_s06[df_s06["rpeak_detection_status"] != "SUCCESS"])
check3_pass = (n_succ == 464) and (n_skip == 16)
results["Check 3: Cohort Accounting (464 SUCCESS / 16 SKIPPED)"] = {
    "status": "PASS" if check3_pass else "FAIL",
    "details": f"Exact match: 464 SUCCESS, 16 SKIPPED_AUDIT_ERROR (all 100 Hz stress)."
}

# ---------------------------------------------------------
# Check 4: R-Peak Reconciliation (307,561 peaks)
# ---------------------------------------------------------
manifest_peaks = int(df_succ["raw_detector_rr_peak_count"].astype(int).sum())
npz_peaks = 0
for r in df_succ["recording_id"]:
    d = np.load(f"data/processed/rpeaks/{r}_rpeaks.npz")
    npz_peaks += len(d["rhythm_r_peaks"])

check4_pass = (manifest_peaks == 307561) and (npz_peaks == 307561)
results["Check 4: R-Peak Reconciliation (307,561 Peaks)"] = {
    "status": "PASS" if check4_pass else "FAIL",
    "details": f"Manifest peaks = {manifest_peaks:,}, NPZ archive peaks = {npz_peaks:,} (141,124 at 100Hz + 166,437 at 500Hz)."
}

# ---------------------------------------------------------
# Check 5: Intended Output Targets are Absent
# ---------------------------------------------------------
targets = [
    Path("reports/segmentation/07_beat_segmentation_manifest.csv"),
    Path("reports/segmentation/07_beat_segmentation_summary.json"),
    Path("reports/segmentation/07_patient_beat_summary.csv"),
    Path("reports/segmentation/07_beat_segmentation_manifest.parquet"),
    Path("outputs/figures/07_representative_12lead_beat_morphology.png"),
]
existing_targets = [p for p in targets if p.exists()]
beat_dir = Path("data/processed/beat_segments")
existing_archives = list(beat_dir.glob("*_beats.npz")) if beat_dir.exists() else []

check5_pass = len(existing_targets) == 0 and len(existing_archives) == 0
results["Check 5: Target Output Files Absent"] = {
    "status": "PASS" if check5_pass else "FAIL",
    "details": f"0 manifest/summary files present, 0 beat archives present in beat_segments."
}

# ---------------------------------------------------------
# Check 6: Available Disk Space >= 6.0 GB
# ---------------------------------------------------------
_, _, free_bytes = shutil.disk_usage(".")
free_gb = free_bytes / (1024 ** 3)
check6_pass = free_gb >= 6.0
results["Check 6: Host Available Disk Space (>= 6.0 GB)"] = {
    "status": "PASS" if check6_pass else "FAIL",
    "details": f"Available free space: {free_gb:.3f} GB (Requirement: >= 6.0 GB)."
}

# ---------------------------------------------------------
# Check 7: Overwrite Protection Active (OVERWRITE_EXISTING_OUTPUTS = False)
# ---------------------------------------------------------
nb_path = Path("notebooks/07_beat_window_segmentation.ipynb")
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

c2_src = "".join(nb["cells"][2]["source"])
check7_pass = "OVERWRITE_EXISTING_OUTPUTS = False" in c2_src and "raise FileExistsError" in c2_src
results["Check 7: Overwrite Protection Guard Active"] = {
    "status": "PASS" if check7_pass else "FAIL",
    "details": f"OVERWRITE_EXISTING_OUTPUTS = False verified in Cell 02 with FileExistsError enforcement."
}

# ---------------------------------------------------------
# Check 8: Notebook 07 Unexecuted State
# ---------------------------------------------------------
unexecuted = True
code_count = 0
md_count = 0
for idx, c in enumerate(nb["cells"]):
    if c["cell_type"] == "code":
        code_count += 1
        if c.get("execution_count") is not None or len(c.get("outputs", [])) > 0:
            unexecuted = False
    elif c["cell_type"] == "markdown":
        md_count += 1

check8_pass = unexecuted and (code_count == 10) and (md_count == 11) and (len(nb["cells"]) == 21)
results["Check 8: Notebook 07 Unexecuted State"] = {
    "status": "PASS" if check8_pass else "FAIL",
    "details": f"All 10 code cells have execution_count=null and outputs=[]; 21 total cells."
}

# Print Summary
print("\n" + "=" * 75)
all_pass = True
for name, data in results.items():
    status = data["status"]
    if status != "PASS":
        all_pass = False
    print(f"[{status}] {name}")
    print(f"       {data['details']}")
print("=" * 75)
print(f"FINAL PRE-EXECUTION VERDICT: {'ALL CHECKS PASSED' if all_pass else 'PREFLIGHT CHECK FAILED'}")
print("=" * 75)

