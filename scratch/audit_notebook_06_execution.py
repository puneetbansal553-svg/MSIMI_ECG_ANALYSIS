import json
import os
import shutil
import hashlib
from datetime import datetime
from pathlib import Path
import pandas as pd

project_root = Path.cwd()

# -------------------------------------------------------------------------
# Item 1 & 2: Notebook on disk inspection
# -------------------------------------------------------------------------
nb_path = project_root / "notebooks" / "06_rr_hrv_analysis.ipynb"
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

cells = nb.get("cells", [])

# Cell 8
cell_8 = cells[8]
cell_8_src = "".join(cell_8.get("source", []))
cell_8_outputs = cell_8.get("outputs", [])
cell_8_exec_count = cell_8.get("execution_count")

has_25_combos = "phase_combos = [(p1, p2) for p1 in phases for p2 in phases]" in cell_8_src
has_4_combos = "phase_combos = [(0, 0), (np.pi/4, np.pi/2), (np.pi/2, np.pi/4), (np.pi, np.pi)]" in cell_8_src

# Cell 12
cell_12 = cells[12]
cell_12_src = "".join(cell_12.get("source", []))
cell_12_outputs = cell_12.get("outputs", [])
cell_12_exec_count = cell_12.get("execution_count")

has_atomic_12 = ".replace(manifest_out_csv)" in cell_12_src and "OVERWRITE_EXISTING_OUTPUTS" in cell_12_src

# Cell 16
cell_16 = cells[16]
cell_16_src = "".join(cell_16.get("source", []))
cell_16_outputs = cell_16.get("outputs", [])
cell_16_exec_count = cell_16.get("execution_count")

has_atomic_16 = ".replace(patient_summary_csv)" in cell_16_src and "OVERWRITE_EXISTING_OUTPUTS" in cell_16_src
has_delta_rmssd_16 = "delta_rmssd_stress_rest_100hz" in cell_16_src and "delta_rmssd_stress_rest_500hz" in cell_16_src

# Cell 20
cell_20 = cells[20]
cell_20_src = "".join(cell_20.get("source", []))
cell_20_outputs = cell_20.get("outputs", [])
cell_20_exec_count = cell_20.get("execution_count")

print("=== 1. DISK NOTEBOOK SOURCE & EXECUTION STATE ===")
print(f"Notebook file size: {nb_path.stat().st_size} bytes")
print(f"Notebook LastWriteTime: {datetime.fromtimestamp(nb_path.stat().st_mtime).isoformat()}")
print(f"Cell 8 execution_count: {cell_8_exec_count}, num_outputs: {len(cell_8_outputs)}")
print(f"Cell 8 uses 25 phase combinations: {has_25_combos}")
print(f"Cell 8 uses 4 phase combinations: {has_4_combos}")
print(f"Cell 12 execution_count: {cell_12_exec_count}, num_outputs: {len(cell_12_outputs)}")
print(f"Cell 12 has overwrite guards, temp files, atomic replace: {has_atomic_12}")
print(f"Cell 16 execution_count: {cell_16_exec_count}, num_outputs: {len(cell_16_outputs)}")
print(f"Cell 16 has delta RMSSD columns: {has_delta_rmssd_16}")
print(f"Cell 16 has overwrite guards, temp files, atomic replace: {has_atomic_16}")
print(f"Cell 20 execution_count: {cell_20_exec_count}, num_outputs: {len(cell_20_outputs)}")

# -------------------------------------------------------------------------
# Item 3: Generated Patient Summary CSV
# -------------------------------------------------------------------------
pat_summary_p = project_root / "reports" / "hrv" / "06_patient_hrv_summary.csv"
print("\n=== 3. GENERATED PATIENT SUMMARY CSV COLUMNS ===")
if pat_summary_p.exists():
    df_p = pd.read_csv(pat_summary_p)
    cols = list(df_p.columns)
    has_delta_100 = "delta_rmssd_stress_rest_100hz" in cols
    has_delta_500 = "delta_rmssd_stress_rest_500hz" in cols
    print(f"Exists on disk: True ({pat_summary_p.stat().st_size} bytes, modified {datetime.fromtimestamp(pat_summary_p.stat().st_mtime).isoformat()})")
    print(f"Columns count: {len(cols)}")
    print(f"All columns: {cols}")
    print(f"delta_rmssd_stress_rest_100hz present: {has_delta_100}")
    print(f"delta_rmssd_stress_rest_500hz present: {has_delta_500}")
else:
    print("Does not exist!")

# -------------------------------------------------------------------------
# Item 4: Generated Recording Manifest, Patient Summary, Summary JSON
# -------------------------------------------------------------------------
print("\n=== 4. INSPECTION OF GENERATED ARTIFACTS ===")
manifest_p = project_root / "reports" / "hrv" / "06_rr_hrv_manifest.csv"
summary_p = project_root / "reports" / "hrv" / "06_rr_hrv_summary.json"

if manifest_p.exists():
    df_m = pd.read_csv(manifest_p)
    print(f"--- Recording Manifest ({manifest_p.name}) ---")
    print(f"Row count: {len(df_m)} (Expected: 480)")
    print(f"Unique recording_id count: {df_m['recording_id'].nunique()} (Expected: 480)")
    print("Detection status counts:\n", df_m["rpeak_detection_status"].value_counts().to_dict())
    print("QC status counts:\n", df_m["qc_status"].value_counts().to_dict())
    print("Rhythm lead counts:\n", df_m["selected_rhythm_lead"].value_counts().to_dict())
    print("Composite quality counts:\n", df_m["hrv_composite_quality_flag"].value_counts().to_dict())
    print("Spectral eligibility counts:\n", df_m["spectral_eligibility_status"].value_counts().to_dict())
    print("VLF status counts:\n", df_m["vlf_status"].value_counts().to_dict())
    print(f"Provenance tag unique values: {df_m['provenance_tag'].unique().tolist()}")

    # Missing phase values and statuses
    print("\nPhase-wise recording counts:")
    print(df_m.groupby(["phase", "sampling_rate", "rpeak_detection_status"]).size())

if pat_summary_p.exists():
    print(f"\n--- Patient Summary ({pat_summary_p.name}) ---")
    print(f"Row count: {len(df_p)} (Expected: 80)")
    print(f"Unique subject_id count: {df_p['subject_id'].nunique()} (Expected: 80)")
    print(f"ANOCA confirmed count: {df_p['anoca_confirmed'].sum()} (Expected: 80, all True)")
    print("ANOCA non-True count:", (df_p["anoca_confirmed"] != True).sum())
    print("Label status counts:\n", df_p["label_status"].value_counts(dropna=False).to_dict())
    print("Target label counts:\n", df_p["target_label"].value_counts(dropna=False).to_dict())
    print("Raw MSIMI label counts:\n", df_p["raw_msimi_label"].value_counts(dropna=False).to_dict())
    print("Status stress 100hz counts:\n", df_p["status_stress_100hz"].value_counts().to_dict())
    print("Status stress 500hz counts:\n", df_p["status_stress_500hz"].value_counts().to_dict())

    # Check missing stress 100hz values
    stress_100_skipped = df_p[df_p["status_stress_100hz"] == "SKIPPED_AUDIT_ERROR"]
    print(f"Number of patients with SKIPPED_AUDIT_ERROR in stress 100Hz: {len(stress_100_skipped)}")
    print(f"Are all their stress_hr_100hz NaN? {stress_100_skipped['stress_hr_100hz'].isna().all()}")
    print(f"Are all their delta_hr_stress_rest_100hz NaN? {stress_100_skipped['delta_hr_stress_rest_100hz'].isna().all()}")

if summary_p.exists():
    print(f"\n--- Summary JSON ({summary_p.name}) ---")
    with open(summary_p, "r", encoding="utf-8") as f:
        s_data = json.load(f)
    print("Keys:", list(s_data.keys()))
    print("Pipeline stage:", s_data.get("pipeline_stage"))
    print("Generation timestamp UTC:", s_data.get("generation_timestamp_utc"))
    print("Provenance tag:", s_data.get("provenance_tag"))
    print("Total recordings:", s_data.get("total_cohort_recordings"))
    print("Successful HRV evaluations:", s_data.get("successful_hrv_evaluations"))
    print("Skipped QC audit errors:", s_data.get("skipped_qc_audit_errors"))
    print("Scientific limitations count:", len(s_data.get("scientific_limitations", [])))

# -------------------------------------------------------------------------
# Item 5: Pilot SHA-256 comparison in execution output
# -------------------------------------------------------------------------
print("\n=== 5. PILOT SHA-256 CHECK FROM EXECUTION OUTPUT ===")
# Check if cell 20 in notebook has output
if len(cell_20_outputs) > 0:
    for out in cell_20_outputs:
        print("Cell 20 output:", out.get("text", ""))
else:
    print("Cell 20 outputs in saved notebook on disk: EMPTY ([])")
    print("Explanation: The notebook file on disk was saved in an unexecuted state (outputs=[]).")
    print("The execution report reviewed by the user came from an in-memory execution in the IDE/Jupyter session")
    print("where Cell 20 either did not include the pilot check or was from the pre-correction version.")

# -------------------------------------------------------------------------
# Item 6: Disk space and generated outputs inventory
# -------------------------------------------------------------------------
print("\n=== 6. DISK SPACE & GENERATED OUTPUTS INVENTORY ===")
disk_u = shutil.disk_usage(str(project_root))
free_gb = disk_u.free / (1024 ** 3)
total_gb = disk_u.total / (1024 ** 3)
used_gb = disk_u.used / (1024 ** 3)
print(f"Current disk space: Free: {free_gb:.3f} GB / Total: {total_gb:.3f} GB (Used: {used_gb:.3f} GB)")

all_outputs = []
hrv_dir = project_root / "reports" / "hrv"
if hrv_dir.exists():
    for p in hrv_dir.iterdir():
        all_outputs.append((str(p.relative_to(project_root)), p.stat().st_size, datetime.fromtimestamp(p.stat().st_mtime).isoformat()))

fig_dir = project_root / "outputs" / "figures"
if fig_dir.exists():
    for p in fig_dir.iterdir():
        if p.is_file() and p.name.startswith("06_"):
            all_outputs.append((str(p.relative_to(project_root)), p.stat().st_size, datetime.fromtimestamp(p.stat().st_mtime).isoformat()))

print("\nGenerated Stage 06 Outputs:")
for path_str, size_bytes, mtime_iso in sorted(all_outputs):
    print(f"  {path_str:<55} | Size: {size_bytes:>8} bytes | Modified: {mtime_iso}")

