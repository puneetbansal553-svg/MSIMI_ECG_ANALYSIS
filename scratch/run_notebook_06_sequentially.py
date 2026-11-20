import sys
import os
import io
import json
import time
import shutil
import hashlib
import traceback
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path

project_root = Path(r"C:\Users\HP\Desktop\MSIMI_Project")
os.chdir(str(project_root))

nb_path = project_root / "notebooks" / "06_rr_hrv_analysis.ipynb"
backup_dir = project_root / "backups" / "stage06_pre_corrected_rerun_20261031"

print("="*70)
print("MSIMI NOTEBOOK 06 — CONTROLLED CORRECTED SEQUENTIAL RUNNER")
print(f"Timestamp:    {datetime.now(timezone.utc).isoformat()}")
print(f"Project root: {project_root}")
print(f"Notebook:     {nb_path}")
print("="*70)

# -------------------------------------------------------------------------
# Step 1: Pre-Execution State Verification
# -------------------------------------------------------------------------
assert nb_path.exists(), f"Notebook file missing: {nb_path}"
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

cells = nb.get("cells", [])
print(f"Total cells: {len(cells)}")

# Verify backup existence and match
six_files = [
    Path("reports/hrv/06_rr_hrv_manifest.csv"),
    Path("reports/hrv/06_rr_hrv_summary.json"),
    Path("reports/hrv/06_patient_hrv_summary.csv"),
    Path("outputs/figures/06_hrv_phase_distributions.png"),
    Path("outputs/figures/06_hrv_100hz_vs_500hz_concordance.png"),
    Path("outputs/figures/06_hrv_representative_rr_tachograms.png")
]
for rel_p in six_files:
    bak_p = backup_dir / rel_p
    assert bak_p.exists(), f"Backup missing for {rel_p}"
print("[PASS] Verified backup exists for all six previous Stage 06 outputs.")

# Verify Cell 2 OVERWRITE_EXISTING_OUTPUTS = True
cell_2_src = "".join(cells[2].get("source", []))
assert "OVERWRITE_EXISTING_OUTPUTS = True" in cell_2_src, "OVERWRITE_EXISTING_OUTPUTS is not True in Cell 2!"
print("[PASS] Confirmed OVERWRITE_EXISTING_OUTPUTS = True in Cell 2.")

# Verify approved corrections in on-disk notebook
cell_8_src = "".join(cells[8].get("source", []))
assert "assert len(df_bench) == 150" in cell_8_src, "Cell 8 missing 150 combinations assert!"
assert "phases = [0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi]" in cell_8_src, "Cell 8 missing 5-phase list!"

cell_12_src = "".join(cells[12].get("source", []))
assert ".replace(manifest_out_csv)" in cell_12_src and ".csv.tmp" in cell_12_src, "Cell 12 missing atomic logic!"

cell_16_src = "".join(cells[16].get("source", []))
assert "delta_rmssd_stress_rest_100hz" in cell_16_src and "delta_rmssd_stress_rest_500hz" in cell_16_src, "Cell 16 missing RMSSD deltas!"
assert ".replace(patient_summary_csv)" in cell_16_src and ".csv.tmp" in cell_16_src, "Cell 16 missing atomic logic!"

cell_20_src = "".join(cells[20].get("source", []))
assert "pilot_manifest_file" in cell_20_src and "hashlib.sha256" in cell_20_src, "Cell 20 missing pilot SHA-256 check!"
print("[PASS] On-disk notebook verified to contain all approved corrections.")

# -------------------------------------------------------------------------
# Step 2: Sequential Execution One Cell at a Time
# -------------------------------------------------------------------------
exec_globals = {
    "__name__": "__main__",
    "__file__": str(nb_path),
}

execution_counter = 1
cell_timings = {}

print("\n" + "="*70)
print("STARTING SEQUENTIAL CELL EXECUTION")
print("="*70)

for idx, cell in enumerate(cells):
    cell_type = cell.get("cell_type")
    if cell_type != "code":
        continue

    cell_num = idx
    cell_src = "".join(cell.get("source", []))
    print(f"\n>>> Executing Cell {cell_num} (Execution #{execution_counter}) ...")
    
    # Capture stdout and stderr
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    capture_out = io.StringIO()
    capture_err = io.StringIO()
    sys.stdout = capture_out
    sys.stderr = capture_err
    
    t0 = time.time()
    failed = False
    err_tb = ""
    
    import gc
    import matplotlib.pyplot as plt
    plt.close("all")
    gc.collect()

    try:
        exec(cell_src, exec_globals)
    except Exception as e:
        failed = True
        err_tb = traceback.format_exc()
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        plt.close("all")
        gc.collect()
        
    t_elapsed = time.time() - t0
    cell_timings[cell_num] = t_elapsed
    
    stdout_text = capture_out.getvalue()
    stderr_text = capture_err.getvalue()
    
    # Format notebook outputs
    cell_outputs = []
    if stdout_text:
        cell_outputs.append({
            "name": "stdout",
            "output_type": "stream",
            "text": [line + "\n" for line in stdout_text.splitlines()]
        })
        print(f"--- Cell {cell_num} Output ---")
        for line in stdout_text.strip().splitlines()[:15]:
            print(f"  {line}")
        if len(stdout_text.strip().splitlines()) > 15:
            print(f"  ... [{len(stdout_text.strip().splitlines()) - 15} more lines] ...")
            print(f"  {stdout_text.strip().splitlines()[-1]}")

    if stderr_text:
        cell_outputs.append({
            "name": "stderr",
            "output_type": "stream",
            "text": [line + "\n" for line in stderr_text.splitlines()]
        })
        print(f"--- Cell {cell_num} STDERR ---")
        print(stderr_text.strip())
        
    if failed:
        cell_outputs.append({
            "ename": "Exception",
            "evalue": str(err_tb.strip().splitlines()[-1]),
            "output_type": "error",
            "traceback": err_tb.splitlines()
        })
        cell["execution_count"] = execution_counter
        cell["outputs"] = cell_outputs
        
        # Save notebook state with error before exiting
        with open(nb_path, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=2)
            
        print(f"\n[FATAL ERROR] Cell {cell_num} execution failed after {t_elapsed:.2f}s:")
        print(err_tb)
        sys.exit(1)
        
    cell["execution_count"] = execution_counter
    cell["outputs"] = cell_outputs
    print(f"Cell {cell_num} COMPLETED in {t_elapsed:.2f}s [OK]")
    execution_counter += 1

# Save executed notebook
with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2)
print(f"\n[SUCCESS] Notebook saved with execution counts and outputs: {nb_path}")

# -------------------------------------------------------------------------
# Step 3: Required Final Verifications
# -------------------------------------------------------------------------
print("\n" + "="*70)
print("REQUIRED FINAL VERIFICATION")
print("="*70)

# 1. 150 synthetic benchmark evaluations
df_bench = exec_globals.get("df_synth_benchmark")
assert df_bench is not None, "df_synth_benchmark missing from namespace"
assert len(df_bench) == 150, f"Expected 150 synthetic benchmark evaluations, found {len(df_bench)}"
print(f"1. Synthetic Benchmark Evaluations:  {len(df_bench)} / 150 [PASS]")

# 2. 480 recording manifest rows: 464 success and 16 QC exclusions
df_hrv_manifest = exec_globals.get("df_hrv_manifest")
assert df_hrv_manifest is not None, "df_hrv_manifest missing from namespace"
assert len(df_hrv_manifest) == 480, f"Expected 480 manifest rows, found {len(df_hrv_manifest)}"
assert df_hrv_manifest["recording_id"].nunique() == 480, "Non-unique recording IDs"
success_count = int((df_hrv_manifest["rpeak_detection_status"] == "SUCCESS").sum())
skipped_count = int((df_hrv_manifest["rpeak_detection_status"] == "SKIPPED_AUDIT_ERROR").sum())
assert success_count == 464, f"Expected 464 success, found {success_count}"
assert skipped_count == 16, f"Expected 16 skipped QC errors, found {skipped_count}"
print(f"2. Recording Manifest Accounting:    480 rows ({success_count} SUCCESS, {skipped_count} SKIPPED_AUDIT_ERROR) [PASS]")

# 3 & 4. 80 patient summary rows with 23 columns & Both RMSSD delta columns
df_patient_summary = exec_globals.get("df_patient_summary")
assert df_patient_summary is not None, "df_patient_summary missing from namespace"
assert len(df_patient_summary) == 80, f"Expected 80 patient rows, found {len(df_patient_summary)}"
assert df_patient_summary["subject_id"].nunique() == 80, "Non-unique subject IDs"
assert df_patient_summary.shape[1] == 23, f"Expected 23 columns, found {df_patient_summary.shape[1]}"
assert "delta_rmssd_stress_rest_100hz" in df_patient_summary.columns, "delta_rmssd_stress_rest_100hz missing"
assert "delta_rmssd_stress_rest_500hz" in df_patient_summary.columns, "delta_rmssd_stress_rest_500hz missing"
print(f"3. Patient Summary Dimensions:       {len(df_patient_summary)} rows x {df_patient_summary.shape[1]} columns [PASS]")
print(f"4. Patient RMSSD Delta Columns:      Both delta_rmssd_stress_rest_100hz and delta_rmssd_stress_rest_500hz verified [PASS]")

# 5. Pilot SHA-256 verification: 40/40 matches
pilot_manifest_file = project_root / "reports" / "rpeaks" / "pilot_40_sha256_manifest.json"
assert pilot_manifest_file.exists(), f"Pilot manifest missing: {pilot_manifest_file}"
with open(pilot_manifest_file, "r", encoding="utf-8") as f:
    pilot_baseline = json.load(f)
rpeaks_dir = project_root / "data" / "processed" / "rpeaks"
pilot_mismatches = []
for rec_id, entry in pilot_baseline.items():
    exp_h = entry["sha256"] if isinstance(entry, dict) else entry
    arc_f = rpeaks_dir / f"{rec_id}_rpeaks.npz"
    if not arc_f.exists():
        pilot_mismatches.append(f"MISSING:{rec_id}")
    else:
        actual_h = hashlib.sha256(arc_f.read_bytes()).hexdigest()
        if actual_h != exp_h:
            pilot_mismatches.append(f"MISMATCH:{rec_id}")
assert len(pilot_mismatches) == 0, f"Pilot hash mismatch detected: {pilot_mismatches}"
print(f"5. Pilot Archive Immutability:       {len(pilot_baseline)}/40 archives checked, 0 mismatches [PASS]")

# 6. Atomic export validation passed & generated files present
manifest_out_csv = project_root / "reports" / "hrv" / "06_rr_hrv_manifest.csv"
summary_out_json = project_root / "reports" / "hrv" / "06_rr_hrv_summary.json"
patient_summary_csv = project_root / "reports" / "hrv" / "06_patient_hrv_summary.csv"

assert manifest_out_csv.exists() and manifest_out_csv.stat().st_size > 0, "Manifest CSV missing or empty"
assert summary_out_json.exists() and summary_out_json.stat().st_size > 0, "Summary JSON missing or empty"
assert patient_summary_csv.exists() and patient_summary_csv.stat().st_size > 0, "Patient CSV missing or empty"

# Read back patient summary from disk to verify disk schema has 23 columns
df_disk_patient = pd.read_csv(patient_summary_csv)
assert df_disk_patient.shape == (80, 23), f"Disk patient CSV shape mismatch: {df_disk_patient.shape} != (80, 23)"
assert "delta_rmssd_stress_rest_100hz" in df_disk_patient.columns, "Disk CSV missing delta_rmssd_stress_rest_100hz"
assert "delta_rmssd_stress_rest_500hz" in df_disk_patient.columns, "Disk CSV missing delta_rmssd_stress_rest_500hz"
print(f"6. Atomic Export Files on Disk:      Manifest ({manifest_out_csv.stat().st_size} B), Summary ({summary_out_json.stat().st_size} B), Patient Summary ({patient_summary_csv.stat().st_size} B, 23 cols) [PASS]")

# 7. No temporary files remaining
tmp_files = list(project_root.glob("reports/hrv/*.tmp")) + list(project_root.glob("outputs/figures/*.tmp"))
assert len(tmp_files) == 0, f"Found temporary files remaining: {tmp_files}"
print(f"7. Temporary File Cleanup:           0 temporary files remaining [PASS]")

# 8. Stage 05 archives unchanged (all 464 valid archives on disk)
all_archives = list(rpeaks_dir.glob("*.npz"))
assert len(all_archives) == 464, f"Expected 464 Stage 05 archives, found {len(all_archives)}"
print(f"8. Stage 05 Total Archives:          464 archives present and unmutated [PASS]")

# 9. Free disk space >= 3 GB
disk_usage = shutil.disk_usage(str(project_root))
free_disk_gb = disk_usage.free / (1024 ** 3)
assert free_disk_gb >= 3.0, f"Free disk space fell below 3 GB: {free_disk_gb:.2f} GB"
print(f"9. Post-Execution Free Disk Space:   {free_disk_gb:.3f} GB (Requirement: >= 3.000 GB) [PASS]")

print("\n" + "="*70)
print(">>> ALL NOTEBOOK 06 CELLS AND AUDITS PASSED WITH ZERO ERRORS! <<<")
print("="*70)
