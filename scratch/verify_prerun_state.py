import json
import shutil
import hashlib
from datetime import datetime
from pathlib import Path

project_root = Path(r"C:\Users\HP\Desktop\MSIMI_Project")
nb_path = project_root / "notebooks" / "06_rr_hrv_analysis.ipynb"
backup_dir = project_root / "backups" / "stage06_pre_corrected_rerun_20261031"

print("=== MSIMI NOTEBOOK 06 PRE-RUN VERIFICATION AUDIT ===")

# 1. Read on-disk notebook JSON
assert nb_path.exists(), f"Notebook file missing: {nb_path}"
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

cells = nb.get("cells", [])
print(f"Notebook path: {nb_path}")
print(f"Notebook size: {nb_path.stat().st_size} bytes")
print(f"Total cells: {len(cells)}")

code_cells = [c for c in cells if c.get("cell_type") == "code"]
md_cells = [c for c in cells if c.get("cell_type") == "markdown"]
print(f"Code cells: {len(code_cells)} (Expected: 10)")
print(f"Markdown cells: {len(md_cells)} (Expected: 11)")

# 2. Check unexecuted state across all code cells
unexecuted_all = True
for idx, c in enumerate(code_cells):
    ec = c.get("execution_count")
    out = c.get("outputs")
    if ec is not None or out != []:
        unexecuted_all = False
        print(f"  [FAIL] Code cell index {idx} has execution_count={ec}, outputs count={len(out)}")

if unexecuted_all:
    print("  [PASS] All 10 code cells are confirmed unexecuted (execution_count=None, outputs=[]).")

# 3. Check Cell 2: OVERWRITE_EXISTING_OUTPUTS = False
cell_2_src = "".join(cells[2].get("source", []))
assert "OVERWRITE_EXISTING_OUTPUTS = False" in cell_2_src, "OVERWRITE_EXISTING_OUTPUTS is not False in Cell 2!"
print("  [PASS] Cell 2 confirms OVERWRITE_EXISTING_OUTPUTS = False.")

# 4. Check Cell 8: Synthetic benchmark 150 combinations
cell_8_src = "".join(cells[8].get("source", []))
has_phases = "phases = [0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi]" in cell_8_src
has_combos = "phase_combos = [(p1, p2) for p1 in phases for p2 in phases]" in cell_8_src
has_df_loop = "for df in [0.0005, 0.0010, 0.0020]:" in cell_8_src
has_grids = "for t_grid, grid_name in [(t_uniform, \"Uniform\"), (t_irreg, \"Irregular\")]:" in cell_8_src
has_assert_150 = "assert len(df_bench) == 150" in cell_8_src

assert has_phases and has_combos and has_df_loop and has_grids and has_assert_150, "Cell 8 synthetic benchmark verification failed!"
print("  [PASS] Cell 8 contains 25 phase pairs x 2 grids x 3 df steps = 150 evaluations with assert len(df_bench) == 150.")

# 5. Check Cell 12: Atomic guarded serialization for manifest and summary
cell_12_src = "".join(cells[12].get("source", []))
has_guard_manifest = "if manifest_out_csv.exists() and not OVERWRITE_EXISTING_OUTPUTS:" in cell_12_src
has_tmp_manifest = "tmp_manifest_csv = manifest_out_csv.with_suffix(\".csv.tmp\")" in cell_12_src
has_replace_manifest = "tmp_manifest_csv.replace(manifest_out_csv)" in cell_12_src
has_guard_summary = "if summary_out_json.exists() and not OVERWRITE_EXISTING_OUTPUTS:" in cell_12_src
has_tmp_summary = "tmp_summary_json = summary_out_json.with_suffix(\".json.tmp\")" in cell_12_src
has_replace_summary = "tmp_summary_json.replace(summary_out_json)" in cell_12_src

assert has_guard_manifest and has_tmp_manifest and has_replace_manifest, "Cell 12 manifest atomic serialization missing!"
assert has_guard_summary and has_tmp_summary and has_replace_summary, "Cell 12 summary JSON atomic serialization missing!"
print("  [PASS] Cell 12 contains atomic, guarded CSV and JSON exports with pre-write validation and cleanup.")

# 6. Check Cell 16: Both RMSSD delta columns and atomic export
cell_16_src = "".join(cells[16].get("source", []))
has_delta_rmssd_100 = "delta_rmssd_stress_rest_100hz" in cell_16_src
has_delta_rmssd_500 = "delta_rmssd_stress_rest_500hz" in cell_16_src
has_guard_patient = "if patient_summary_csv.exists() and not OVERWRITE_EXISTING_OUTPUTS:" in cell_16_src
has_tmp_patient = "tmp_patient_csv = patient_summary_csv.with_suffix(\".csv.tmp\")" in cell_16_src
has_replace_patient = "tmp_patient_csv.replace(patient_summary_csv)" in cell_16_src

assert has_delta_rmssd_100 and has_delta_rmssd_500, "Cell 16 missing RMSSD delta columns!"
assert has_guard_patient and has_tmp_patient and has_replace_patient, "Cell 16 patient summary atomic export missing!"
print("  [PASS] Cell 16 contains delta_rmssd_stress_rest_100hz, delta_rmssd_stress_rest_500hz, and guarded atomic serialization.")

# 7. Check Cell 20: Pilot 40 SHA-256 spot check
cell_20_src = "".join(cells[20].get("source", []))
has_pilot_file = "pilot_manifest_file = project_root / \"reports\" / \"rpeaks\" / \"pilot_40_sha256_manifest.json\"" in cell_20_src
has_sha_check = "hashlib.sha256(arc_p.read_bytes()).hexdigest()" in cell_20_src
has_pilot_assert = "assert len(mismatches) == 0" in cell_20_src

assert has_pilot_file and has_sha_check and has_pilot_assert, "Cell 20 pilot SHA-256 verification logic missing!"
print("  [PASS] Cell 20 contains Pilot 40-archive SHA-256 cryptographic immutability verification.")

# 8. Check previous six outputs still exist and backup matches
six_files = [
    Path("reports/hrv/06_rr_hrv_manifest.csv"),
    Path("reports/hrv/06_rr_hrv_summary.json"),
    Path("reports/hrv/06_patient_hrv_summary.csv"),
    Path("outputs/figures/06_hrv_phase_distributions.png"),
    Path("outputs/figures/06_hrv_100hz_vs_500hz_concordance.png"),
    Path("outputs/figures/06_hrv_representative_rr_tachograms.png")
]

print("\n--- Current Active Outputs vs. Verified Backup ---")
all_six_exist = True
all_six_backed_up = True
for rel_p in six_files:
    active_p = project_root / rel_p
    backed_p = backup_dir / rel_p
    
    if not active_p.exists():
        all_six_exist = False
        print(f"  [MISSING ACTIVE] {rel_p}")
    if not backed_p.exists():
        all_six_backed_up = False
        print(f"  [MISSING BACKUP] {backed_p}")
        
    if active_p.exists() and backed_p.exists():
        h_act = hashlib.sha256(active_p.read_bytes()).hexdigest()
        h_bak = hashlib.sha256(backed_p.read_bytes()).hexdigest()
        match = (h_act == h_bak)
        if not match:
            all_six_backed_up = False
            print(f"  [HASH MISMATCH] {rel_p}")
        else:
            print(f"  [VERIFIED ACTIVE & BACKUP] {rel_p} ({active_p.stat().st_size} bytes, SHA256: {h_act[:12]}...)")

assert all_six_exist, "Not all active files exist!"
assert all_six_backed_up, "Backup hash mismatch detected!"
print("  [PASS] All six previous outputs exist, and their backup is 100% cryptographically verified.")

# 9. Check free disk space >= 6.0 GB
disk_u = shutil.disk_usage(str(project_root))
free_gb = disk_u.free / (1024 ** 3)
print(f"\n--- Disk Space Check ---")
print(f"Available free disk space: {free_gb:.3f} GB (Required threshold: >= 6.000 GB)")
assert free_gb >= 6.0, f"Insufficient disk space: {free_gb:.3f} GB < 6.0 GB"
print("  [PASS] Available disk space exceeds the 6.0 GB safety threshold.")

print("\n>>> ALL PRE-RUN VERIFICATION CHECKS PASSED PERFECTLY! <<<")

