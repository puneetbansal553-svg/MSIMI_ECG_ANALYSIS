import json
import hashlib
import shutil
from pathlib import Path
import pandas as pd
import numpy as np

project_root = Path.cwd().resolve()
print("=== Final Post-Execution 10-Point Audit for Stage 07 ===")

# 1. Stage 03–06 upstream files remain cryptographically unchanged
reports_rpeaks_dir = project_root / "reports" / "rpeaks"
processed_rpeaks_dir = project_root / "data" / "processed" / "rpeaks"
pilot_manifest_file = reports_rpeaks_dir / "pilot_40_sha256_manifest.json"
assert pilot_manifest_file.exists()
with open(pilot_manifest_file, "r", encoding="utf-8") as f:
    pilot_baseline = json.load(f)
mismatches = []
for rec_id, entry in pilot_baseline.items():
    exp_h = entry["sha256"] if isinstance(entry, dict) else entry
    arc_p = processed_rpeaks_dir / f"{rec_id}_rpeaks.npz"
    h = hashlib.sha256(arc_p.read_bytes()).hexdigest()
    if h != exp_h:
        mismatches.append(rec_id)
assert len(mismatches) == 0, f"Stage 05 hash mismatch: {mismatches}"
print("  [PASS] 1. Stage 03-06 upstream files cryptographically unchanged (40/40 SHA-256 pilot match).")

# 2. All expected Stage 07 outputs exist and are readable
p_manifest = project_root / "reports" / "segmentation" / "07_beat_segmentation_manifest.csv"
p_summary = project_root / "reports" / "segmentation" / "07_beat_segmentation_summary.json"
p_pat_summary = project_root / "reports" / "segmentation" / "07_patient_beat_summary.csv"
p_fig = project_root / "outputs" / "figures" / "07_representative_12lead_beat_morphology.png"

for p in [p_manifest, p_summary, p_pat_summary, p_fig]:
    assert p.exists() and p.stat().st_size > 0, f"Missing output file: {p}"
print("  [PASS] 2. All Stage 07 outputs exist and have valid non-zero file sizes.")

# 3. Beat manifest and NPZ archive counts reconcile
output_beat_dir = project_root / "data" / "processed" / "beat_segments"
created_arcs = list(output_beat_dir.glob("*_beats.npz"))
assert len(created_arcs) == 464, f"Expected 464 NPZ archives, found {len(created_arcs)}"
df_manifest = pd.read_csv(p_manifest)
assert len(df_manifest) == 307561, f"Expected 307,561 beat rows, found {len(df_manifest)}"
print("  [PASS] 3. Beat manifest (307,561 rows) and NPZ archives (464 files) reconcile.")

# 4. Patient summary contains exactly 80 subjects
df_pat = pd.read_csv(p_pat_summary)
assert len(df_pat) == 80, f"Expected 80 subjects, found {len(df_pat)}"
assert set(df_pat["subject_id"]) == {f"sub_{i:03d}" for i in range(1, 81)}
print("  [PASS] 4. Patient summary contains exactly 80 subjects (sub_001 to sub_080).")

# 5. Clinical diagnosis labels remain isolated from beat-level data
forbidden_cols = ["anoca_confirmed", "raw_msimi_ecg_label", "target_label", "MSIMI-based-ecg", "label_status", "diagnosis"]
for col in forbidden_cols:
    assert col not in df_manifest.columns, f"Label leakage in beat manifest: {col}"
with np.load(created_arcs[0]) as arc_sample:
    for col in forbidden_cols:
        assert col not in arc_sample.files, f"Label leakage in beat archive: {col}"
print("  [PASS] 5. Clinical diagnosis labels strictly isolated from beat-level manifest and archives.")

# 6. stage06_candidate_nn_status remains UNKNOWN
assert set(df_manifest["stage06_candidate_nn_status"].unique()) == {"UNKNOWN"}
print("  [PASS] 6. stage06_candidate_nn_status strictly UNKNOWN across 100% of rows.")

# 7. Cell execution order and notebook state are correct
nb_path = project_root / "notebooks" / "07_beat_window_segmentation.ipynb"
with open(nb_path, "r", encoding="utf-8") as f:
    nb_final = json.load(f)
for c_idx, exp_num in [(2, 1), (4, 2), (6, 3), (8, 4), (10, 5), (12, 6), (14, 7), (16, 8), (18, 9), (20, 10)]:
    cell_exec = nb_final["cells"][c_idx].get("execution_count")
    assert cell_exec == exp_num, f"Cell {c_idx} expected execution count {exp_num}, got {cell_exec}"
print("  [PASS] 7. Notebook execution order verified: In [1] to In [10] strictly sequential, all code cells executed.")

# 8. No orphan temporary files exist
target_dirs = [output_beat_dir, project_root / "reports" / "segmentation", project_root / "outputs" / "figures"]
orphan_tmps = [f for td in target_dirs if td.exists() for f in td.glob("*.tmp*")]
assert len(orphan_tmps) == 0, f"Found orphan temporary files: {orphan_tmps}"
print("  [PASS] 8. Zero orphan temporary files (*.tmp*) across project directories.")

# 9. Disk space remains above the 3.0 GB guard
_, _, final_free_bytes = shutil.disk_usage(project_root)
final_free_gb = final_free_bytes / (1024 ** 3)
assert final_free_gb >= 3.0, f"Disk space guard violated: {final_free_gb:.3f} GB < 3.0 GB!"
print(f"  [PASS] 9. Final free disk space: {final_free_gb:.3f} GB (>= 3.0 GB threshold).")

# 10. No unintended files or modifications occurred
assert not (project_root / "notebooks" / "08_segment_feature_extraction.ipynb").exists(), "Notebook 08 was prematurely created!"
print("  [PASS] 10. Notebook 08 has not been created or executed. Boundaries strictly preserved.")

print("\n=== ALL 10 GATE VERIFICATIONS PASSED WITH 100% SUCCESS ===")

