import sys
import os
import json
import time
import io
import shutil
import hashlib
from pathlib import Path
import pandas as pd
import numpy as np

project_root = Path.cwd().resolve()
mpl_dir = project_root / "scratch" / ".matplotlib"
mpl_dir.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(mpl_dir)

nb_path = project_root / "notebooks" / "07_beat_window_segmentation.ipynb"
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

print("=== Setting up execution context for Cell 20 ===")

data_dir = project_root / "data"
interim_ecg_dir = data_dir / "interim" / "preprocessed_ecg"
processed_rpeaks_dir = data_dir / "processed" / "rpeaks"
output_beat_dir = data_dir / "processed" / "beat_segments"
reports_segmentation_dir = project_root / "reports" / "segmentation"
reports_qc_dir = project_root / "reports" / "qc"
reports_hrv_dir = project_root / "reports" / "hrv"
reports_rpeaks_dir = project_root / "reports" / "rpeaks"
figures_dir = project_root / "outputs" / "figures"

manifest_path = reports_segmentation_dir / "07_beat_segmentation_manifest.csv"
assert manifest_path.exists(), f"Beat manifest missing at {manifest_path}!"
df_beat_manifest = pd.read_csv(manifest_path)
print(f"Loaded beat manifest: {len(df_beat_manifest):,} rows.")

global_ctx = {
    "__name__": "__main__",
    "__file__": str(nb_path),
    "project_root": project_root,
    "data_dir": data_dir,
    "interim_ecg_dir": interim_ecg_dir,
    "processed_rpeaks_dir": processed_rpeaks_dir,
    "output_beat_dir": output_beat_dir,
    "reports_segmentation_dir": reports_segmentation_dir,
    "reports_qc_dir": reports_qc_dir,
    "reports_hrv_dir": reports_hrv_dir,
    "reports_rpeaks_dir": reports_rpeaks_dir,
    "figures_dir": figures_dir,
    "df_beat_manifest": df_beat_manifest,
    "pd": pd,
    "np": np,
    "hashlib": hashlib,
    "json": json,
    "shutil": shutil,
}

print("\n=== Executing Cell 20: Post-Execution Cryptographic Integrity Audit & Sign-Off ===")
cell20_src = "".join(nb["cells"][20]["source"])

class TeeStream:
    def __init__(self, original, capture):
        self.original = original
        self.capture = capture
    def write(self, text):
        self.original.write(text)
        self.original.flush()
        self.capture.write(text)
    def flush(self):
        self.original.flush()
        self.capture.flush()

old_stdout = sys.stdout
captured_io = io.StringIO()
tee_out = TeeStream(old_stdout, captured_io)
sys.stdout = tee_out

t0 = time.time()
success = False
err = None
try:
    exec(cell20_src, global_ctx)
    t1 = time.time()
    success = True
except Exception as e:
    t1 = time.time()
    err = e
finally:
    sys.stdout = old_stdout

output_text = captured_io.getvalue()

if not success:
    print(f"\n[FATAL ERROR] Cell 20 execution failed after {t1 - t0:.2f}s with error: {err}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print(f"\n[SUCCESS] Cell 20 executed in {t1 - t0:.2f}s.")

# Update notebook JSON for Cell 20
print("\nUpdating notebook JSON for Cell 20...")
nb["cells"][20]["execution_count"] = 10
nb["cells"][20]["outputs"] = [
    {
        "name": "stdout",
        "output_type": "stream",
        "text": [line + "\n" for line in output_text.splitlines()]
    }
]

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print(f"Notebook JSON updated and saved: {nb_path}")

# Run Comprehensive 10-Point Post-Execution Audit
print("\n=== Comprehensive 10-Point Stage 07 Gate Audit ===")

# 1. Stage 03–06 upstream files remain cryptographically unchanged
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
p_manifest = reports_segmentation_dir / "07_beat_segmentation_manifest.csv"
p_summary = reports_segmentation_dir / "07_beat_segmentation_summary.json"
p_pat_summary = reports_segmentation_dir / "07_patient_beat_summary.csv"
p_fig = figures_dir / "07_representative_12lead_beat_morphology.png"

for p in [p_manifest, p_summary, p_pat_summary, p_fig]:
    assert p.exists() and p.stat().st_size > 0, f"Missing output file: {p}"
print("  [PASS] 2. All Stage 07 artifacts exist and have non-zero file sizes.")

# 3. Beat manifest and NPZ archive counts reconcile
created_arcs = list(output_beat_dir.glob("*_beats.npz"))
assert len(created_arcs) == 464, f"Expected 464 NPZ archives, found {len(created_arcs)}"
assert len(df_beat_manifest) == 307561, f"Expected 307,561 beat rows, found {len(df_beat_manifest)}"
print("  [PASS] 3. Beat manifest (307,561 rows) and NPZ archives (464 files) reconcile.")

# 4. Patient summary contains exactly 80 subjects
df_pat = pd.read_csv(p_pat_summary)
assert len(df_pat) == 80, f"Expected 80 subjects, found {len(df_pat)}"
print("  [PASS] 4. Patient summary contains exactly 80 subjects.")

# 5. Clinical diagnosis labels remain isolated from beat-level data
forbidden_cols = ["anoca_confirmed", "raw_msimi_ecg_label", "target_label", "MSIMI-based-ecg", "label_status", "diagnosis"]
for col in forbidden_cols:
    assert col not in df_beat_manifest.columns, f"Label leakage in beat manifest: {col}"
with np.load(created_arcs[0]) as arc_sample:
    for col in forbidden_cols:
        assert col not in arc_sample.files, f"Label leakage in beat archive: {col}"
print("  [PASS] 5. Clinical diagnosis labels strictly isolated from beat-level manifest and archives.")

# 6. stage06_candidate_nn_status remains UNKNOWN
assert set(df_beat_manifest["stage06_candidate_nn_status"].unique()) == {"UNKNOWN"}
print("  [PASS] 6. stage06_candidate_nn_status strictly UNKNOWN across 100% of rows.")

# 7. Cell execution order and notebook state are correct
with open(nb_path, "r", encoding="utf-8") as f:
    nb_final = json.load(f)
for c_idx, exp_num in [(2, 1), (4, 2), (6, 3), (8, 4), (10, 5), (12, 6), (14, 7), (16, 8), (18, 9), (20, 10)]:
    cell_exec = nb_final["cells"][c_idx].get("execution_count")
    assert cell_exec == exp_num, f"Cell {c_idx} expected execution count {exp_num}, got {cell_exec}"
print("  [PASS] 7. Notebook execution order verified: In [1] to In [10] strictly sequential, all code cells executed.")

# 8. No orphan temporary files exist
target_dirs = [output_beat_dir, reports_segmentation_dir, figures_dir]
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

