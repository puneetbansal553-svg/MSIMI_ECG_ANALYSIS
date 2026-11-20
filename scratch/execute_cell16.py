import sys
import os
import json
import time
import io
import shutil
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

print("=== Setting up execution context for Cell 16 ===")

reports_seg_dir = project_root / "reports" / "segmentation"
manifest_path = reports_seg_dir / "07_beat_segmentation_manifest.csv"
patient_summary_csv_path = reports_seg_dir / "07_patient_beat_summary.csv"

assert manifest_path.exists(), f"Beat manifest missing at {manifest_path}!"
df_beat_manifest = pd.read_csv(manifest_path)
print(f"Loaded beat manifest: {len(df_beat_manifest):,} rows.")

global_ctx = {
    "__name__": "__main__",
    "__file__": str(nb_path),
    "project_root": project_root,
    "reports_segmentation_dir": reports_seg_dir,
    "patient_summary_csv_path": patient_summary_csv_path,
    "OVERWRITE_EXISTING_OUTPUTS": False,
    "df_beat_manifest": df_beat_manifest,
    "pd": pd,
    "np": np,
}

# Verify overwrite guard
if patient_summary_csv_path.exists() and not global_ctx["OVERWRITE_EXISTING_OUTPUTS"]:
    raise FileExistsError(f"Target {patient_summary_csv_path} already exists and OVERWRITE_EXISTING_OUTPUTS is False!")

# Disk space check
_, _, pre_free_bytes = shutil.disk_usage(project_root)
pre_free_gb = pre_free_bytes / (1024 ** 3)
print(f"Host free disk space before Cell 16: {pre_free_gb:.3f} GB")
if pre_free_gb < 3.0:
    raise RuntimeError(f"Insufficient disk space: {pre_free_gb:.3f} GB (< 3.0 GB minimum threshold)")

print("\n=== Executing Cell 16: Patient-Level Grouping Metadata & Clinical Label Isolation ===")
cell16_src = "".join(nb["cells"][16]["source"])

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
    exec(cell16_src, global_ctx)
    t1 = time.time()
    success = True
except Exception as e:
    t1 = time.time()
    err = e
finally:
    sys.stdout = old_stdout

output_text = captured_io.getvalue()

if not success:
    print(f"\n[FATAL ERROR] Cell 16 execution failed after {t1 - t0:.2f}s with error: {err}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print(f"\n[SUCCESS] Cell 16 executed in {t1 - t0:.2f}s.")

# Update notebook JSON for Cell 16
print("\nUpdating notebook JSON for Cell 16...")
nb["cells"][16]["execution_count"] = 8
nb["cells"][16]["outputs"] = [
    {
        "name": "stdout",
        "output_type": "stream",
        "text": [line + "\n" for line in output_text.splitlines()]
    }
]

# Ensure subsequent cells remain strictly unexecuted
for c_idx in [18, 20]:
    nb["cells"][c_idx]["execution_count"] = None
    nb["cells"][c_idx]["outputs"] = []

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print(f"Notebook JSON updated and saved: {nb_path}")

# Post-execution validation
print("\n=== Comprehensive Validation of Cell 16 Outputs ===")
assert patient_summary_csv_path.exists(), f"Patient summary missing at {patient_summary_csv_path}"
df_pat = pd.read_csv(patient_summary_csv_path)

# 1. Subject Count
print(f"Patient summary row count: {len(df_pat)} (expected 80)")
assert len(df_pat) == 80, f"Expected 80 subjects, got {len(df_pat)}"

# 2. Reconcile Peak Totals
tot_peaks = int(df_pat["patient_total_peaks"].sum())
elig_beats = int(df_pat["patient_eligible_beats"].sum())
fwd_ov = int(df_pat["patient_forward_overlap_beats"].sum())
bwd_ov = int(df_pat["patient_backward_overlap_beats"].sum())

print(f"Total peaks across 80 patients:     {tot_peaks:,} (expected 307,561)")
print(f"Eligible beats across 80 patients:  {elig_beats:,} (expected 304,663)")
print(f"Forward overlap across 80 patients: {fwd_ov:,} (expected 74,460)")
print(f"Backward overlap across 80 patients:{bwd_ov:,} (expected 5,231)")

assert tot_peaks == 307561, f"Peak total mismatch: {tot_peaks}"
assert elig_beats == 304663, f"Eligible beat mismatch: {elig_beats}"
assert fwd_ov == 74460, f"Forward overlap mismatch: {fwd_ov}"
assert bwd_ov == 5231, f"Backward overlap mismatch: {bwd_ov}"

# 3. Label Provenance and Isolation Check
expected_labels = ["anoca_confirmed", "raw_msimi_ecg_label", "target_label"]
for lbl in expected_labels:
    assert lbl in df_pat.columns, f"Expected label column '{lbl}' missing in patient summary!"
    assert lbl not in df_beat_manifest.columns, f"Label leakage: '{lbl}' found in beat manifest!"
print("  [PASS] Clinical diagnosis labels correctly present at patient level and absent from beat manifest.")

# 4. stage06_candidate_nn_status check
assert set(df_beat_manifest["stage06_candidate_nn_status"].unique()) == {"UNKNOWN"}
print("  [PASS] stage06_candidate_nn_status strictly UNKNOWN across cohort.")

# 5. Check for Orphan Temporary Files
orphan_tmps = list(reports_seg_dir.glob("*.tmp*"))
print(f"Orphan temporary files in {reports_seg_dir.name}: {len(orphan_tmps)}")
assert len(orphan_tmps) == 0, f"Found orphan temporary files: {orphan_tmps}"

# 6. Disk Space Check
_, _, post_free_bytes = shutil.disk_usage(project_root)
post_free_gb = post_free_bytes / (1024 ** 3)
print(f"Host free disk space: {post_free_gb:.3f} GB (minimum 3.0 GB required)")
assert post_free_gb >= 3.0, f"Disk space guard violated: {post_free_gb:.3f} GB < 3.0 GB!"

print("\n=== ALL MANDATORY CELL 16 VALIDATIONS PASSED ===")

