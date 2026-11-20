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

print("=== Setting up execution context for Cell 14 ===")

manifest_path = project_root / "reports" / "segmentation" / "07_beat_segmentation_manifest.csv"
assert manifest_path.exists(), f"Beat manifest missing at {manifest_path}!"

df_beat_manifest = pd.read_csv(manifest_path)
print(f"Loaded beat manifest from {manifest_path.relative_to(project_root)}: {len(df_beat_manifest):,} rows.")
assert len(df_beat_manifest) == 307561, f"Expected 307,561 rows, got {len(df_beat_manifest)}"

global_ctx = {
    "__name__": "__main__",
    "__file__": str(nb_path),
    "pd": pd,
    "np": np,
    "df_beat_manifest": df_beat_manifest,
    "project_root": project_root,
}

print("\n=== Executing Cell 14: Stratified Beat Yield & Exclusion Accounting ===")
cell14_src = "".join(nb["cells"][14]["source"])

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
    exec(cell14_src, global_ctx)
    t1 = time.time()
    success = True
except Exception as e:
    t1 = time.time()
    err = e
finally:
    sys.stdout = old_stdout

output_text = captured_io.getvalue()

if not success:
    print(f"\n[FATAL ERROR] Cell 14 execution failed after {t1 - t0:.2f}s with error: {err}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print(f"\n[SUCCESS] Cell 14 executed in {t1 - t0:.2f}s.")

# Update notebook JSON for Cell 14
print("\nUpdating notebook JSON for Cell 14...")
nb["cells"][14]["execution_count"] = 7
nb["cells"][14]["outputs"] = [
    {
        "name": "stdout",
        "output_type": "stream",
        "text": [line + "\n" for line in output_text.splitlines()]
    }
]

# Ensure subsequent cells remain strictly unexecuted
for c_idx in [16, 18, 20]:
    nb["cells"][c_idx]["execution_count"] = None
    nb["cells"][c_idx]["outputs"] = []

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print(f"Notebook JSON updated and saved: {nb_path}")

# Post-execution validation
print("\n=== Comprehensive Validation of Cell 14 Results ===")
df_strat = global_ctx["df_strat"]
phase_summary = global_ctx["phase_summary"]

# 1. Total Peaks Reconciliation
strat_total_peaks = df_strat["Total Peaks"].sum()
phase_total_peaks = phase_summary["total_beats"].sum()
print(f"Stratified table total peaks: {strat_total_peaks:,} (expected 307,561)")
print(f"Phase summary total peaks:    {phase_total_peaks:,} (expected 307,561)")
assert strat_total_peaks == 307561, f"Stratified total mismatch: {strat_total_peaks}"
assert phase_total_peaks == 307561, f"Phase total mismatch: {phase_total_peaks}"

# 2. Eligible Beats Reconciliation
strat_eligible = df_strat["Eligible Beats"].sum()
phase_eligible = phase_summary["eligible_beats"].sum()
print(f"Stratified table eligible beats: {strat_eligible:,} (expected 304,663)")
print(f"Phase summary eligible beats:    {phase_eligible:,} (expected 304,663)")
assert strat_eligible == 304663, f"Stratified eligible mismatch: {strat_eligible}"
assert phase_eligible == 304663, f"Phase eligible mismatch: {phase_eligible}"

# 3. Exclusions Reconciliation
head_loss = df_strat["Head Loss"].sum()
tail_loss = df_strat["Tail Loss"].sum()
v6_loss = df_strat["V6 Deficit Loss"].sum()
total_loss = head_loss + tail_loss + v6_loss
print(f"Head Loss:       {head_loss:,} (expected 261)")
print(f"Tail Loss:       {tail_loss:,} (expected 276)")
print(f"V6 Deficit Loss: {v6_loss:,} (expected 2,361)")
print(f"Total Loss:      {total_loss:,} (expected 2,898)")
assert head_loss == 261, f"Head loss mismatch: {head_loss}"
assert tail_loss == 276, f"Tail loss mismatch: {tail_loss}"
assert v6_loss == 2361, f"V6 deficit loss mismatch: {v6_loss}"
assert total_loss == 2898, f"Total loss mismatch: {total_loss}"

# 4. Clinical Label Isolation Check
forbidden_labels = ["anoca_confirmed", "raw_msimi_ecg_label", "target_label", "MSIMI-based-ecg", "label_status", "diagnosis"]
for col in forbidden_labels:
    assert col not in df_strat.columns, f"Label leakage in df_strat: {col}"
    assert col not in phase_summary.columns, f"Label leakage in phase_summary: {col}"
print("  [PASS] Clinical diagnosis label isolation verified in all stratified outputs.")

# 5. Candidate NN status verification
assert set(df_beat_manifest["stage06_candidate_nn_status"].unique()) == {"UNKNOWN"}
print("  [PASS] stage06_candidate_nn_status remains strictly UNKNOWN.")

# 6. Check for Orphan Temporary Files
reports_seg_dir = project_root / "reports" / "segmentation"
orphan_tmps = list(reports_seg_dir.glob("*.tmp*"))
print(f"Orphan temporary files in {reports_seg_dir.name}: {len(orphan_tmps)}")
assert len(orphan_tmps) == 0, f"Found orphan temporary files: {orphan_tmps}"

# 7. Disk Space Check
_, _, post_free_bytes = shutil.disk_usage(project_root)
post_free_gb = post_free_bytes / (1024 ** 3)
print(f"Host free disk space: {post_free_gb:.3f} GB (minimum 3.0 GB required)")
assert post_free_gb >= 3.0, f"Disk space guard violated: {post_free_gb:.3f} GB < 3.0 GB!"

print("\n=== ALL MANDATORY CELL 14 VALIDATIONS PASSED ===")

