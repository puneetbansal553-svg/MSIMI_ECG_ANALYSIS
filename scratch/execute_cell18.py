import sys
import os
import json
import time
import io
import shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gc

project_root = Path.cwd().resolve()
mpl_dir = project_root / "scratch" / ".matplotlib"
mpl_dir.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(mpl_dir)

nb_path = project_root / "notebooks" / "07_beat_window_segmentation.ipynb"
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

print("=== Setting up execution context for Cell 18 ===")

figures_dir = project_root / "outputs" / "figures"
figures_dir.mkdir(parents=True, exist_ok=True)
figure_morphology_path = figures_dir / "07_representative_12lead_beat_morphology.png"
output_beat_dir = project_root / "data" / "processed" / "beat_segments"

STANDARD_12_LEADS = [
    "I", "II", "III", "aVR", "aVL", "aVF",
    "V1", "V2", "V3", "V4", "V5", "V6"
]

global_ctx = {
    "__name__": "__main__",
    "__file__": str(nb_path),
    "project_root": project_root,
    "figures_dir": figures_dir,
    "figure_morphology_path": figure_morphology_path,
    "output_beat_dir": output_beat_dir,
    "STANDARD_12_LEADS": STANDARD_12_LEADS,
    "OVERWRITE_EXISTING_OUTPUTS": False,
    "plt": plt,
    "np": np,
    "gc": gc,
    "Dict": dict,
}

# Verify overwrite guard
if figure_morphology_path.exists() and not global_ctx["OVERWRITE_EXISTING_OUTPUTS"]:
    raise FileExistsError(f"Target figure {figure_morphology_path} already exists and OVERWRITE_EXISTING_OUTPUTS is False!")

# Disk space check
_, _, pre_free_bytes = shutil.disk_usage(project_root)
pre_free_gb = pre_free_bytes / (1024 ** 3)
print(f"Host free disk space before Cell 18: {pre_free_gb:.3f} GB")
if pre_free_gb < 3.0:
    raise RuntimeError(f"Insufficient disk space: {pre_free_gb:.3f} GB (< 3.0 GB minimum threshold)")

print("\n=== Executing Cell 18: Diagnostic Morphology Visualization & Median Beat Waveforms ===")
cell18_src = "".join(nb["cells"][18]["source"])

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
    exec(cell18_src, global_ctx)
    t1 = time.time()
    success = True
except Exception as e:
    t1 = time.time()
    err = e
finally:
    sys.stdout = old_stdout

output_text = captured_io.getvalue()

if not success:
    print(f"\n[FATAL ERROR] Cell 18 execution failed after {t1 - t0:.2f}s with error: {err}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print(f"\n[SUCCESS] Cell 18 executed in {t1 - t0:.2f}s.")

# Update notebook JSON for Cell 18
print("\nUpdating notebook JSON for Cell 18...")
nb["cells"][18]["execution_count"] = 9
nb["cells"][18]["outputs"] = [
    {
        "name": "stdout",
        "output_type": "stream",
        "text": [line + "\n" for line in output_text.splitlines()]
    }
]

# Ensure Cell 20 remains strictly unexecuted
nb["cells"][20]["execution_count"] = None
nb["cells"][20]["outputs"] = []

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print(f"Notebook JSON updated and saved: {nb_path}")

# Post-execution validation
print("\n=== Comprehensive Validation of Cell 18 Outputs ===")
assert figure_morphology_path.exists(), f"Figure missing at {figure_morphology_path}"
fig_size_kb = figure_morphology_path.stat().st_size / 1024.0
print(f"Figure file: {figure_morphology_path.relative_to(project_root)}")
print(f"Figure size: {fig_size_kb:.2f} KB")
assert fig_size_kb > 50, f"Figure unexpectedly small ({fig_size_kb:.2f} KB)"
print("  [PASS] Morphology figure exists and has valid non-zero size.")

# Check for orphan temporary files
orphan_tmps = list(figures_dir.glob("*.tmp*"))
print(f"Orphan temporary files in {figures_dir.name}: {len(orphan_tmps)}")
assert len(orphan_tmps) == 0, f"Found orphan temporary files: {orphan_tmps}"

# Disk space check
_, _, post_free_bytes = shutil.disk_usage(project_root)
post_free_gb = post_free_bytes / (1024 ** 3)
print(f"Host free disk space: {post_free_gb:.3f} GB (minimum 3.0 GB required)")
assert post_free_gb >= 3.0, f"Disk space guard violated: {post_free_gb:.3f} GB < 3.0 GB!"

# Verify Cell 20 unexecuted
with open(nb_path, "r", encoding="utf-8") as f:
    nb_check = json.load(f)
assert nb_check["cells"][20]["execution_count"] is None, "Cell 20 execution count is not None!"
assert len(nb_check["cells"][20]["outputs"]) == 0, "Cell 20 has outputs!"
print("  [PASS] Confirmed Cell 20 remains strictly unexecuted.")

print("\n=== ALL MANDATORY CELL 18 VALIDATIONS PASSED ===")

