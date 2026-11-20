import sys
import os
import json
import time
import io
import shutil
from pathlib import Path

project_root = Path.cwd()
mpl_dir = project_root / "scratch" / ".matplotlib"
mpl_dir.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(mpl_dir)

nb_path = project_root / "notebooks" / "07_beat_window_segmentation.ipynb"
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

print("Preparing execution context for Cells 02, 04, 06...")

global_ctx = {
    "__name__": "__main__",
    "__file__": str(nb_path),
}

# Run Cell 02
exec("".join(nb["cells"][2]["source"]), global_ctx)
# Run Cell 04
exec("".join(nb["cells"][4]["source"]), global_ctx)
# Run Cell 06
exec("".join(nb["cells"][6]["source"]), global_ctx)

print("Cells 02, 04, 06 environment established successfully.")
print("\nRe-executing Cell 08 with authorized threshold (< 3.2 GB)...")

cell8_src = "".join(nb["cells"][8]["source"])

# Capture stdout
old_stdout = sys.stdout
capture_out = io.StringIO()
sys.stdout = capture_out

t0 = time.time()
try:
    exec(cell8_src, global_ctx)
    t1 = time.time()
    success = True
    err = None
except Exception as e:
    t1 = time.time()
    success = False
    err = e
finally:
    sys.stdout = old_stdout

output_text = capture_out.getvalue()
print(output_text)

if not success:
    print(f"\n[ERROR] Cell 08 failed with exception: {err}")
    sys.exit(1)

print(f"\n[PASS] Cell 08 executed and passed all assertions in {t1 - t0:.2f}s!")

# Update Cell 08 in notebook JSON
nb["cells"][8]["execution_count"] = 4
nb["cells"][8]["outputs"] = [
    {
        "name": "stdout",
        "output_type": "stream",
        "text": [line + "\n" for line in output_text.splitlines()]
    }
]

# Ensure Cells 10, 12, 14, 16, 18, 20 remain strictly unexecuted
for c_idx in [10, 12, 14, 16, 18, 20]:
    nb["cells"][c_idx]["execution_count"] = None
    nb["cells"][c_idx]["outputs"] = []

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print(f"Updated on-disk notebook: {nb_path}")

# Independent Recalculation & Storage Safety Verification
print("\n=== Independent Storage & Safety Recalculation ===")
df_pilot = global_ctx["df_pilot"]
avg_comp_ratio = df_pilot["compression_ratio"].mean()
total_uncompressed_bytes = 3737506560
proj_comp_bytes = total_uncompressed_bytes / avg_comp_ratio
proj_comp_gb = proj_comp_bytes / (1024 ** 3)
proj_comp_decimal_gb = proj_comp_bytes / 1e9

_, _, current_free_bytes = shutil.disk_usage(project_root)
current_free_gb = current_free_bytes / (1024 ** 3)
projected_remaining_free_gb = current_free_gb - proj_comp_gb

print(f"Measured Pilot Compression Ratio: {avg_comp_ratio:.4f}x")
print(f"Total Uncompressed Cohort Size:   3.481 GiB (3.738 GB)")
print(f"Projected Compressed Footprint:   {proj_comp_gb:.3f} GiB ({proj_comp_decimal_gb:.3f} GB)")
print(f"Current Host Free Disk Space:     {current_free_gb:.3f} GB")
print(f"Projected Remaining Free Disk:    {projected_remaining_free_gb:.3f} GB")
print(f"Dynamic Stop Guard in Cell 10:    >= 3.000 GB")
assert projected_remaining_free_gb > 3.0, f"Remaining free space {projected_remaining_free_gb:.2f} GB <= 3.0 GB!"
print("Safety Condition Satisfied: Projected remaining free space exceeds 3.0 GB stop guard.")

