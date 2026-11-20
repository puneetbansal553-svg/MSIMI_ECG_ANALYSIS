import sys
import os
import json
import time
import io
import traceback
import shutil
from pathlib import Path

# Ensure scratch directory and matplotlib config dir exist
project_root = Path.cwd()
mpl_dir = project_root / "scratch" / ".matplotlib"
mpl_dir.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(mpl_dir)

nb_path = project_root / "notebooks" / "07_beat_window_segmentation.ipynb"
assert nb_path.exists(), f"Notebook missing at {nb_path}"

with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

print(f"Loaded {nb_path.name} with {len(nb['cells'])} cells.")
print("Starting Sequential Execution of Notebook 07...")

# Global execution namespace
global_ctx = {
    "__name__": "__main__",
    "__file__": str(nb_path),
}

class TeeStream(io.StringIO):
    def __init__(self, original_stream):
        super().__init__()
        self.original_stream = original_stream

    def write(self, s):
        self.original_stream.write(s)
        self.original_stream.flush()
        return super().write(s)

exec_count = 1
total_t0 = time.time()

for cell_idx, cell in enumerate(nb["cells"]):
    cell_type = cell.get("cell_type")
    if cell_type != "code":
        continue

    src_lines = cell.get("source", [])
    src_code = "".join(src_lines)
    first_line = src_lines[0].strip() if src_lines else ""

    print(f"\n" + "=" * 70)
    print(f"Executing Cell {cell_idx:02d} [In {exec_count}]: {first_line[:60]}...")
    print("=" * 70)

    # Capture stdout and stderr
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    capture_out = TeeStream(old_stdout)
    capture_err = TeeStream(old_stderr)
    sys.stdout = capture_out
    sys.stderr = capture_err

    t0 = time.time()
    success = False
    err_info = None

    try:
        # Execute cell code in persistent global_ctx
        exec(src_code, global_ctx)
        success = True
    except Exception as e:
        err_info = {
            "ename": type(e).__name__,
            "evalue": str(e),
            "traceback": traceback.format_exception(*sys.exc_info())
        }
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr

    t1 = time.time()
    elapsed = t1 - t0

    # Collect captured outputs
    out_text = capture_out.getvalue()
    err_text = capture_err.getvalue()

    cell["execution_count"] = exec_count
    cell_outputs = []

    if out_text:
        cell_outputs.append({
            "name": "stdout",
            "output_type": "stream",
            "text": [line + "\n" for line in out_text.splitlines()]
        })

    if err_text:
        cell_outputs.append({
            "name": "stderr",
            "output_type": "stream",
            "text": [line + "\n" for line in err_text.splitlines()]
        })

    if err_info:
        cell_outputs.append({
            "output_type": "error",
            "ename": err_info["ename"],
            "evalue": err_info["evalue"],
            "traceback": err_info["traceback"]
        })

    cell["outputs"] = cell_outputs

    # Save notebook progress on disk after each cell
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)

    if not success:
        print(f"\n[CRITICAL ERROR] Cell {cell_idx:02d} FAILED after {elapsed:.2f}s:")
        print(f"{err_info['ename']}: {err_info['evalue']}")
        print("".join(err_info["traceback"]))
        print("\nPipeline execution halted immediately. Notebook saved with failure output.")
        sys.exit(1)

    print(f"\n[PASS] Cell {cell_idx:02d} completed successfully in {elapsed:.2f}s.")
    exec_count += 1

total_elapsed = time.time() - total_t0
print("\n" + "=" * 70)
print(f"NOTEBOOK 07 SEQUENTIAL EXECUTION FULLY COMPLETED in {total_elapsed:.1f}s.")
print(f"All {exec_count - 1} code cells executed without error.")
print(f"Updated on-disk notebook: {nb_path}")
print("=" * 70)

