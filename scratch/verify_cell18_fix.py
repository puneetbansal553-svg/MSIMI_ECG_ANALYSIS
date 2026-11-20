import json
from pathlib import Path

nb_path = Path("notebooks/07_beat_window_segmentation.ipynb")
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

c18_text = "".join(nb["cells"][18]["source"])

# 1. Verify corrected filename
assert 'tmp_fig_path = figure_morphology_path.with_suffix(".tmp.png")' in c18_text, "Check 1 failed: .tmp.png not found"
p = Path("test.png").with_suffix(".tmp.png")
assert p.suffix == ".png", f"Suffix is not .png: {p.suffix}"
print("  [PASS] Check 1: Temporary filename explicitly ends with .png.")

# 2. Verify atomic replacement logic
assert "tmp_fig_path.replace(figure_morphology_path)" in c18_text, "Check 2 failed: atomic replace missing"
print("  [PASS] Check 2: Atomic replacement targets figure_morphology_path.")

# 3. Verify finally cleanup
assert "if tmp_fig_path.exists():" in c18_text and "tmp_fig_path.unlink(missing_ok=True)" in c18_text, "Check 3 failed: cleanup missing"
print("  [PASS] Check 3: Finally cleanup unlinks temporary file.")

# 4. Verify cells 02..16 preserved, cell 20 unexecuted
for i in [2, 4, 6, 8, 10, 12, 14, 16]:
    assert nb["cells"][i]["execution_count"] is not None
assert nb["cells"][18]["execution_count"] is None
assert nb["cells"][20]["execution_count"] is None
print("  [PASS] Check 4: Preceding cells preserved; Cell 18 and 20 unexecuted.")

print("\n=== STATIC VERIFICATION OF CELL 18 FIX PASSED ===")

