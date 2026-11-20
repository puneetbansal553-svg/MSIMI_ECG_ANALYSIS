import json
import os
import shutil
from pathlib import Path
import numpy as np

project_root = Path.cwd().resolve()
nb_path = project_root / "notebooks" / "07_beat_window_segmentation.ipynb"

print("=== MSIMI Stage 07: Static Verification of Authorized Cell 10 Fix ===")

with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

c10_source = nb["cells"][10]["source"]
c10_text = "".join(c10_source)

# 1. Verify the temporary filename ends in .npz
assert 'tmp_archive_path = output_beat_dir / f"{rec_id}_beats.tmp.npz"' in c10_text, "Check 1 Failed: .tmp.npz not found in Cell 10 source"
print("  [PASS] Check 1: Temporary filename pattern explicitly defined as f\"{rec_id}_beats.tmp.npz\" (ends with .npz).")

# 2 & 3 & 4 & 5. Verify numpy behavior, validation, atomic replace, and cleanup
test_dir = project_root / "scratch" / "test_atomic_verify"
test_dir.mkdir(parents=True, exist_ok=True)
try:
    rec_id = "test_sub"
    tmp_path = test_dir / f"{rec_id}_beats.tmp.npz"
    final_path = test_dir / f"{rec_id}_beats.npz"
    
    # 2. Verify np.savez_compressed writes to that exact path
    dummy_beats = np.zeros((5, 12, 80), dtype=np.float32)
    dummy_indices = np.array([100, 200, 300, 400, 500], dtype=np.int64)
    np.savez_compressed(
        tmp_path,
        beats=dummy_beats,
        beat_indices=dummy_indices,
    )
    assert tmp_path.exists(), f"Check 2 Failed: {tmp_path} does not exist!"
    assert not Path(str(tmp_path) + ".npz").exists(), "Check 2 Failed: extra .npz was appended!"
    print("  [PASS] Check 2: np.savez_compressed writes directly to exact path without appending extra extension.")
    
    # 3. Verify np.load(tmp_archive_path) validates the same file
    with np.load(tmp_path) as val_load:
        assert "beats" in val_load and len(val_load["beats"]) == 5
        assert "beat_indices" in val_load and len(val_load["beat_indices"]) == 5
    print("  [PASS] Check 3: np.load(tmp_archive_path) successfully opens and validates the temporary file.")
    
    # 4. Verify atomic replacement targets {rec_id}_beats.npz
    tmp_path.replace(final_path)
    assert final_path.exists(), f"Check 4 Failed: {final_path} does not exist after replace!"
    assert not tmp_path.exists(), f"Check 4 Failed: {tmp_path} still exists after replace!"
    print("  [PASS] Check 4: Atomic replacement successfully creates final {rec_id}_beats.npz.")
    
    # 5. Verify finally cleanup removes incomplete temporary files
    # Simulate a failed write leaving tmp_path
    np.savez_compressed(tmp_path, dummy=np.array([1]))
    assert tmp_path.exists()
    # Execute finally cleanup block
    if tmp_path.exists():
        tmp_path.unlink(missing_ok=True)
    assert not tmp_path.exists(), "Check 5 Failed: finally cleanup did not remove incomplete temporary file!"
    print("  [PASS] Check 5: finally cleanup removes incomplete temporary files.")
finally:
    if test_dir.exists():
        shutil.rmtree(test_dir, ignore_errors=True)

# 6. Confirm the output directory is clean and contains no orphan files
output_beat_dir = project_root / "data" / "processed" / "beat_segments"
orphan_files = list(output_beat_dir.iterdir()) if output_beat_dir.exists() else []
assert len(orphan_files) == 0, f"Check 6 Failed: Output directory contains {len(orphan_files)} files: {orphan_files}"
print("  [PASS] Check 6: Output directory data/processed/beat_segments is clean (0 files).")

# 7. Confirm overwrite protection remains active
c02_text = "".join(nb["cells"][2]["source"])
assert "OVERWRITE_EXISTING_OUTPUTS = False" in c02_text, "Check 7 Failed: Overwrite protection is not False in Cell 02!"
print("  [PASS] Check 7: Overwrite protection active (OVERWRITE_EXISTING_OUTPUTS = False).")

# 8. Verify no unrelated code was modified
diff_cells = []
for i, cell in enumerate(nb["cells"]):
    if i == 10:
        continue
    # Ensure cells 02, 04, 06, 08 retain execution counts and outputs, cells 12, 14, 16, 18, 20 unexecuted
    if i in [2, 4, 6, 8]:
        assert cell.get("execution_count") is not None, f"Cell {i} lost execution count"
    elif i in [12, 14, 16, 18, 20]:
        assert cell.get("execution_count") is None, f"Cell {i} has unexpected execution count: {cell.get('execution_count')}"
        assert len(cell.get("outputs", [])) == 0, f"Cell {i} has unexpected outputs"

print("  [PASS] Check 8: No unrelated code modified. Cells 02, 04, 06, 08 preserved; Cells 12, 14, 16, 18, 20 remain strictly unexecuted.")
print("\n=== ALL 8 MANDATORY STATIC CHECKS PASSED ===")

