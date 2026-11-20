import json
import ast
import os
import subprocess
from pathlib import Path

nb_path = Path("notebooks/07_beat_window_segmentation.ipynb")

print("=== Starting Comprehensive Static Validation for Notebook 07 ===")

# 1. Notebook JSON validity
assert nb_path.exists(), f"Notebook file {nb_path} does not exist!"
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

print(f"1. Notebook JSON: VALID (nbformat {nb.get('nbformat')}.{nb.get('nbformat_minor')})")
assert nb.get("nbformat") == 4, "nbformat must be 4"

# 2. Cell Count & Types
cells = nb["cells"]
code_cells = [c for c in cells if c["cell_type"] == "code"]
md_cells = [c for c in cells if c["cell_type"] == "markdown"]
print(f"2. Cell Count: Total = {len(cells)}, Code = {len(code_cells)}, Markdown = {len(md_cells)}")
assert len(cells) == 21, f"Expected 21 cells, found {len(cells)}"
assert len(code_cells) == 10, f"Expected 10 code cells, found {len(code_cells)}"
assert len(md_cells) == 11, f"Expected 11 markdown cells, found {len(md_cells)}"

# 3. Python Syntax & Compilation
compile_errors = []
for i, c in enumerate(cells):
    if c["cell_type"] == "code":
        src = "".join(c["source"])
        try:
            ast.parse(src)
            compile(src, f"Cell_{i:02d}", "exec")
        except SyntaxError as e:
            compile_errors.append((i, str(e)))

if compile_errors:
    print(f"Compilation errors detected: {compile_errors}")
    assert False, "Python code compilation failed!"
else:
    print("3. Python Code Compilation: ALL 10 CODE CELLS COMPILE CLEANLY")

# 4. Execution State Checks
unexecuted_errors = []
for i, c in enumerate(cells):
    if c["cell_type"] == "code":
        if c.get("execution_count") is not None:
            unexecuted_errors.append(f"Cell {i} has execution_count={c.get('execution_count')}")
        if len(c.get("outputs", [])) > 0:
            unexecuted_errors.append(f"Cell {i} has non-empty outputs ({len(c['outputs'])} items)")

assert len(unexecuted_errors) == 0, f"Executed cells found: {unexecuted_errors}"
print("4. Execution State: ALL CODE CELLS UNEXECUTED (execution_count=None, outputs=[])")

# 5. Version 2.1 Parameter Verification
all_code = "\n".join("".join(c["source"]) for c in code_cells)

# Check parameters
assert "250.0" in all_code and "550.0" in all_code, "250ms and 550ms allocations missing"
assert "n_pre = int(round(250.0 * fs / 1000.0))" in all_code, "n_pre formula missing"
assert "n_post = int(round(550.0 * fs / 1000.0))" in all_code, "n_post formula missing"
assert "800.0" in all_code or "(250.0 + 550.0)" in all_code, "800ms window missing"
print("5. Version 2.1 Parameters: VERIFIED (800 ms, 250 ms pre, 550 ms post)")

# 6. Indexing and Overlap Inequalities
assert "delta_prev <= n_pre" in all_code or "prev_r >= curr_r - n_pre" in all_code, "Backward overlap boundary inequality missing"
assert "delta_next < n_post" in all_code, "Forward overlap inequality missing"
assert "overlap_pre_ms = round(250.0 - prev_offset_ms, 3)" in all_code, "Backward overlap depth calculation missing"
assert "overlap_post_ms = round(550.0 - next_offset_ms, 3)" in all_code, "Forward overlap depth calculation missing"
print("6. Indexing & Overlap Inequalities: VERIFIED (Boundary-inclusive backward overlap)")

# 7. 12-Lead Synchronized Slicing & Minimum Lead Length
assert "STANDARD_12_LEADS = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']" in all_code, "Standard 12 leads missing"
assert "l_min = min(lead_lengths)" in all_code, "l_min calculation missing"
assert "V6_LEAD_DEFICIT_LIMIT" in all_code, "V6 deficit reason missing"
print("7. 12-Lead Synchronization: VERIFIED (min lead length synchronous slicing)")

# 8. Candidate-NN Status Governance
assert '"stage06_candidate_nn_status": "UNKNOWN"' in all_code or "'stage06_candidate_nn_status': 'UNKNOWN'" in all_code, "UNKNOWN NN status missing"
assert "local_rr_continuity" in all_code, "local_rr_continuity heuristic missing"
assert "local_rr_preceding_valid" in all_code and "local_rr_jump_valid" in all_code, "local RR heuristic sub-metrics missing"
print("8. Candidate-NN Governance: VERIFIED (UNKNOWN status with separate local RR continuity)")

# 9. Clinical Label Isolation
assert "forbidden_label_columns = [\"anoca_confirmed\", \"MSIMI-based-ecg\", \"label_status\", \"diagnosis\"]" in all_code, "Label isolation check missing"
print("9. Clinical Label Isolation: VERIFIED (Strict isolation check in place)")

# 10. Atomic Export & Overwrite Guards
assert "OVERWRITE_EXISTING_OUTPUTS = False" in all_code, "OVERWRITE_EXISTING_OUTPUTS guard missing"
assert ".replace(" in all_code and ".tmp" in all_code, "Atomic tmp replacement missing"
print("10. Atomic Export & Overwrite Guards: VERIFIED")

# 11. Disk Safeguards
assert "free_gb < 6.0" in all_code, "6 GB initial disk check missing"
assert "current_free_gb < 3.0" in all_code, "3 GB dynamic disk check missing"
print("11. Disk Safeguards: VERIFIED (6.0 GB pre-run guard, 3.0 GB dynamic loop guard)")

# 12. Upstream Files Immutability Check
git_status = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
modified_files = git_status.stdout.strip().splitlines()
print(f"12. Git status modifications check: {len(modified_files)} modified entries found.")
for mf in modified_files:
    print(f"    {mf}")

# Verify no upstream files were changed
for mf in modified_files:
    fname = mf[3:].strip()
    if fname.startswith("data/") or fname.startswith("reports/hrv/") or fname.startswith("reports/qc/"):
        assert False, f"ILLEGAL UPSTREAM MODIFICATION: {fname}"

print("\n=== ALL 12 STATIC VALIDATION CHECKS PASSED PERFECTLY ===")
