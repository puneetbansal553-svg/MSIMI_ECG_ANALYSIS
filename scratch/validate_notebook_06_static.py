import json
import os
import sys
import hashlib
from pathlib import Path

def validate_notebook_static():
    project_root = Path.cwd()
    nb_path = project_root / "notebooks" / "06_rr_hrv_analysis.ipynb"
    assert nb_path.exists(), f"File not found: {nb_path}"

    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    print("=== STATIC VALIDATION: notebooks/06_rr_hrv_analysis.ipynb ===")
    
    # 1. Structure check
    assert "cells" in nb, "Missing 'cells' key in notebook JSON"
    assert "metadata" in nb, "Missing 'metadata' key in notebook JSON"
    assert nb.get("nbformat") == 4, f"Unexpected nbformat: {nb.get('nbformat')}"

    cells = nb["cells"]
    print(f"Total cells: {len(cells)}")
    assert len(cells) == 21, f"Expected exactly 21 cells, found {len(cells)}"

    code_cells = [c for c in cells if c["cell_type"] == "code"]
    md_cells = [c for c in cells if c["cell_type"] == "markdown"]
    print(f"Markdown cells: {len(md_cells)} (Expected: 11)")
    print(f"Code cells:     {len(code_cells)} (Expected: 10)")
    assert len(md_cells) == 11, f"Expected 11 markdown cells, found {len(md_cells)}"
    assert len(code_cells) == 10, f"Expected 10 code cells, found {len(code_cells)}"

    # 2. Execution state check (must be unexecuted)
    for idx, c in enumerate(code_cells):
        assert c.get("execution_count") is None, f"Code cell {idx} has non-null execution_count: {c.get('execution_count')}"
        assert c.get("outputs") == [], f"Code cell {idx} has non-empty outputs: {c.get('outputs')}"

    print("  [PASS] All code cells confirmed unexecuted (execution_count = null, outputs = []).")

    # 3. Syntax compile check
    compile_errors = []
    for idx, c in enumerate(cells):
        if c["cell_type"] == "code":
            src = "".join(c.get("source", []))
            try:
                compile(src, f"<cell_{idx}>", "exec")
            except SyntaxError as e:
                compile_errors.append((idx, str(e)))

    if compile_errors:
        print(f"  [FAIL] Syntax errors found: {compile_errors}")
        sys.exit(1)
    else:
        print(f"  [PASS] All {len(code_cells)} code cells compiled successfully with Python syntax checks.")

    # 4. Correction 1: Cell 8 (index 8 in cells) full 25 phase combo grid
    cell_8_src = "".join(cells[8]["source"])
    assert "phases = [0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi]" in cell_8_src, "Cell 8 missing 5-phase definition"
    assert "phase_combos = [(p1, p2) for p1 in phases for p2 in phases]" in cell_8_src, "Cell 8 missing 25-combo grid"
    assert "assert len(df_bench) == 150" in cell_8_src, "Cell 8 missing 150 combinations assertion"
    print("  [PASS] Correction 1 verified: Full 25-combination phase grid evaluated in synthetic benchmark (150 total evaluations).")

    # 5. Correction 2: Cell 12 (index 12 in cells) Safe Atomic Serialization
    cell_12_src = "".join(cells[12]["source"])
    assert "OVERWRITE_EXISTING_OUTPUTS" in cell_12_src, "Cell 12 missing OVERWRITE_EXISTING_OUTPUTS check"
    assert "tmp_manifest_csv = manifest_out_csv.with_suffix(\".csv.tmp\")" in cell_12_src, "Cell 12 missing manifest temp file"
    assert "tmp_summary_json = summary_out_json.with_suffix(\".json.tmp\")" in cell_12_src, "Cell 12 missing summary temp file"
    assert "tmp_manifest_csv.replace(manifest_out_csv)" in cell_12_src, "Cell 12 missing atomic replace for manifest"
    assert "tmp_summary_json.replace(summary_out_json)" in cell_12_src, "Cell 12 missing atomic replace for summary"
    assert "tmp_manifest_csv.unlink(missing_ok=True)" in cell_12_src, "Cell 12 missing manifest tmp cleanup"
    assert "tmp_summary_json.unlink(missing_ok=True)" in cell_12_src, "Cell 12 missing summary tmp cleanup"
    print("  [PASS] Correction 2 verified: Safe atomic serialization with overwrite guards implemented for manifest and summary.")

    # 6. Correction 3: Cell 16 (index 16 in cells) RMSSD Trajectories & Safe Atomic Serialization
    cell_16_src = "".join(cells[16]["source"])
    assert "delta_rmssd_stress_rest_100hz" in cell_16_src, "Cell 16 missing delta_rmssd_stress_rest_100hz"
    assert "delta_rmssd_stress_rest_500hz" in cell_16_src, "Cell 16 missing delta_rmssd_stress_rest_500hz"
    assert "OVERWRITE_EXISTING_OUTPUTS" in cell_16_src, "Cell 16 missing OVERWRITE_EXISTING_OUTPUTS check"
    assert "tmp_patient_csv = patient_summary_csv.with_suffix(\".csv.tmp\")" in cell_16_src, "Cell 16 missing patient summary temp file"
    assert "tmp_patient_csv.replace(patient_summary_csv)" in cell_16_src, "Cell 16 missing atomic replace for patient summary"
    assert "tmp_patient_csv.unlink(missing_ok=True)" in cell_16_src, "Cell 16 missing patient summary tmp cleanup"
    print("  [PASS] Correction 3 verified: RMSSD delta trajectory columns and safe atomic export added to patient summary.")

    # 7. Optional Hardening: Cell 20 (index 20 in cells) Pilot SHA-256 Immutability Check
    cell_20_src = "".join(cells[20]["source"])
    assert "pilot_manifest_file = project_root / \"reports\" / \"rpeaks\" / \"pilot_40_sha256_manifest.json\"" in cell_20_src, "Cell 20 missing pilot manifest path"
    assert "hashlib.sha256(arc_p.read_bytes()).hexdigest()" in cell_20_src, "Cell 20 missing sha256 calculation"
    print("  [PASS] Optional Hardening verified: Cell 20 includes cryptographic immutability spot-check for Stage 05 pilot archives.")

    # 8. Check outputs directory: ensure no premature files were created
    hrv_reports_dir = project_root / "reports" / "hrv"
    created_reports = list(hrv_reports_dir.iterdir()) if hrv_reports_dir.exists() else []
    print(f"Files currently in reports/hrv/: {[p.name for p in created_reports]}")
    assert not any(p.name.startswith("06_") for p in created_reports), "Premature report file generated!"

    figures_dir = project_root / "outputs" / "figures"
    created_figures = list(figures_dir.iterdir()) if figures_dir.exists() else []
    assert not any(p.name.startswith("06_") for p in created_figures), "Premature figure generated!"
    print("  [PASS] Output directories clean: Zero premature reports or figures generated.")

    # 9. Verify Stage 05 Pilot Archives cryptographic immutability right now
    pilot_file = project_root / "reports" / "rpeaks" / "pilot_40_sha256_manifest.json"
    if pilot_file.exists():
        with open(pilot_file, "r", encoding="utf-8") as f:
            pilot_baseline = json.load(f)
        rpeaks_dir = project_root / "data" / "processed" / "rpeaks"
        mismatches = []
        for rec_id, entry in pilot_baseline.items():
            exp_hash = entry["sha256"] if isinstance(entry, dict) else entry
            p = rpeaks_dir / f"{rec_id}_rpeaks.npz"
            if not p.exists():
                mismatches.append(f"MISSING: {rec_id}")
            else:
                h = hashlib.sha256(p.read_bytes()).hexdigest()
                if h != exp_hash:
                    mismatches.append(f"MUTATED: {rec_id}")
        assert len(mismatches) == 0, f"Stage 05 pilot archives modified! {mismatches}"
        print(f"  [PASS] Pre-execution audit: All {len(pilot_baseline)} Stage 05 pilot archives verified 100% byte-for-byte identical.")

    print("\n>>> ALL STATIC VALIDATION CHECKS PASSED PERFECTLY! <<<")

if __name__ == "__main__":
    validate_notebook_static()
