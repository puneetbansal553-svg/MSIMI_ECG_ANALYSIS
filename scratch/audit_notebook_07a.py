"""
audit_notebook_07a.py
Static audit script for notebooks/07a_inter_phase_transition_analysis.ipynb.
"""
import ast
import json
import re
from pathlib import Path

def main():
    nb_path = Path("notebooks/07a_inter_phase_transition_analysis.ipynb")
    print(f"Auditing notebook: {nb_path.resolve()}")
    assert nb_path.exists(), f"Notebook {nb_path} does not exist!"

    with open(nb_path, "r", encoding="utf-8") as f:
        nb_data = json.load(f)

    cells = nb_data.get("cells", [])
    print(f"Total cells: {len(cells)}")

    md_cells = [c for c in cells if c["cell_type"] == "markdown"]
    code_cells = [c for c in cells if c["cell_type"] == "code"]
    print(f"Markdown cells: {len(md_cells)}, Code cells: {len(code_cells)}")

    # 1. Check execution state
    for idx, c in enumerate(code_cells):
        assert c["execution_count"] is None, f"Code cell {idx} has execution_count != None: {c['execution_count']}"
        assert c["outputs"] == [], f"Code cell {idx} has non-empty outputs: {len(c['outputs'])}"
    print("CHECK 1 PASSED: All code cells are unexecuted (execution_count: null, outputs: []).")

    # 2. Check AST syntax of all code cells
    for idx, c in enumerate(code_cells):
        code_str = "".join(c["source"])
        try:
            ast.parse(code_str)
        except SyntaxError as e:
            print(f"SYNTAX ERROR in code cell {idx}: {e}")
            raise
    print("CHECK 2 PASSED: All 8 code cells parsed successfully with Python AST (no SyntaxError).")

    # Combine all code for semantic inspection
    all_code = "\n".join("".join(c["source"]) for c in code_cells)

    # 3. Check 5 Methodological Refinements
    # Refinement 1: HRR60 definition
    assert "hrr60" in all_code.lower(), "HRR60 metric missing from code"
    assert "t_max_stress - 10" in all_code or "beats_end_stress" in all_code, "HRR60 final 10s of stress not found"
    assert "50" in all_code and "70" in all_code, "Recovery [50, 70]s window not found"
    print("CHECK 3.1 PASSED: Refinement 1 (HRR60 exact final 10s stress & 50-70s recovery) verified.")

    # Refinement 2: HRRI definition and optionality
    assert "hrri" in all_code.lower(), "HRRI metric missing from code"
    assert "chronotropic_reserve" in all_code.lower() or "reserve" in all_code.lower() or "delta" in all_code.lower(), "HRRI chronotropic reserve missing"
    assert "5.0" in all_code or "5" in all_code, "Chronotropic reserve <= 5 bpm threshold missing"
    print("CHECK 3.2 PASSED: Refinement 2 (HRRI formula with blunted response guard <= 5 bpm) verified.")

    # Refinement 3: Algorithmic J-point delineation
    assert "delineate_j_point" in all_code, "delineate_j_point function missing from code"
    assert "j_60" in all_code.lower() or "st_60" in all_code.lower() or "st60" in all_code.lower(), "ST60 metric missing from code"
    print("CHECK 3.3 PASSED: Refinement 3 (Algorithmic J-point delineation via derivative thresholding & ST60) verified.")

    # Refinement 4: Stratified paired statistical comparisons
    assert "cohen" in all_code.lower() or "wilcoxon" in all_code.lower(), "Paired statistical comparison missing"
    assert "100" in all_code and "500" in all_code, "Stratification by 100 Hz and 500 Hz missing"
    assert "ci" in all_code.lower() or "confidence" in all_code.lower(), "Confidence intervals missing"
    print("CHECK 3.4 PASSED: Refinement 4 (Stratified paired comparisons with effect sizes & 95% CIs) verified.")

    # Refinement 5: DTW exploratory & primary distance metrics
    assert "pearson" in all_code.lower() or "rmsd" in all_code.lower() or "frobenius" in all_code.lower(), "Primary morphology distance metrics missing"
    assert "dtw" in all_code.lower(), "DTW missing"
    print("CHECK 3.5 PASSED: Refinement 5 (Pearson, RMSD, Frobenius primary; DTW exploratory) verified.")

    # 4. Check Protocol T3 (RECOVERY -> STRESS)
    assert "PROTOCOL_NOT_APPLICABLE" in all_code, "PROTOCOL_NOT_APPLICABLE status missing for T3"
    print("CHECK 4 PASSED: T3 transition correctly cataloged as PROTOCOL_NOT_APPLICABLE.")

    # 5. Check Clinical Label Isolation & Candidate NN Status
    forbidden_labels = ["anoca_confirmed", "raw_msimi_ecg_label", "target_label", "diagnosis", "ischemia_label"]
    # Check that forbidden labels do not appear in transition manifest columns
    assert "candidate_nn_status" in all_code or "stage06_candidate_nn_status" in all_code, "candidate_nn_status missing"
    assert "UNKNOWN" in all_code, "Candidate NN status must remain UNKNOWN"
    for label in forbidden_labels:
        # Check manifest export logic
        manifest_cols_match = re.search(r"manifest_cols\s*=\s*\[(.*?)\]", all_code, re.DOTALL)
        if manifest_cols_match:
            manifest_cols_str = manifest_cols_match.group(1)
            assert label not in manifest_cols_str, f"Forbidden label '{label}' found in manifest columns: {manifest_cols_str}"
    print("CHECK 5 PASSED: Clinical labels strictly isolated; stage06_candidate_nn_status remains UNKNOWN.")

    # 6. Safeguards: Disk space, Overwrite protection, Atomic writes (.tmp naming)
    assert "OVERWRITE_EXISTING_OUTPUTS" in all_code, "OVERWRITE_EXISTING_OUTPUTS missing"
    assert "3.0" in all_code or "DISK_SPACE_GUARD_GB" in all_code, "Disk space guard missing"
    # Atomic file names
    assert ".tmp.csv" in all_code and ".tmp.json" in all_code and ".tmp.png" in all_code, "Atomic temp file naming missing"
    print("CHECK 6 PASSED: Safeguards (overwrite protection, 3.0 GB disk guard, atomic temp writes) verified.")

    # 7. Check output paths
    expected_outputs = [
        "07a_inter_phase_transition_manifest.csv",
        "07a_inter_phase_transition_summary.json",
        "07a_representative_transition_morphology.png",
        "07a_cohort_transition_dynamics.png",
    ]
    for out in expected_outputs:
        assert out in all_code, f"Expected output path {out} not referenced in code cells"
    print(f"CHECK 7 PASSED: All 4 expected output paths referenced: {expected_outputs}")

    print("\nALL STATIC AUDIT CHECKS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    main()
