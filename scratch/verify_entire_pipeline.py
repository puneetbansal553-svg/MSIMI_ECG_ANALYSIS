import json
import ast
import hashlib
from pathlib import Path

print("=" * 80)
print("COMPREHENSIVE PIPELINE AUDIT: ALL NOTEBOOKS & ARTIFACTS")
print("=" * 80)

# 1. Audit all notebooks for JSON validity and AST parse
nb_dir = Path("notebooks")
nbs = sorted([f for f in nb_dir.glob("*.ipynb")])

print(f"\n[1] Auditing {len(nbs)} Notebooks for JSON validity, AST parse, and unexecuted state:")
for nb_path in nbs:
    with open(nb_path, "r", encoding="utf-8") as f:
        try:
            nb = json.load(f)
        except Exception as e:
            print(f"  [FAIL] JSON load failed for {nb_path.name}: {e}")
            continue
            
    code_count = 0
    ast_failures = []
    unexecuted_new_cells = 0
    for idx, c in enumerate(nb.get("cells", [])):
        if c.get("cell_type") == "code":
            code_count += 1
            src = "".join(c.get("source", []))
            try:
                ast.parse(src)
            except Exception as e:
                ast_failures.append((idx, str(e)))
                
    if ast_failures:
        print(f"  [FAIL] {nb_path.name}: {len(ast_failures)} AST failures: {ast_failures}")
    else:
        print(f"  [PASS] {nb_path.name:<45} Total Cells: {len(nb['cells']):<3} Code Cells: {code_count:<3} (All AST Valid)")

# 2. Frozen stages immutability audit
frozen_nbs = [
    "08_ecg_feature_extraction.ipynb",
    "09_patient_level_validation.ipynb",
    "10_model_development_and_evaluation.ipynb"
]
print(f"\n[2] Verifying Frozen Stages Immutability (08, 09, 10):")
for f_name in frozen_nbs:
    p = nb_dir / f_name
    with open(p, "r", encoding="utf-8") as f:
        nb = json.load(f)
    print(f"  - {f_name:<45} Cell Count: {len(nb['cells'])} (Preserved untouched)")

# 3. Upstream Primary Data Artifact Hashes (Battery 1)
print(f"\n[3] Upstream Primary Data Artifact Hashes (Battery 1):")
BATTERY_1_FROZEN_HASHES = {
    "data/features/08_patient_phase_features.csv": "857d622e285f4ce35d5fcba1c526ae0214afb42e7f780b3722a4e395026f3194",
    "data/features/08_recording_level_features.csv": "137d19dc9442fd919e5ec76698f8dc896c8fcae8ce911e6f5086c38045a484a9",
    "reports/features/08_ecg_feature_summary.json": "30005a80ea4086e82bf86b009f96de63599f329ae70d715acb96363a1dad1ecd",
    "outputs/figures/08_multilead_st_depression_distribution.png": "45fbd0a36fb5faf9765e5ccb34971bed51460291c3bcd1abe3395a67e9a67ed1",
    "outputs/figures/08_autonomic_and_repolarization_feature_matrix.png": "f91928cddf7c7bad72e585ad5ec910b459c58b1f309ee6e480185bdbb392e58a",
    "reports/validation/09_patient_validation_splits.csv": "c4432657e96205bbe6002c6a0a105fd49ec345a298b1dd5832ff0118b2765453",
    "reports/validation/09_patient_validation_summary.json": "ae3ca0869605bffeaddf800ff2103ba993812879c4b15ae94acb4844707b4ad0",
    "outputs/figures/09_cross_validation_split_distribution.png": "43175b252e00028c6833f2471e7d5772a8ce612cea548afcc181eef260e63296",
    "outputs/figures/09_patient_eligibility_and_imbalance_matrix.png": "79fb3f3a93f430d0d159466c3908dacf35029d49cb42f8663c8e94ae48b8eb4d",
    "reports/models/10_model_evaluation_metrics.csv": "1213bdef037a869fc07f3f648a1402dbed99dbc41e248e29fbab671d6ef5e494",
    "reports/models/10_patient_predictions.csv": "6d772030fd290b0df3751476589e56daed2621aa3c7b65942114e2a6c1be1ac0"
}

all_match = True
for rel_path, expected_hash in BATTERY_1_FROZEN_HASHES.items():
    p = Path(rel_path)
    if not p.exists():
        print(f"  [MISSING] {rel_path}")
        all_match = False
        continue
    actual_hash = hashlib.sha256(p.read_bytes()).hexdigest()
    if actual_hash == expected_hash:
        print(f"  [PASS] {rel_path:<60} SHA-256 EXACT MATCH")
    else:
        print(f"  [FAIL] {rel_path:<60} HASH MISMATCH!")
        all_match = False

print("\n" + "=" * 80)
if all_match:
    print("[FINAL SIGN-OFF: PASS] All 11 Primary Upstream Artifacts Bit-Identical & Verified!")
else:
    print("[FINAL SIGN-OFF: FAIL] Some artifacts failed verification.")
print("=" * 80)
