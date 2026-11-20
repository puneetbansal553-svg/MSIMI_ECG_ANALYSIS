# Forensic Audit Verification Battery
stage08_expected_hashes = {
    "patient_features": "857d622e285f4ce35d5fcba1c526ae0214afb42e7f780b3722a4e395026f3194",
    "recording_features": "137d19dc9442fd919e5ec76698f8dc896c8fcae8ce911e6f5086c38045a484a9",
    "feature_summary": "30005a80ea4086e82bf86b009f96de63599f329ae70d715acb96363a1dad1ecd",
    "figure_st": "45fbd0a36fb5faf9765e5ccb34971bed51460291c3bcd1abe3395a67e9a67ed1",
    "figure_matrix": "f91928cddf7c7bad72e585ad5ec910b459c58b1f309ee6e480185bdbb392e58a"
}

stage09_expected_hashes = {
    "splits_csv": "c4432657e96205bbe6002c6a0a105fd49ec345a298b1dd5832ff0118b2765453",
    "summary_json": "ae3ca0869605bffeaddf800ff2103ba993812879c4b15ae94acb4844707b4ad0"
}

def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

print("=== MSIMI Stage 10: Upstream Immutability Audit ===")
for name, p, exp_h in [
    ("Stage 08 Patient CSV", patient_features_csv, stage08_expected_hashes["patient_features"]),
    ("Stage 08 Recording CSV", features_dir / "08_recording_level_features.csv", stage08_expected_hashes["recording_features"]),
    ("Stage 08 Summary JSON", reports_features_dir / "08_ecg_feature_summary.json", stage08_expected_hashes["feature_summary"]),
    ("Stage 08 ST Fig PNG", figures_dir / "08_multilead_st_depression_distribution.png", stage08_expected_hashes["figure_st"]),
    ("Stage 08 Matrix Fig PNG", figures_dir / "08_autonomic_and_repolarization_feature_matrix.png", stage08_expected_hashes["figure_matrix"]),
    ("Stage 09 Splits CSV", splits_csv_path, stage09_expected_hashes["splits_csv"]),
    ("Stage 09 Summary JSON", summary_json_path, stage09_expected_hashes["summary_json"]),
]:
    act_h = compute_sha256(p)
    assert act_h == exp_h, f"HALT: {name} was altered! Got {act_h}"
    print(f"  [IMMUTABLE] {name:25s} SHA-256 verified ({act_h[:16]}...)")

print("\nUpstream cryptographic immutability verified 100% bit-for-bit.")
