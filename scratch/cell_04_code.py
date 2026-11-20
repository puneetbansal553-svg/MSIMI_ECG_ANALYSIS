# Upstream artifact paths
patient_features_csv = features_dir / "08_patient_phase_features.csv"
splits_csv_path = reports_validation_dir / "09_patient_validation_splits.csv"
summary_json_path = reports_validation_dir / "09_patient_validation_summary.json"

for p in [patient_features_csv, splits_csv_path, summary_json_path]:
    assert p.exists(), f"HALT: Critical upstream artifact missing: {p}"

# Ingest feature store and split registry
df_patient_raw = pd.read_csv(patient_features_csv)
df_splits = pd.read_csv(splits_csv_path)
with open(summary_json_path, "r", encoding="utf-8") as f:
    split_summary = json.load(f)

# 1. Unconditional Drop of 3 Dead Columns (160/160 Nulls)
dead_cols = ["rest_raw_detector_rr_mean_ms", "stress_raw_detector_rr_mean_ms", "recover_raw_detector_rr_mean_ms"]
df_patient_features = df_patient_raw.drop(columns=[c for c in dead_cols if c in df_patient_raw.columns])
print(f"Ingested Feature Store: {df_patient_features.shape[0]} rows x {df_patient_features.shape[1]} columns (after dropping dead columns).")

# 2. Invariant Verification: Cohort Accounting (80 Total Subjects)
assert len(df_splits) == 80, f"Expected 80 subjects in split registry, found {len(df_splits)}"
assert df_splits["subject_id"].is_unique, "Subject IDs must be strictly unique."

# 3. Segregate Holdouts (N=2)
df_holdouts = df_splits[~df_splits["validation_eligible"]].copy()
holdout_ids = sorted(df_holdouts["subject_id"].tolist())
assert holdout_ids == ["sub_005", "sub_049"], f"Unexpected holdout IDs: {holdout_ids}"
assert (df_holdouts["target_label"].isna()).all(), "Holdout targets must be NaN."
for r_i in range(5):
    assert (df_holdouts[f"repeat_{r_i}_fold"] == -1).all(), f"Holdouts must have fold_id = -1 in repeat {r_i}"

# 4. Filter CV-Eligible Cohort (N=78)
df_eligible = df_splits[df_splits["validation_eligible"]].copy()
assert len(df_eligible) == 78, f"Expected 78 eligible subjects, found {len(df_eligible)}"
n_pos_total = int((df_eligible["target_label"] == 1).sum())
n_neg_total = int((df_eligible["target_label"] == 0).sum())
assert n_pos_total == 9, f"Expected 9 total positives, found {n_pos_total}"
assert n_neg_total == 69, f"Expected 69 total negatives, found {n_neg_total}"

# 5. Identify Matched Dual-Rate Sub-cohort (N=62) vs Skipped 100 Hz Stress Sub-cohort (N=16)
df_100_stratum = df_patient_features[df_patient_features["sampling_rate"] == 100]
m_100_split = pd.merge(df_100_stratum, df_eligible, on="subject_id")

matched_mask = m_100_split["stress_recording_status"] == "SUCCESS"
df_matched_eligible = m_100_split[matched_mask].copy()
df_skipped_eligible = m_100_split[~matched_mask].copy()

matched_ids = sorted(df_matched_eligible["subject_id"].tolist())
skipped_ids = sorted(df_skipped_eligible["subject_id"].tolist())

assert len(matched_ids) == 62, f"Expected 62 matched eligible subjects, got {len(matched_ids)}"
assert len(skipped_ids) == 16, f"Expected 16 skipped eligible subjects, got {len(skipped_ids)}"

n_pos_matched = int((df_matched_eligible["target_label"] == 1).sum())
n_neg_matched = int((df_matched_eligible["target_label"] == 0).sum())
n_pos_skipped = int((df_skipped_eligible["target_label"] == 1).sum())
n_neg_skipped = int((df_skipped_eligible["target_label"] == 0).sum())

assert n_pos_matched == 7 and n_neg_matched == 55, f"Unexpected matched counts: {n_pos_matched}+ / {n_neg_matched}-"
assert n_pos_skipped == 2 and n_neg_skipped == 14, f"Unexpected skipped counts: {n_pos_skipped}+ / {n_neg_skipped}-"

print("=== Upstream Cohort Membership & Invariant Verification ===")
print(f"  Total Enrolled Cohort:                  {len(df_splits)} subjects (100.0%)")
print(f"  Unlabelled Complete Holdouts:           {len(df_holdouts)} subjects ({holdout_ids})")
print(f"  Total Labelled CV-Eligible Cohort:      {len(df_eligible)} subjects (9 positive, 69 negative)")
print(f"    - Matched Complete Dual-Rate Sub-cohort: {len(df_matched_eligible)} subjects (7 positive, 55 negative)")
print(f"    - Incomplete 100 Hz Stress Sub-cohort:   {len(df_skipped_eligible)} subjects (2 positive, 14 negative)")
print("  [PASS] Cohort accounting invariant verified: 62 + 16 + 2 = 80.")
