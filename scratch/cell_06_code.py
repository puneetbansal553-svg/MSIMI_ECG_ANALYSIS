# Define Parsimonious Clinical Feature Subsets (Verified exact column names)
FEATURE_SUBSETS: Dict[str, List[str]] = {
    "Subset_A_ST_Dynamics": [
        "stress_min_st60_global_adc",
        "stress_min_st60_inferior_adc",
        "stress_min_st60_anterior_adc",
        "t1_delta_min_st60_global_adc"
    ],
    "Subset_B_Autonomic": [
        "t2_hrr60_bpm",
        "t1_delta_hr_bpm",
        "rest_raw_detector_rr_rmssd_ms"
    ],
    "Subset_C_Multimodal": [
        "stress_min_st60_global_adc",
        "t1_delta_min_st60_global_adc",
        "t2_hrr60_bpm",
        "t1_delta_hr_bpm",
        "t1_frobenius_dissimilarity_adc_norm"
    ]
}

# Verify schema: ensure every column exists in df_patient_features
all_features_flattened = sorted(list(set([f for sub in FEATURE_SUBSETS.values() for f in sub])))
print("=== Feature Subset Schema Audit ===")
for sub_name, feat_list in FEATURE_SUBSETS.items():
    missing_in_store = [f for f in feat_list if f not in df_patient_features.columns]
    assert len(missing_in_store) == 0, f"HALT: Missing features in store for {sub_name}: {missing_in_store}"
    print(f"  {sub_name} (k={len(feat_list)}): {feat_list}")

print("  [PASS] All referenced feature columns exist in Stage 08 patient feature store.")
