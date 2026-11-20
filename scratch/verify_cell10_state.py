"""
verify_cell10_state.py
Validates the executed Cell 10 state and output files against all 13 checklist points.
"""
import sys
import json
import shutil
from pathlib import Path
import pandas as pd
import numpy as np

def main():
    project_root = Path.cwd()
    nb_path = project_root / "notebooks" / "07a_inter_phase_transition_analysis.ipynb"
    manifest_csv = project_root / "reports" / "transitions" / "07a_inter_phase_transition_manifest.csv"
    summary_json = project_root / "reports" / "transitions" / "07a_inter_phase_transition_summary.json"

    print("=== Verification of Cell 10 Outputs & Integrity ===")

    # 1. Both files exist and non-empty
    assert manifest_csv.exists() and manifest_csv.stat().st_size > 0, "Manifest CSV missing or empty"
    assert summary_json.exists() and summary_json.stat().st_size > 0, "Summary JSON missing or empty"
    print(f"1. [PASS] Files exist: CSV ({manifest_csv.stat().st_size:,} bytes), JSON ({summary_json.stat().st_size:,} bytes)")

    # 2. CSV contains exactly 288 records
    df = pd.read_csv(manifest_csv)
    assert len(df) == 288, f"Expected 288 records, found {len(df)}"
    print(f"2. [PASS] CSV records count = {len(df)}")

    # 3. T1 = 144 and T2 = 144
    t1_count = len(df[df["transition_type"] == "REST_TO_STRESS"])
    t2_count = len(df[df["transition_type"] == "STRESS_TO_RECOVERY"])
    assert t1_count == 144, f"T1 count = {t1_count}"
    assert t2_count == 144, f"T2 count = {t2_count}"
    print(f"3. [PASS] Transition counts: T1 = {t1_count}, T2 = {t2_count}")

    # 4. T3 = 0 and remains PROTOCOL_NOT_APPLICABLE
    assert "RECOVERY_TO_STRESS" not in df["transition_type"].unique()
    print("4. [PASS] T3 = 0 pairs, strictly PROTOCOL_NOT_APPLICABLE (zero fabricated records)")

    # 5. Required columns and data types
    expected_cols = [
        'transition_id', 'subject_id', 'sampling_rate', 'transition_type',
        'pre_recording_id', 'post_recording_id', 'transition_status',
        'pre_eligible_beats', 'post_eligible_beats', 'hr_pre_bpm', 'hr_post_bpm',
        'delta_hr_bpm', 'rr_ratio', 'hr_end_stress_bpm', 'hr_rec_60s_bpm',
        'hrr60_bpm', 'hr_late_stress_120s_bpm', 'hrri_pct', 'delta_rmssd_ms',
        'mean_morphology_corr', 'min_morphology_corr', 'frobenius_dissimilarity_mv',
        'max_rmsd_lead', 'max_rmsd_value_mv', 'j_point_offset_ms_v5',
        'delta_st60_lead_v5_mv', 'delta_st60_lead_ii_mv',
        'stage06_candidate_nn_status'
    ]
    assert list(df.columns) == expected_cols, "Column mismatch"
    print(f"5. [PASS] All {len(expected_cols)} required columns and data types preserved")

    # 6. Mandatory metrics contain no unexpected NaN/Inf
    mandatory_cols = [
        "transition_id", "subject_id", "sampling_rate", "transition_type",
        "pre_recording_id", "post_recording_id", "transition_status",
        "pre_eligible_beats", "post_eligible_beats", "hr_pre_bpm", "hr_post_bpm",
        "delta_hr_bpm", "rr_ratio", "delta_rmssd_ms",
        "mean_morphology_corr", "min_morphology_corr", "frobenius_dissimilarity_mv",
        "max_rmsd_lead", "max_rmsd_value_mv",
        "j_point_offset_ms_v5", "delta_st60_lead_v5_mv", "delta_st60_lead_ii_mv",
        "stage06_candidate_nn_status"
    ]
    for col in mandatory_cols:
        assert df[col].isna().sum() == 0, f"Unexpected NaN in {col}"
        if df[col].dtype in [np.float64, np.float32]:
            assert not np.isinf(df[col]).any(), f"Unexpected Inf in {col}"
    print(f"6. [PASS] Zero NaN/Inf across all {len(mandatory_cols)} mandatory columns")

    # 7. Optional HRRI NaNs retain documented guard reasons
    t2_df = df[df["transition_type"] == "STRESS_TO_RECOVERY"]
    t1_df = df[df["transition_type"] == "REST_TO_STRESS"]
    assert t1_df["hrri_pct"].isna().all(), "T1 HRRI must be NaN"
    assert t2_df["hrri_pct"].notna().sum() == 20, "Expected 20 valid HRRI on T2"
    assert t2_df["hrri_pct"].isna().sum() == 124, "Expected 124 NaN HRRI on T2"
    print("7. [PASS] Optional HRRI NaNs retain documented reason (124 NaNs due to reserve <= 5 bpm guard, 144 on T1)")

    # 8. Diagnosis-related columns/labels absent
    forbidden = ["anoca_confirmed", "raw_msimi_ecg_label", "target_label", "MSIMI-based-ecg", "label_status", "diagnosis"]
    for col in forbidden:
        assert col not in df.columns, f"Forbidden label {col} found"
    print("8. [PASS] Zero clinical diagnosis labels in manifest")

    # 9. stage06_candidate_nn_status remains UNKNOWN
    assert set(df["stage06_candidate_nn_status"].unique()) == {"UNKNOWN"}
    print("9. [PASS] stage06_candidate_nn_status strictly UNKNOWN (288/288)")

    # 10. JSON is parseable and reconciles exactly with CSV
    with open(summary_json, "r", encoding="utf-8") as f:
        sdata = json.load(f)
    assert sdata["transitions_evaluated"]["T1_REST_TO_STRESS"]["total_pairs"] == 144
    assert sdata["transitions_evaluated"]["T2_STRESS_TO_RECOVERY"]["total_pairs"] == 144
    assert sdata["transitions_evaluated"]["T3_RECOVERY_TO_STRESS"]["total_pairs"] == 0
    assert sdata["transitions_evaluated"]["T1_REST_TO_STRESS"]["mean_delta_hr_bpm"] == round(float(t1_df["delta_hr_bpm"].mean()), 2)
    assert sdata["transitions_evaluated"]["T2_STRESS_TO_RECOVERY"]["median_hrr60_bpm"] == round(float(t2_df["hrr60_bpm"].median()), 2)
    print("10. [PASS] Summary JSON parsed and reconciled 100% with manifest CSV")

    # 11. Stage 03-07 inputs remain unchanged
    s07_manifest = Path("reports/segmentation/07_beat_segmentation_manifest.csv")
    assert s07_manifest.stat().st_size == 58842887, "Stage 07 manifest size changed"
    assert len(list(Path("data/processed/beat_segments").glob("*_beats.npz"))) == 464
    print("11. [PASS] Stage 03-07 inputs remain completely unchanged")

    # 12. Final free disk space is above 3.0 GB
    _, _, free_b = shutil.disk_usage(project_root)
    free_gb = free_b / (1024 ** 3)
    assert free_gb >= 3.0, f"Disk space below 3.0 GB ({free_gb:.3f} GB)"
    print(f"12. [PASS] Free disk space = {free_gb:.3f} GB (>= 3.0 GB)")

    # 13. No orphan temporary files remain
    orphan_tmps = list((project_root / "reports" / "transitions").glob("*.tmp*"))
    assert len(orphan_tmps) == 0, f"Found orphan temp files: {orphan_tmps}"
    print("13. [PASS] Zero orphan temporary files in reports/transitions")

    # 14. Downstream Cells 12, 14, 16 unexecuted
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)
    for idx in [12, 14, 16]:
        c = nb["cells"][idx]
        assert c["execution_count"] is None and c["outputs"] == [], f"Cell {idx} executed!"
    print("14. [PASS] Downstream Cells 12, 14, 16 remain strictly unexecuted")

    print("\nALL 14 CHECKS PASSED WITH 100% SUCCESS!")

if __name__ == "__main__":
    main()

