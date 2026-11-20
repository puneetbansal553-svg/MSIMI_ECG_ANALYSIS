import json
import pandas as pd
from pathlib import Path

manifest_p = Path("reports/hrv/06_rr_hrv_manifest.csv")
patient_p = Path("reports/hrv/06_patient_hrv_summary.csv")
summary_p = Path("reports/hrv/06_rr_hrv_summary.json")

print("=== SUMMARY JSON ===")
if summary_p.exists():
    with open(summary_p, "r", encoding="utf-8") as f:
        s = json.load(f)
    print("Timestamp UTC:", s.get("generation_timestamp_utc"))
    print("Total recordings:", s.get("total_cohort_recordings"))
    print("Successful evaluations:", s.get("successful_hrv_evaluations"))
    print("Skipped QC audit errors:", s.get("skipped_qc_audit_errors"))
    print("Provenance tag:", s.get("provenance_tag"))
    print("Composite quality dist:", s.get("composite_quality_distribution"))
    print("Spectral eligibility dist:", s.get("spectral_eligibility_distribution"))

print("\n=== PATIENT SUMMARY CSV ===")
if patient_p.exists():
    df_p = pd.read_csv(patient_p)
    print("Shape:", df_p.shape)
    print("Columns:", list(df_p.columns))
    print("Unique subjects:", df_p["subject_id"].nunique())
    print("ANOCA confirmed count:", df_p["anoca_confirmed"].sum())
    print("Label status counts:\n", df_p["label_status"].value_counts())
    print("Target label counts:\n", df_p["target_label"].value_counts(dropna=False))
    print("Status stress 100hz counts:\n", df_p["status_stress_100hz"].value_counts())
    print("delta_rmssd_stress_rest_100hz in df_p:", "delta_rmssd_stress_rest_100hz" in df_p.columns)
    print("delta_rmssd_stress_rest_500hz in df_p:", "delta_rmssd_stress_rest_500hz" in df_p.columns)
    if "delta_rmssd_stress_rest_100hz" in df_p.columns:
        print("RMSSD delta 100hz null count:", df_p["delta_rmssd_stress_rest_100hz"].isna().sum())
        print("RMSSD delta 500hz null count:", df_p["delta_rmssd_stress_rest_500hz"].isna().sum())
        print("Sample delta RMSSD:\n", df_p[["subject_id", "delta_rmssd_stress_rest_100hz", "delta_rmssd_stress_rest_500hz"]].head())

print("\n=== RECORDING MANIFEST CSV ===")
if manifest_p.exists():
    df_m = pd.read_csv(manifest_p)
    print("Shape:", df_m.shape)
    print("Columns:", list(df_m.columns))
    print("Unique recording IDs:", df_m["recording_id"].nunique())
    print("Detection status counts:\n", df_m["rpeak_detection_status"].value_counts())
    print("Provenance tag unique:", df_m["provenance_tag"].unique())
    print("Rhythm lead counts:\n", df_m["selected_rhythm_lead"].value_counts(dropna=False))

