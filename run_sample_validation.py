"""Stage 1 Small-Sample Validation Script.

Strict constraints:
- Evaluates a representative sample of approved subjects: sub_001, sub_002, sub_005, sub_029, sub_054.
- Evaluates 100 Hz and 500 Hz across Rest, Stress, and Recovery.
- Does NOT automatically truncate or pad sub_029; flags its unequal lead lengths for review.
- Records preprocessing and R-peak parameters; does not claim clinical validation.
- Saves all reports inside reports/qc/ without modifying the raw dataset.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd

from configs.dataset_config import (
    RAW_DATA_DIR,
    PHASES,
    SAMPLING_RATES,
    LEAD_NAMES,
    STANDARD_DURATIONS_SEC,
    FILTER_PARAMS,
    RPEAK_PARAMS,
    ADC_MIDSCALE_OFFSET,
)
from src.data.loader import load_raw_ecg_leads, check_ecg_integrity
from src.preprocessing.filter_pipeline import preprocess_ecg
from src.peak_detection.pan_tompkins import detect_r_peaks_lead
from src.qc.quality_control import evaluate_lead_quality, evaluate_multilead_qc


SAMPLE_SUBJECTS = ["sub_001", "sub_002", "sub_005", "sub_029", "sub_054"]

WORKSPACE_ROOT = Path(".").resolve()
QC_REPORTS_DIR = WORKSPACE_ROOT / "reports" / "qc"
QC_REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def run_sample_validation():
    print("=" * 70)
    print("STARTING STAGE 1 SMALL-SAMPLE INTEGRITY & QC VALIDATION")
    print(f"Sample cohort: {SAMPLE_SUBJECTS}")
    print(f"Sampling rates: {SAMPLING_RATES}")
    print(f"Phases: {PHASES}")
    print("=" * 70)
    
    integrity_records = []
    qc_details = {}
    rpeak_summary = []
    
    for sid in SAMPLE_SUBJECTS:
        print(f"\n--- Processing Subject: {sid} ---")
        qc_details[sid] = {}
        
        for phase in PHASES:
            for rate in SAMPLING_RATES:
                rec_id = f"{sid}_{phase}_{rate}hz"
                
                # 1. Ingestion without truncation or padding
                leads = load_raw_ecg_leads(sid, phase, rate, raw_dir=RAW_DATA_DIR)
                
                # 2. Integrity check
                integ = check_ecg_integrity(leads, sid, phase, rate)
                
                # Check clipping on all leads (ADC <= 5 or >= 4090)
                total_samples = sum(len(l) for l in leads)
                total_clipped = sum(int(np.sum((l <= 5.0) | (l >= 4090.0))) for l in leads)
                clipping_pct = round(100.0 * total_clipped / max(1, total_samples), 4)
                
                # Flag unequal lead lengths
                has_unequal = integ["has_unequal_lead_lengths"]
                unequal_detail = integ["unequal_lead_lengths_detail"]
                
                integ_record = {
                    "recording_id": rec_id,
                    "subject_id": sid,
                    "phase": phase,
                    "sampling_rate": rate,
                    "lead_count": integ["lead_count"],
                    "lead_count_valid": integ["lead_count_valid"],
                    "duration_sec": integ["duration_sec"],
                    "expected_duration_sec": integ["expected_duration_sec"],
                    "is_duration_irregular": integ["is_duration_irregular"],
                    "has_unequal_lead_lengths": has_unequal,
                    "unequal_lead_lengths_detail": str(unequal_detail) if has_unequal else "None",
                    "nan_count": integ["nan_count"],
                    "inf_count": integ["inf_count"],
                    "all_finite": integ["all_finite"],
                    "flatline_leads": ",".join(integ["flatline_leads"]) if integ["flatline_leads"] else "None",
                    "clipping_pct": clipping_pct,
                    "min_adc_val": integ["min_raw_val"],
                    "max_adc_val": integ["max_raw_val"],
                    "mean_adc_val": round(integ["mean_raw_val"], 2),
                }
                integrity_records.append(integ_record)
                
                # 3. Preprocessing & R-Peak validation on Lead II (Index 1)
                # Note: Lead II is evaluated directly without altering other leads
                lead_ii_raw = leads[1]  # Lead II is index 1
                
                # Single-lead 2D wrapper for preprocessing
                lead_ii_2d = lead_ii_raw.reshape(1, -1)
                filtered_lead_ii_2d, filter_meta = preprocess_ecg(lead_ii_2d, sampling_rate=rate)
                filtered_lead_ii = filtered_lead_ii_2d[0]
                
                # Quality metrics on Lead II
                lead_qc = evaluate_lead_quality(lead_ii_raw, filtered_lead_ii, rate, lead_name="II")
                
                # Pan-Tompkins R-peak detection
                r_res = detect_r_peaks_lead(filtered_lead_ii, sampling_rate=rate, lead_name="II")
                
                rpeak_rec = {
                    "recording_id": rec_id,
                    "subject_id": sid,
                    "phase": phase,
                    "sampling_rate": rate,
                    "lead_evaluated": "II",
                    "lead_samples": len(lead_ii_raw),
                    "lead_quality": lead_qc["quality_rating"],
                    "estimated_snr_db": lead_qc["estimated_snr_db"],
                    "r_peaks_detected": r_res["r_peak_count"],
                    "mean_hr_bpm": r_res["mean_hr_bpm"],
                    "median_rr_ms": r_res["median_rr_ms"],
                    "std_rr_ms": r_res["std_rr_ms"],
                    "implausible_rr_pct": r_res["anomaly_stats"]["physiologically_implausible_pct"],
                }
                rpeak_summary.append(rpeak_rec)
                
                qc_details[sid][f"{phase}_{rate}hz"] = {
                    "integrity": integ_record,
                    "lead_ii_qc": lead_qc,
                    "lead_ii_rpeaks": {
                        "count": r_res["r_peak_count"],
                        "mean_hr_bpm": r_res["mean_hr_bpm"],
                        "median_rr_ms": r_res["median_rr_ms"],
                        "implausible_rr_pct": r_res["anomaly_stats"]["physiologically_implausible_pct"],
                    }
                }
                
                status_flag = "ALERT" if (integ["is_duration_irregular"] or has_unequal) else "OK"
                print(f"[{status_flag}] {rec_id:<28} | Dur: {integ['duration_sec']:>6.1f}s | "
                      f"HR: {r_res['mean_hr_bpm']:>5.1f} bpm | Beating: {r_res['r_peak_count']:>4} | "
                      f"UnequalLeads: {str(has_unequal):<5} | QC: {lead_qc['quality_rating']}")

    # Save outputs
    df_integ = pd.DataFrame(integrity_records)
    df_rpeak = pd.DataFrame(rpeak_summary)
    
    csv_integ_path = QC_REPORTS_DIR / "sample_validation_integrity.csv"
    csv_rpeak_path = QC_REPORTS_DIR / "sample_validation_rpeaks.csv"
    json_path = QC_REPORTS_DIR / "sample_validation_qc_report.json"
    
    df_integ.to_csv(csv_integ_path, index=False)
    df_rpeak.to_csv(csv_rpeak_path, index=False)
    
    report_dict = {
        "sample_cohort": SAMPLE_SUBJECTS,
        "total_recordings_validated": len(integrity_records),
        "parameters": {
            "adc_offset": ADC_MIDSCALE_OFFSET,
            "filter_params": FILTER_PARAMS,
            "rpeak_params": RPEAK_PARAMS,
            "clinical_status": "Proposed research parameters; not clinically validated.",
        },
        "findings": {
            "lead_count_all_12": bool(df_integ["lead_count_valid"].all()),
            "finite_values_all": bool(df_integ["all_finite"].all()),
            "flatlines_detected": [r["recording_id"] for r in integrity_records if r["flatline_leads"] != "None"],
            "irregular_duration_recordings": df_integ[df_integ["is_duration_irregular"]]["recording_id"].tolist(),
            "unequal_lead_length_recordings": df_integ[df_integ["has_unequal_lead_lengths"]]["recording_id"].tolist(),
            "mean_clipping_pct": round(float(df_integ["clipping_pct"].mean()), 4),
        },
        "detailed_results": qc_details,
    }
    
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2)
        
    print("\n" + "=" * 70)
    print("SAMPLE VALIDATION COMPLETED")
    print(f"Integrity report saved to: {csv_integ_path}")
    print(f"R-peak summary saved to: {csv_rpeak_path}")
    print(f"Full JSON report saved to: {json_path}")
    print("=" * 70)
    
    return report_dict


if __name__ == "__main__":
    run_sample_validation()

