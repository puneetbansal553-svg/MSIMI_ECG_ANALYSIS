"""Cohort Manifest and Label Extraction Module.

Strict requirements:
- Cohort: sub_001 through sub_080 (the 80 verified ANOCA-positive subjects).
- Target field: MSIMI-based-ecg.
- Preserve unspecified labels as missing/None; never infer or impute them.
- Exclude medical images and clinical predictors from model inputs.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd

from configs.dataset_config import (
    RAW_DATA_DIR,
    DATA_DIR,
    COHORT_SUBJECT_IDS,
    TARGET_LABEL_FIELD,
    ANOCA_FIELD,
    PHASES,
    PHASE_PREFIX_MAP,
    SAMPLING_RATES,
)


def load_subject_disease_json(subject_id: str, raw_dir: Path = RAW_DATA_DIR) -> Dict[str, Any]:
    """Read a single subject's disease.json in read-only mode."""
    filepath = raw_dir / f"{subject_id}_disease.json"
    if not filepath.exists():
        raise FileNotFoundError(f"Missing disease.json for {subject_id}: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def check_ecg_files_exist(subject_id: str, raw_dir: Path = RAW_DATA_DIR) -> Dict[str, bool]:
    """Verify that all 6 required ECG files exist for a given subject."""
    status = {}
    for phase in PHASES:
        prefix = PHASE_PREFIX_MAP[phase]
        for rate in SAMPLING_RATES:
            fname = f"{subject_id}_{prefix}-{rate}hz.csv"
            fpath = raw_dir / fname
            status[f"{phase}_{rate}hz"] = fpath.exists()
    return status


def build_cohort_manifest(
    raw_dir: Path = RAW_DATA_DIR,
    cohort_ids: List[str] = COHORT_SUBJECT_IDS,
    output_dir: Path = DATA_DIR,
) -> pd.DataFrame:
    """Build and save the patient-level manifest for the 80 approved subjects.
    
    Unspecified labels ('no description') are strictly preserved as None / NaN,
    with an explicit label_status='UNSPECIFIED'.
    """
    records = []
    
    for sid in cohort_ids:
        disease_data = load_subject_disease_json(sid, raw_dir=raw_dir)
        
        # Verify ANOCA field
        anoca_val = disease_data.get(ANOCA_FIELD)
        if anoca_val != 1:
            raise ValueError(f"Subject {sid} has unexpected ANOCA status: {anoca_val} (expected 1)")
        
        # Extract target label
        raw_ecg_label = disease_data.get(TARGET_LABEL_FIELD)
        
        if raw_ecg_label == 1:
            label_status = "POSITIVE"
            label_int = 1
        elif raw_ecg_label == 0:
            label_status = "NEGATIVE"
            label_int = 0
        elif raw_ecg_label == "no description" or raw_ecg_label is None:
            label_status = "UNSPECIFIED"
            label_int = None
        else:
            label_status = "AMBIGUOUS"
            label_int = None
        
        # Check ECG files
        ecg_status = check_ecg_files_exist(sid, raw_dir=raw_dir)
        all_ecg_present = all(ecg_status.values())
        
        record = {
            "subject_id": sid,
            "anoca_confirmed": True,
            "raw_msimi_ecg_label": str(raw_ecg_label),
            "label_status": label_status,
            "target_label": label_int,  # None for unspecified
            "all_ecg_files_present": all_ecg_present,
            "has_rest_100hz": ecg_status.get("rest_100hz", False),
            "has_rest_500hz": ecg_status.get("rest_500hz", False),
            "has_stress_100hz": ecg_status.get("stress_100hz", False),
            "has_stress_500hz": ecg_status.get("stress_500hz", False),
            "has_recover_100hz": ecg_status.get("recover_100hz", False),
            "has_recover_500hz": ecg_status.get("recover_500hz", False),
        }
        records.append(record)
    
    df = pd.DataFrame(records)
    
    # Save outputs inside workspace data directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    csv_path = output_dir / "cohort_manifest.csv"
    json_path = output_dir / "cohort_manifest.json"
    
    df.to_csv(csv_path, index=False)
    
    # Save JSON preserving None as null
    manifest_dict = {
        "cohort_size": len(df),
        "target_field": TARGET_LABEL_FIELD,
        "positive_count": int((df["label_status"] == "POSITIVE").sum()),
        "negative_count": int((df["label_status"] == "NEGATIVE").sum()),
        "unspecified_count": int((df["label_status"] == "UNSPECIFIED").sum()),
        "subjects": records,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(manifest_dict, f, indent=2)
        
    return df


if __name__ == "__main__":
    df = build_cohort_manifest()
    print("Manifest created successfully:")
    print(f"Total subjects: {len(df)}")
    print(f"Label breakdown:\n{df['label_status'].value_counts()}")

