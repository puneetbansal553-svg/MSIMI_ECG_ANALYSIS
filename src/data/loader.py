"""Read-only ECG Signal Loader and Integrity Check Module.

Strict requirements:
- Read raw CSV files from RAW_DATA_DIR without modifying them.
- Preserve 12-lead structure: rows correspond to leads I, II, III, aVR, aVL, aVF, V1..V6.
- Check sampling rate, duration, lead count, NaN / Inf non-finite values, flatlines.
- Flag irregular-duration recordings.
"""

from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import numpy as np

from configs.dataset_config import (
    RAW_DATA_DIR,
    COHORT_SUBJECT_IDS,
    PHASES,
    PHASE_PREFIX_MAP,
    SAMPLING_RATES,
    LEAD_NAMES,
    EXPECTED_LEAD_COUNT,
    STANDARD_DURATIONS_SEC,
    IRREGULAR_RECORDINGS,
)


def get_ecg_filepath(
    subject_id: str,
    phase: str,
    sampling_rate: int,
    raw_dir: Path = RAW_DATA_DIR,
) -> Path:
    """Resolve the raw ECG CSV file path."""
    if phase not in PHASES:
        raise ValueError(f"Unknown phase '{phase}'. Expected one of {PHASES}")
    if sampling_rate not in SAMPLING_RATES:
        raise ValueError(f"Unsupported sampling rate '{sampling_rate}'. Expected {SAMPLING_RATES}")
    
    prefix = PHASE_PREFIX_MAP[phase]
    filename = f"{subject_id}_{prefix}-{sampling_rate}hz.csv"
    return raw_dir / filename


class UnequalLeadLengthsError(ValueError):
    """Raised when an ECG recording has unequal lead lengths across the 12 leads."""
    def __init__(self, subject_id: str, phase: str, sampling_rate: int, lead_lengths: list):
        self.subject_id = subject_id
        self.phase = phase
        self.sampling_rate = sampling_rate
        self.lead_lengths = lead_lengths
        unique_lens = sorted(list(set(lead_lengths)))
        msg = (
            f"Subject '{subject_id}' ({phase}, {sampling_rate} Hz) has unequal lead lengths: "
            f"{unique_lens}. Automatic truncation or padding is disabled pending review."
        )
        super().__init__(msg)


def load_raw_ecg_leads(
    subject_id: str,
    phase: str,
    sampling_rate: int,
    raw_dir: Path = RAW_DATA_DIR,
) -> list:
    """Load raw 12-lead ECG recording as a list of 1D arrays without any truncation or padding.
    
    Returns:
        list of 12 np.ndarray elements, each corresponding to LEAD_NAMES[i].
    """
    filepath = get_ecg_filepath(subject_id, phase, sampling_rate, raw_dir=raw_dir)
    if not filepath.exists():
        raise FileNotFoundError(f"ECG file not found: {filepath}")
    
    leads = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip().rstrip(",")
            if not line_str:
                continue
            lead_samples = np.fromstring(line_str, dtype=np.float64, sep=",")
            leads.append(lead_samples)
    
    if len(leads) != EXPECTED_LEAD_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_LEAD_COUNT} leads in {filepath.name}, but found {len(leads)}"
        )
    return leads


def load_raw_ecg(
    subject_id: str,
    phase: str,
    sampling_rate: int,
    raw_dir: Path = RAW_DATA_DIR,
) -> np.ndarray:
    """Load raw 12-lead ECG recording as a 2D matrix (12, N).
    
    If leads have unequal lengths (such as sub_029 rest), raises UnequalLeadLengthsError
    rather than silently truncating or padding.
    """
    leads = load_raw_ecg_leads(subject_id, phase, sampling_rate, raw_dir=raw_dir)
    lead_lengths = [len(l) for l in leads]
    if len(set(lead_lengths)) > 1:
        raise UnequalLeadLengthsError(subject_id, phase, sampling_rate, lead_lengths)
    
    return np.vstack(leads)


def check_ecg_integrity(
    signal_or_leads,
    subject_id: str,
    phase: str,
    sampling_rate: int,
) -> Dict[str, Any]:
    """Perform integrity and quality validation on an ECG recording.
    
    Accepts either a 2D numpy array (12, N) or a list of 1D lead arrays.
    Detects and flags unequal lead lengths without modifying the signal.
    """
    if isinstance(signal_or_leads, list):
        lead_count = len(signal_or_leads)
        lead_lengths = [len(l) for l in signal_or_leads]
        has_unequal = len(set(lead_lengths)) > 1
        min_samples = min(lead_lengths) if lead_lengths else 0
        max_samples = max(lead_lengths) if lead_lengths else 0
        sample_count = min_samples
        nan_count = int(sum(np.isnan(l).sum() for l in signal_or_leads))
        inf_count = int(sum(np.isinf(l).sum() for l in signal_or_leads))
        flatline_leads = [
            LEAD_NAMES[i] for i, l in enumerate(signal_or_leads) if np.std(l) < 1e-6
        ]
        min_val = float(min(np.min(l) for l in signal_or_leads))
        max_val = float(max(np.max(l) for l in signal_or_leads))
        mean_val = float(np.mean([np.mean(l) for l in signal_or_leads]))
    else:
        lead_count, sample_count = signal_or_leads.shape
        lead_lengths = [sample_count] * lead_count
        has_unequal = False
        min_samples = sample_count
        max_samples = sample_count
        nan_count = int(np.isnan(signal_or_leads).sum())
        inf_count = int(np.isinf(signal_or_leads).sum())
        lead_stds = np.std(signal_or_leads, axis=1)
        flatline_leads = [
            LEAD_NAMES[i] for i, s in enumerate(lead_stds) if s < 1e-6
        ]
        min_val = float(np.min(signal_or_leads))
        max_val = float(np.max(signal_or_leads))
        mean_val = float(np.mean(signal_or_leads))
        
    duration_sec = sample_count / float(sampling_rate)
    expected_duration = STANDARD_DURATIONS_SEC[phase]
    all_finite = (nan_count == 0) and (inf_count == 0)
    
    # Check duration regularity
    is_irregular = (subject_id, phase) in IRREGULAR_RECORDINGS or abs(duration_sec - expected_duration) > 0.1
    
    return {
        "subject_id": subject_id,
        "phase": phase,
        "sampling_rate": sampling_rate,
        "lead_count": lead_count,
        "lead_count_valid": (lead_count == EXPECTED_LEAD_COUNT),
        "sample_count": sample_count,
        "lead_lengths": lead_lengths,
        "has_unequal_lead_lengths": has_unequal,
        "unequal_lead_lengths_detail": sorted(list(set(lead_lengths))) if has_unequal else None,
        "duration_sec": round(duration_sec, 2),
        "expected_duration_sec": expected_duration,
        "is_duration_irregular": is_irregular,
        "all_finite": all_finite,
        "nan_count": nan_count,
        "inf_count": inf_count,
        "flatline_leads": flatline_leads,
        "min_raw_val": min_val,
        "max_raw_val": max_val,
        "mean_raw_val": mean_val,
    }


def audit_subject_all_ecgs(
    subject_id: str,
    raw_dir: Path = RAW_DATA_DIR,
) -> Dict[str, Dict[str, Any]]:
    """Audit all 6 ECG files (3 phases x 2 sampling rates) for a subject."""
    audit_results = {}
    for phase in PHASES:
        for rate in SAMPLING_RATES:
            leads = load_raw_ecg_leads(subject_id, phase, rate, raw_dir=raw_dir)
            report = check_ecg_integrity(leads, subject_id, phase, rate)
            audit_results[f"{phase}_{rate}hz"] = report
    return audit_results

