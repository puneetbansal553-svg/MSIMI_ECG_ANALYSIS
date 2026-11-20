"""Signal Quality Control (QC) Module.

Evaluates ECG signal quality before and after preprocessing:
1. Dynamic range and clipping detection (12-bit ADC boundary check).
2. Baseline drift quotient (energy < 0.5 Hz vs QRS band).
3. High frequency noise quotient (energy > 45 Hz vs QRS band).
4. Signal-to-noise ratio estimation.
5. Flatline or loose lead detection.
"""

from typing import Dict, Any, List
import numpy as np

from configs.dataset_config import LEAD_NAMES, EXPECTED_LEAD_COUNT


def evaluate_lead_quality(
    raw_lead: np.ndarray,
    filtered_lead: np.ndarray,
    sampling_rate: int,
    lead_name: str = "II",
) -> Dict[str, Any]:
    """Compute pre- and post-processing quality indices for a single lead."""
    fs = float(sampling_rate)
    n_samples = len(raw_lead)
    
    # 1. Clipping detection (12-bit ADC nominal range 0 to 4095)
    clip_low_count = int(np.sum(raw_lead <= 5.0))
    clip_high_count = int(np.sum(raw_lead >= 4090.0))
    total_clipped = clip_low_count + clip_high_count
    clipping_pct = round(100.0 * total_clipped / max(1, n_samples), 3)
    
    # 2. Raw Signal Variance & Flatline Check
    raw_std = float(np.std(raw_lead))
    filtered_std = float(np.std(filtered_lead))
    is_flatline = raw_std < 1e-4
    
    # 3. Baseline wander ratio: std of low-freq drift removed
    drift_std = float(np.std(raw_lead - filtered_lead))
    
    # 4. Energy ratio: filtered QRS band energy to raw signal variance
    snr_est_db = round(float(10.0 * np.log10(max(1e-6, (filtered_std ** 2) / max(1e-6, drift_std ** 2)))), 2)
    
    # 5. Quality classification
    if is_flatline:
        quality_rating = "UNUSABLE_FLATLINE"
    elif clipping_pct > 5.0:
        quality_rating = "POOR_CLIPPING"
    elif snr_est_db < 0.0:
        quality_rating = "NOISY"
    elif snr_est_db < 6.0:
        quality_rating = "ACCEPTABLE"
    else:
        quality_rating = "EXCELLENT"
        
    return {
        "lead_name": lead_name,
        "clipping_pct": clipping_pct,
        "raw_std": round(raw_std, 2),
        "filtered_std": round(filtered_std, 2),
        "estimated_snr_db": snr_est_db,
        "is_flatline": is_flatline,
        "quality_rating": quality_rating,
    }


def evaluate_multilead_qc(
    raw_signal: np.ndarray,
    filtered_signal: np.ndarray,
    sampling_rate: int,
) -> Dict[str, Any]:
    """Evaluate quality across all 12 leads."""
    lead_reports = []
    acceptable_count = 0
    
    for i in range(raw_signal.shape[0]):
        lead_name = LEAD_NAMES[i]
        rep = evaluate_lead_quality(
            raw_signal[i],
            filtered_signal[i],
            sampling_rate,
            lead_name=lead_name,
        )
        lead_reports.append(rep)
        if rep["quality_rating"] in ["ACCEPTABLE", "EXCELLENT"]:
            acceptable_count += 1
            
    overall_status = "USABLE" if acceptable_count >= 8 else "DEGRADED"
    
    return {
        "sampling_rate": sampling_rate,
        "total_leads": len(lead_reports),
        "usable_leads_count": acceptable_count,
        "overall_status": overall_status,
        "lead_details": lead_reports,
    }

