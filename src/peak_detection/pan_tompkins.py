"""R-Peak Detection and RR Interval Profiling Module.

Pan-Tompkins Algorithm Implementation with Physiological Plausibility Checks:
1. QRS Bandpass Filtering (5 - 15 Hz)
2. Derivative Filtering
3. Non-linear Squaring
4. Moving Window Integration (150 ms)
5. Adaptive Dual-Threshold Peak Detection & Searchback
6. Refractory Period Blanking (200 ms)
7. Local R-Peak True Deflection Alignment
8. RR Interval Computation and Anomaly Profiling
"""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np

from configs.dataset_config import RPEAK_PARAMS, LEAD_NAMES
from src.preprocessing import (
    design_butterworth_sos_highpass,
    design_butterworth_sos_lowpass,
    apply_zero_phase_sos,
)


def pan_tompkins_qrs_filter(
    signal_1d: np.ndarray,
    fs: float,
    low_hz: float = 5.0,
    high_hz: float = 15.0,
    order: int = 2,
) -> np.ndarray:
    """Bandpass filter to isolate QRS energy (5 - 15 Hz) using Zero-Phase Butterworth SOS."""
    sos_hp = design_butterworth_sos_highpass(low_hz, fs, order=order)
    sos_lp = design_butterworth_sos_lowpass(high_hz, fs, order=order)
    s_hp = apply_zero_phase_sos(signal_1d, sos_hp)
    s_bp = apply_zero_phase_sos(s_hp, sos_lp)
    return s_bp


def pan_tompkins_derivative(signal_1d: np.ndarray, fs: float) -> np.ndarray:
    """Five-point derivative filter to emphasize QRS slope."""
    # y[n] = (1 / 8T) * (2x[n] + x[n-1] - x[n-3] - 2x[n-4])
    kernel = np.array([2.0, 1.0, 0.0, -1.0, -2.0]) * (fs / 8.0)
    return np.convolve(signal_1d, kernel, mode="same")


def moving_window_integration(signal_1d: np.ndarray, fs: float, window_ms: float = 150.0) -> np.ndarray:
    """Moving window integration (rectangular boxcar filter)."""
    window_samples = max(1, int(round((window_ms / 1000.0) * fs)))
    kernel = np.ones(window_samples) / float(window_samples)
    return np.convolve(signal_1d, kernel, mode="same")


def detect_r_peaks_lead(
    ecg_lead: np.ndarray,
    sampling_rate: int,
    lead_name: str = "II",
    params: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Detect R-peaks on a single ECG lead using Pan-Tompkins algorithm.
    
    Returns:
        Dict with:
            r_peaks: array of sample indices of detected R-peaks
            rr_intervals_ms: array of RR intervals in milliseconds
            mean_hr_bpm: estimated mean heart rate
            anomaly_stats: counts of ectopic/short/long RR intervals
    """
    if params is None:
        params = RPEAK_PARAMS
    
    fs = float(sampling_rate)
    refractory_samples = int((params.get("refractory_ms", 200.0) / 1000.0) * fs)
    min_rr_samples = int((params.get("min_rr_ms", 300.0) / 1000.0) * fs)
    max_rr_samples = int((params.get("max_rr_ms", 2000.0) / 1000.0) * fs)
    searchback_mult = params.get("searchback_rr_factor", 1.66)
    
    # 1. Bandpass filter (5-15 Hz)
    filtered = pan_tompkins_qrs_filter(ecg_lead, fs)
    
    # 2. Derivative
    derived = pan_tompkins_derivative(filtered, fs)
    
    # 3. Squaring
    squared = derived ** 2
    
    # 4. Moving Window Integration (150 ms)
    integrated = moving_window_integration(squared, fs, window_ms=150.0)
    
    # 5. Adaptive Thresholding
    # Find all local maxima in the integrated signal
    diff = np.diff(integrated)
    local_peaks = np.where((diff[:-1] > 0) & (diff[1:] <= 0))[0] + 1
    
    if len(local_peaks) == 0:
        return {
            "lead_name": lead_name,
            "r_peaks": np.array([], dtype=int),
            "rr_intervals_ms": np.array([], dtype=float),
            "mean_hr_bpm": 0.0,
            "anomaly_stats": {"short_rr": 0, "long_rr": 0, "total_rr": 0},
        }
    
    # Initialize adaptive thresholds using first 2 seconds of signal
    init_window = int(min(len(integrated), fs * 2))
    init_peaks = [p for p in local_peaks if p < init_window]
    
    if init_peaks:
        spki = float(np.percentile([integrated[p] for p in init_peaks], 75))
        npki = float(np.percentile([integrated[p] for p in init_peaks], 25))
    else:
        spki = float(np.max(integrated[:init_window])) * 0.5
        npki = spki * 0.1
    
    spki = max(spki, 1e-6)
    npki = max(npki, 1e-7)
    
    threshold_i1 = npki + 0.25 * (spki - npki)
    threshold_i2 = 0.5 * threshold_i1
    
    qrs_peaks = []
    last_peak = -refractory_samples
    recent_rr = [int(0.8 * fs)]  # Seed nominal 75 bpm (0.8s)
    
    for p in local_peaks:
        # Respect refractory period
        if (p - last_peak) < refractory_samples:
            continue
        
        peak_val = integrated[p]
        rr_current = p - last_peak if last_peak >= 0 else int(0.8 * fs)
        rr_avg = float(np.mean(recent_rr[-8:]))
        
        # Check for missed beat via searchback
        if last_peak >= 0 and (p - last_peak) > (searchback_mult * rr_avg):
            # Look between last_peak and p for candidates above threshold_i2
            search_start = last_peak + refractory_samples
            search_end = p - refractory_samples
            candidates = [c for c in local_peaks if search_start <= c <= search_end and integrated[c] >= threshold_i2]
            if candidates:
                best_c = max(candidates, key=lambda c: integrated[c])
                qrs_peaks.append(best_c)
                spki = 0.25 * integrated[best_c] + 0.75 * spki
                threshold_i1 = npki + 0.25 * (spki - npki)
                threshold_i2 = 0.5 * threshold_i1
                recent_rr.append(best_c - last_peak)
                last_peak = best_c
        
        # Enforce refractory period relative to last_peak (including searchback beat)
        if (p - last_peak) < refractory_samples:
            continue
        
        # Standard threshold check
        if peak_val >= threshold_i1:
            qrs_peaks.append(p)
            spki = 0.125 * peak_val + 0.875 * spki
            if last_peak >= 0:
                recent_rr.append(p - last_peak)
            last_peak = p
        else:
            npki = 0.125 * peak_val + 0.875 * npki
        
        threshold_i1 = npki + 0.25 * (spki - npki)
        threshold_i2 = 0.5 * threshold_i1
    
    # 6. Alignment: Find the true maximum peak in the filtered ECG
    # Look in window [-100 ms, +50 ms] around integrated peak
    search_left = int(0.100 * fs)
    search_right = int(0.050 * fs)
    
    aligned_r_peaks = []
    sig_len = len(ecg_lead)
    
    for q_idx in qrs_peaks:
        w_start = max(0, q_idx - search_left)
        w_end = min(sig_len, q_idx + search_right)
        if w_end > w_start:
            # Find peak deflection (maximum absolute or maximum positive)
            local_region = ecg_lead[w_start:w_end]
            # Max positive deflection
            true_r = w_start + int(np.argmax(local_region))
            aligned_r_peaks.append(true_r)
    
    # 6b. Post-alignment refractory blanking (physiological minimum 200 ms)
    # If two peaks align to the same complex or within refractory period, retain larger deflection
    cleaned_r_peaks = []
    for r_cand in sorted(list(set(aligned_r_peaks))):
        if not cleaned_r_peaks or (r_cand - cleaned_r_peaks[-1]) >= refractory_samples:
            cleaned_r_peaks.append(r_cand)
        else:
            if ecg_lead[r_cand] > ecg_lead[cleaned_r_peaks[-1]]:
                cleaned_r_peaks[-1] = r_cand
                
    aligned_r_peaks = np.array(cleaned_r_peaks, dtype=int)
    
    # 7. Compute RR intervals in milliseconds
    if len(aligned_r_peaks) > 1:
        rr_intervals_ms = np.diff(aligned_r_peaks) * (1000.0 / fs)
    else:
        rr_intervals_ms = np.array([], dtype=float)
    
    # 8. Anomaly profiling (physiological outliers)
    short_rr = int(np.sum(rr_intervals_ms < (min_rr_samples * 1000.0 / fs)))
    long_rr = int(np.sum(rr_intervals_ms > (max_rr_samples * 1000.0 / fs)))
    mean_hr_bpm = round(float(60000.0 / np.mean(rr_intervals_ms)), 1) if len(rr_intervals_ms) > 0 else 0.0
    
    return {
        "lead_name": lead_name,
        "sampling_rate": sampling_rate,
        "r_peaks": aligned_r_peaks,
        "r_peak_count": len(aligned_r_peaks),
        "rr_intervals_ms": rr_intervals_ms,
        "mean_hr_bpm": mean_hr_bpm,
        "median_rr_ms": round(float(np.median(rr_intervals_ms)), 1) if len(rr_intervals_ms) > 0 else 0.0,
        "std_rr_ms": round(float(np.std(rr_intervals_ms)), 1) if len(rr_intervals_ms) > 0 else 0.0,
        "anomaly_stats": {
            "total_rr": len(rr_intervals_ms),
            "short_rr_count (<300ms)": short_rr,
            "long_rr_count (>2000ms)": long_rr,
            "physiologically_implausible_pct": round(
                100.0 * (short_rr + long_rr) / max(1, len(rr_intervals_ms)), 2
            ),
        },
    }

