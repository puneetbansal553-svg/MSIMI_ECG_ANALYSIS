"""ECG Signal Preprocessing and Baseline Correction Module.

Numerically stable implementation using Second-Order Sections (SOS) Butterworth filters
with forward-backward zero-phase filtering (`scipy.signal.sosfiltfilt`).

Parameters and pipeline features:
- Baseline offset subtraction: empirical lead-wise median subtraction (zero-centered ADC count deviations).
- Highpass baseline wander removal: configurable Butterworth SOS (empirical evaluation of 0.05 Hz vs 0.5 Hz).
- Lowpass noise removal: 45.0 Hz cutoff Butterworth SOS.
- Powerline notch filter: 50.0 Hz notch (applied at 500 Hz; evaluated against lowpass alone; Q=30, bandwidth ~1.67 Hz).
- Lead-by-lead independent filtering: preserves exact original lead lengths without padding or truncation.
- Strict error handling: raises explicit exceptions without automatic fallback.
"""

from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import scipy.signal

from configs.dataset_config import (
    ADC_MIDSCALE_OFFSET,
    FILTER_PARAMS,
    LEAD_NAMES,
)


def remove_adc_offset_lead(lead_1d: np.ndarray) -> np.ndarray:
    """Subtract empirical median to zero-center around baseline while preserving ADC count deviation scale."""
    med = float(np.median(lead_1d))
    return lead_1d - med


def remove_adc_offset(signal: np.ndarray, nominal_offset: float = ADC_MIDSCALE_OFFSET) -> np.ndarray:
    """Subtract nominal offset and empirical lead-wise medians for a 2D matrix (n_leads, n_samples)."""
    centered = signal - nominal_offset
    lead_medians = np.median(centered, axis=1, keepdims=True)
    return centered - lead_medians


def design_butterworth_sos_highpass(cutoff_hz: float, fs: float, order: int = 2) -> np.ndarray:
    """Design a Butterworth highpass filter in Second-Order Sections (SOS)."""
    nyq = 0.5 * fs
    if cutoff_hz <= 0 or cutoff_hz >= nyq:
        raise ValueError(f"Highpass cutoff {cutoff_hz} Hz must be strictly between 0 and Nyquist {nyq} Hz")
    normal_cutoff = cutoff_hz / nyq
    return scipy.signal.butter(order, normal_cutoff, btype="highpass", output="sos")


def design_butterworth_sos_lowpass(cutoff_hz: float, fs: float, order: int = 4) -> np.ndarray:
    """Design a Butterworth lowpass filter in Second-Order Sections (SOS)."""
    nyq = 0.5 * fs
    if cutoff_hz <= 0 or cutoff_hz >= nyq:
        raise ValueError(f"Lowpass cutoff {cutoff_hz} Hz must be strictly between 0 and Nyquist {nyq} Hz")
    normal_cutoff = cutoff_hz / nyq
    return scipy.signal.butter(order, normal_cutoff, btype="lowpass", output="sos")


def design_butterworth_sos_notch(notch_hz: float, fs: float, q: float = 30.0) -> np.ndarray:
    """Design a digital IIR notch filter in Second-Order Sections (SOS).
    
    Bandwidth delta_f = notch_hz / q. For notch_hz=50.0, q=30.0, delta_f = 1.6667 Hz.
    """
    nyq = 0.5 * fs
    if notch_hz >= nyq:
        raise ValueError(f"Notch frequency {notch_hz} Hz must be strictly below Nyquist {nyq} Hz")
    w0 = notch_hz / nyq
    b, a = scipy.signal.iirnotch(w0, q)
    return scipy.signal.tf2sos(b, a)


def apply_zero_phase_sos(signal_1d: np.ndarray, sos: np.ndarray, padlen: Optional[int] = None) -> np.ndarray:
    """Apply zero-phase digital filtering in forward and backward directions using cascaded SOS.
    
    Zero-phase behavior is guaranteed in the signal interior; boundary reflection transients occur at edges.
    Strict error handling: raises ValueError if signal length is insufficient for padding; no automatic fallback.
    """
    n_samples = len(signal_1d)
    if padlen is None:
        padlen = 3 * (2 * len(sos) + 1 - min((sos[:, 2] == 0).sum(), (sos[:, 5] == 0).sum()))
    
    if n_samples <= padlen:
        raise ValueError(
            f"Signal length ({n_samples}) must be strictly greater than padlen ({padlen}). "
            "Cannot apply sosfiltfilt without sufficient padding."
        )
    
    return scipy.signal.sosfiltfilt(sos, signal_1d, padtype="odd", padlen=padlen)


def preprocess_single_lead(
    lead_1d: np.ndarray,
    fs: float,
    hp_sos: Optional[np.ndarray] = None,
    lp_sos: Optional[np.ndarray] = None,
    notch_sos: Optional[np.ndarray] = None,
    padlen: Optional[int] = None,
) -> np.ndarray:
    """Preprocess a single 1D lead trace. Exact sample count is strictly preserved (N_out == N_in)."""
    # 1. Median baseline subtraction (zero-centered ADC count deviation)
    centered = remove_adc_offset_lead(lead_1d)
    
    # 2. Highpass filter
    out = apply_zero_phase_sos(centered, hp_sos, padlen=padlen) if hp_sos is not None else centered
    
    # 3. Lowpass filter
    if lp_sos is not None:
        out = apply_zero_phase_sos(out, lp_sos, padlen=padlen)
        
    # 4. Notch filter (if provided)
    if notch_sos is not None:
        out = apply_zero_phase_sos(out, notch_sos, padlen=padlen)
        
    return out


def preprocess_ecg_leads(
    leads: List[np.ndarray],
    sampling_rate: int,
    hp_cutoff_hz: float = 0.5,
    lp_cutoff_hz: float = 45.0,
    apply_notch: bool = False,
    notch_hz: float = 50.0,
    notch_q: float = 30.0,
) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    """Preprocess 12-lead ECG recording as a list of 1D arrays, preserving individual lead lengths.
    
    No truncation, padding, or interpolation is performed.
    """
    fs = float(sampling_rate)
    hp_sos = design_butterworth_sos_highpass(hp_cutoff_hz, fs, order=2) if hp_cutoff_hz > 0 else None
    lp_sos = design_butterworth_sos_lowpass(lp_cutoff_hz, fs, order=4) if lp_cutoff_hz > 0 else None
    
    # Notch only permitted if strictly below Nyquist
    if apply_notch and notch_hz < (fs / 2.0):
        notch_sos = design_butterworth_sos_notch(notch_hz, fs, q=notch_q)
    else:
        notch_sos = None
        
    filtered_leads = []
    lead_lengths = []
    
    for idx, lead_arr in enumerate(leads):
        lead_lengths.append(len(lead_arr))
        filt_lead = preprocess_single_lead(
            lead_arr,
            fs=fs,
            hp_sos=hp_sos,
            lp_sos=lp_sos,
            notch_sos=notch_sos,
        )
        filtered_leads.append(filt_lead)
        
    # Empirical boundary transient status:
    # A single universal edge exclusion boundary cannot be scientifically claimed without downstream
    # fiducial delineation. Marked as NOT_ESTABLISHED to prevent premature assumption of validated boundaries.
    meta = {
        "sampling_rate": sampling_rate,
        "hp_cutoff_hz": hp_cutoff_hz,
        "lp_cutoff_hz": lp_cutoff_hz,
        "notch_applied": (notch_sos is not None),
        "notch_hz": notch_hz if (notch_sos is not None) else None,
        "notch_q": notch_q if (notch_sos is not None) else None,
        "notch_bandwidth_hz": round(notch_hz / notch_q, 2) if (notch_sos is not None) else None,
        "filter_type": "Zero-Phase Butterworth SOS (sosfiltfilt)",
        "edge_exclusion_sec": "NOT_ESTABLISHED",
        "lead_lengths": lead_lengths,
        "has_unequal_lead_lengths": (len(set(lead_lengths)) > 1),
    }
    return filtered_leads, meta


def preprocess_ecg(
    signal: np.ndarray,
    sampling_rate: int,
    filter_params: Optional[Dict[str, float]] = None,
    apply_notch: bool = False,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """2D matrix wrapper around preprocess_ecg_leads for equal-length leads."""
    leads = [signal[i] for i in range(signal.shape[0])]
    hp = filter_params.get("highpass_hz", 0.5) if filter_params else 0.5
    lp = filter_params.get("lowpass_hz", 45.0) if filter_params else 45.0
    notch = filter_params.get("notch_hz", 50.0) if filter_params else 50.0
    filt_leads, meta = preprocess_ecg_leads(
        leads,
        sampling_rate,
        hp_cutoff_hz=hp,
        lp_cutoff_hz=lp,
        apply_notch=apply_notch,
        notch_hz=notch,
    )
    return np.vstack(filt_leads), meta
