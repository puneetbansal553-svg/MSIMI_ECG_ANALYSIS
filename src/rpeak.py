"""Top-level backward-compatibility module re-exporting from src.peak_detection.pan_tompkins.

Preserves the public API:
- detect_r_peaks_lead
- pan_tompkins_qrs_filter
- pan_tompkins_derivative
- moving_window_integration
"""

from src.peak_detection.pan_tompkins import (
    detect_r_peaks_lead,
    pan_tompkins_qrs_filter,
    pan_tompkins_derivative,
    moving_window_integration,
)

__all__ = [
    "detect_r_peaks_lead",
    "pan_tompkins_qrs_filter",
    "pan_tompkins_derivative",
    "moving_window_integration",
]
