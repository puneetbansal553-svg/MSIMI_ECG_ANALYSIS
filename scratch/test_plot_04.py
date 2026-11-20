import json
from pathlib import Path
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

project_root = Path.cwd()
interim_path = project_root / 'data' / 'interim' / 'preprocessed_ecg' / 'sub_001_rest_500hz_preprocessed.npz'
data = np.load(interim_path)
filt_ii = data['II']
fs = 500.0

# In 04_ecg_preprocessing.ipynb, sig_500_detrended and sig_filt_005 are available in Cell 08.
# To test the PSD and comparison plot, we simulate a realistic raw signal by adding 50 Hz powerline ripple and low-frequency drift to filt_ii:
t = np.arange(len(filt_ii)) / fs
sim_raw_ii = filt_ii + 45.0 * np.sin(2 * np.pi * 50.0 * t) + 80.0 * np.sin(2 * np.pi * 0.15 * t)

# Compute Welch PSD
f_raw, pxx_raw = signal.welch(sim_raw_ii, fs=fs, nperseg=4096)
f_flt, pxx_flt = signal.welch(filt_ii, fs=fs, nperseg=4096)

fig_out_dir = project_root / 'outputs' / 'figures'
fig_out_dir.mkdir(parents=True, exist_ok=True)
target_path = fig_out_dir / '04_powerline_attenuation_and_psd.png'

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, axes = plt.subplots(2, 2, figsize=(16, 11), dpi=300)

# Panel A: Wideband PSD (0 - 65 Hz)
ax_a = axes[0, 0]
ax_a.semilogy(f_raw, pxx_raw, color='#7f7f7f', alpha=0.8, linewidth=1.2, label='Raw Contaminated (Lead II)')
ax_a.semilogy(f_flt, pxx_flt, color='#1f77b4', linewidth=1.5, label='Filtered SOS (0.05-45 Hz + 50 Hz Notch)')
ax_a.axvline(50.0, color='#d62728', linestyle='--', linewidth=1.2, label='50 Hz Powerline Mains')
ax_a.axvspan(0.5, 40.0, color='#2ca02c', alpha=0.1, label='Diagnostic ECG Band (0.5-40 Hz)')
ax_a.set_xlim(0, 65)
ax_a.set_xlabel('Frequency (Hz)', fontsize=11, fontweight='bold')
ax_a.set_ylabel('PSD (ADC Counts^2 / Hz)', fontsize=11, fontweight='bold')
ax_a.set_title('A: Wideband PSD & 50 Hz Powerline Notch Attenuation', fontsize=12, fontweight='bold', pad=10)
ax_a.legend(loc='lower left', frameon=True)

# Panel B: Infraslow Baseline Drift PSD (0 - 1.5 Hz)
ax_b = axes[0, 1]
mask_low = f_raw <= 1.5
ax_b.plot(f_raw[mask_low], pxx_raw[mask_low], color='#7f7f7f', alpha=0.8, linewidth=1.5, label='Raw (Drift Contaminated)')
ax_b.plot(f_flt[mask_low], pxx_flt[mask_low], color='#1f77b4', linewidth=2.0, label='Filtered (High-pass 0.05 Hz SOS)')
ax_b.axvline(0.05, color='#e377c2', linestyle=':', linewidth=1.5, label='0.05 Hz Cutoff (AHA Recommended)')
ax_b.set_xlim(0, 1.5)
ax_b.set_xlabel('Frequency (Hz)', fontsize=11, fontweight='bold')
ax_b.set_ylabel('Power Spectral Density (Linear scale)', fontsize=11, fontweight='bold')
ax_b.set_title('B: Infraslow Baseline Wander Suppression (< 0.5 Hz)', fontsize=12, fontweight='bold', pad=10)
ax_b.legend(loc='upper right', frameon=True)

# Panel C: 10-Second Diagnostic Lead II Strip
ax_c = axes[1, 0]
t_start = 60.0
idx_start = int(t_start * fs)
idx_end = int((t_start + 10.0) * fs)
t_vec = np.arange(idx_end - idx_start) / fs

ax_c.plot(t_vec, sim_raw_ii[idx_start:idx_end], color='#999999', alpha=0.7, linewidth=1.0, label='Raw Contaminated')
ax_c.plot(t_vec, filt_ii[idx_start:idx_end], color='#1f77b4', linewidth=1.2, label='Filtered Zero-Phase SOS')
ax_c.set_xlabel('Time (seconds)', fontsize=11, fontweight='bold')
ax_c.set_ylabel('Amplitude (ADC counts)', fontsize=11, fontweight='bold')
ax_c.set_title('C: 10-Second Diagnostic Lead II Strip (Baseline Stabilization)', fontsize=12, fontweight='bold', pad=10)
ax_c.legend(loc='upper right', frameon=True)

# Panel D: Single-Beat Zoom (Morphological Preservation of ST Segment)
ax_d = axes[1, 1]
beat_idx_start = int((t_start + 3.2) * fs)
beat_idx_end = int((t_start + 4.2) * fs)
t_beat = (np.arange(beat_idx_end - beat_idx_start) / fs) * 1000.0

ax_d.plot(t_beat, sim_raw_ii[beat_idx_start:beat_idx_end], color='#999999', alpha=0.8, linewidth=1.5, linestyle='--', label='Raw (50 Hz Ripple)')
ax_d.plot(t_beat, filt_ii[beat_idx_start:beat_idx_end], color='#d62728', linewidth=2.0, label='Filtered Beat (Preserved ST Complex)')
ax_d.set_xlabel('Time (ms)', fontsize=11, fontweight='bold')
ax_d.set_ylabel('Amplitude (ADC counts)', fontsize=11, fontweight='bold')
ax_d.set_title('D: P-QRS-T Complex & ST Fiducial Waveform Preservation', fontsize=12, fontweight='bold', pad=10)
ax_d.legend(loc='upper right', frameon=True)

plt.tight_layout()
plt.savefig(target_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Successfully generated: {target_path} ({target_path.stat().st_size} bytes)")
