import json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

project_root = Path.cwd()
summary_path = project_root / 'reports' / 'segmentation' / '07_patient_beat_summary.csv'
df_patient_summary = pd.read_csv(summary_path)

beat_npz_path = project_root / 'data' / 'processed' / 'beat_segments' / 'sub_001_rest_500hz_beats.npz'
beat_data = np.load(beat_npz_path)
beats = beat_data['beats']  # (N_beats, 12, 400)
lead_names = list(beat_data['lead_names'])

fig_out_dir = project_root / 'outputs' / 'figures'
fig_out_dir.mkdir(parents=True, exist_ok=True)
target_path = fig_out_dir / '07_beat_alignment_and_cohort_counts.png'

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, axes = plt.subplots(2, 2, figsize=(16, 11), dpi=300)

fs = 500.0
n_samples = beats.shape[2]
# In Stage 07, window is -250 ms to +550 ms (or -250 ms to +450 ms) -> sample 125 is R-peak at 500 Hz
r_peak_idx = int(0.250 * fs)
t_ms = (np.arange(n_samples) - r_peak_idx) * (1000.0 / fs)

# Panel A: Lead II Ensemble Beat Alignment
ax_a = axes[0, 0]
lead_ii_idx = lead_names.index('II')
n_plot_beats = min(100, beats.shape[0])
for b_i in range(n_plot_beats):
    ax_a.plot(t_ms, beats[b_i, lead_ii_idx, :], color='#4575b4', alpha=0.08, linewidth=0.8)

mean_beat_ii = np.mean(beats[:n_plot_beats, lead_ii_idx, :], axis=0)
std_beat_ii = np.std(beats[:n_plot_beats, lead_ii_idx, :], axis=0)
ax_a.plot(t_ms, mean_beat_ii, color='#d62728', linewidth=2.5, label='Mean Beat Profile')
ax_a.fill_between(t_ms, mean_beat_ii - std_beat_ii, mean_beat_ii + std_beat_ii, color='#d62728', alpha=0.2, label='+/- 1 SD Envelope')
ax_a.axvline(0, color='black', linestyle='--', linewidth=1.2, label='R-Peak Alignment Fiducial (t=0)')
ax_a.set_xlabel('Time Relative to R-Peak (ms)', fontsize=11, fontweight='bold')
ax_a.set_ylabel('Amplitude (ADC counts)', fontsize=11, fontweight='bold')
ax_a.set_title(f'A: Lead II Ensemble Beat Window Alignment (N={n_plot_beats} Beats)', fontsize=12, fontweight='bold', pad=10)
ax_a.legend(loc='upper right', frameon=True)

# Panel B: Lead V5 Ensemble Beat Alignment
ax_b = axes[0, 1]
lead_v5_idx = lead_names.index('V5')
for b_i in range(n_plot_beats):
    ax_b.plot(t_ms, beats[b_i, lead_v5_idx, :], color='#74add1', alpha=0.08, linewidth=0.8)

mean_beat_v5 = np.mean(beats[:n_plot_beats, lead_v5_idx, :], axis=0)
std_beat_v5 = np.std(beats[:n_plot_beats, lead_v5_idx, :], axis=0)
ax_b.plot(t_ms, mean_beat_v5, color='#1f77b4', linewidth=2.5, label='Mean Beat Profile')
ax_b.fill_between(t_ms, mean_beat_v5 - std_beat_v5, mean_beat_v5 + std_beat_v5, color='#1f77b4', alpha=0.2, label='+/- 1 SD Envelope')
ax_b.axvline(0, color='black', linestyle='--', linewidth=1.2, label='R-Peak Alignment Fiducial (t=0)')
ax_b.set_xlabel('Time Relative to R-Peak (ms)', fontsize=11, fontweight='bold')
ax_b.set_ylabel('Amplitude (ADC counts)', fontsize=11, fontweight='bold')
ax_b.set_title(f'B: Precordial Lead V5 Ensemble Alignment (N={n_plot_beats} Beats)', fontsize=12, fontweight='bold', pad=10)
ax_b.legend(loc='upper right', frameon=True)

# Panel C: Cohort Total Peak Counts by Phase
ax_c = axes[1, 0]
phases = ['Rest (360s)', 'Stress (720s)', 'Recovery (360s)']
total_rest = df_patient_summary['rest_peaks'].sum()
total_stress = df_patient_summary['stress_peaks'].sum()
total_rec = df_patient_summary['recover_peaks'].sum()
peak_sums = [total_rest, total_stress, total_rec]

colors_c = ['#3182bd', '#e6550d', '#31a354']
bars_c = ax_c.bar(phases, peak_sums, color=colors_c, edgecolor='black', width=0.5)
ax_c.set_ylabel('Total Detected R-Peaks Across Cohort', fontsize=11, fontweight='bold')
ax_c.set_title('C: Cumulative Beat Detection Yield by Phase (N=80 Subjects)', fontsize=12, fontweight='bold', pad=10)
for bar in bars_c:
    h = bar.get_height()
    ax_c.annotate(f'{int(h):,}', xy=(bar.get_x() + bar.get_width()/2, h), xytext=(0, 3),
                  textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')

# Panel D: Per-Patient Eligible Beat Count by MSIMI Status
ax_d = axes[1, 1]
df_labeled = df_patient_summary[df_patient_summary['label_status'].isin(['POSITIVE', 'NEGATIVE'])].copy()
beats_neg = df_labeled[df_labeled['label_status'] == 'NEGATIVE']['patient_eligible_beats'].dropna()
beats_pos = df_labeled[df_labeled['label_status'] == 'POSITIVE']['patient_eligible_beats'].dropna()

bp_d = ax_d.boxplot([beats_neg, beats_pos], tick_labels=['MSIMI-Negative\n(N=62)', 'MSIMI-Positive\n(N=16)'],
                    patch_artist=True, medianprops=dict(color='black', linewidth=1.5))
bp_d['boxes'][0].set_facecolor('#abd9e9')
bp_d['boxes'][1].set_facecolor('#fdae61')

ax_d.scatter(np.random.normal(1, 0.04, size=len(beats_neg)), beats_neg, color='#1f77b4', alpha=0.6, s=30)
ax_d.scatter(np.random.normal(2, 0.04, size=len(beats_pos)), beats_pos, color='#d62728', alpha=0.6, s=30)
ax_d.set_ylabel('Eligible Segmented Beats Per Patient', fontsize=11, fontweight='bold')
ax_d.set_title('D: Patient-Level Segmented Beat Yield Distribution', fontsize=12, fontweight='bold', pad=10)

plt.tight_layout()
plt.savefig(target_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Successfully generated: {target_path} ({target_path.stat().st_size} bytes)")

