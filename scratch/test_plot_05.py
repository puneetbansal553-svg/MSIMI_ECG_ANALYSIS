import json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

project_root = Path.cwd()
manifest_path = project_root / 'reports' / 'rpeaks' / '05_rpeak_detection_manifest.csv'
df_rpeaks = pd.read_csv(manifest_path)
valid_df = df_rpeaks[df_rpeaks['rpeak_detection_status'] == 'SUCCESS'].copy()

fig_out_dir = project_root / 'outputs' / 'figures'
fig_out_dir.mkdir(parents=True, exist_ok=True)
target_path = fig_out_dir / '05_cohort_rr_distributions_and_plausibility.png'

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, axes = plt.subplots(2, 2, figsize=(16, 11), dpi=300)

phases_order = ['rest', 'stress', 'recover']
phase_labels = ['Rest', 'Stress', 'Recover']

# Panel A: Heart Rate Distribution by Phase and Sampling Rate
ax_a = axes[0, 0]
hr_data_100 = [valid_df[(valid_df['phase'] == p) & (valid_df['sampling_rate'] == 100)]['rhythm_mean_hr_bpm'].dropna() for p in phases_order]
hr_data_500 = [valid_df[(valid_df['phase'] == p) & (valid_df['sampling_rate'] == 500)]['rhythm_mean_hr_bpm'].dropna() for p in phases_order]

x = np.arange(3)
w = 0.35
bp1 = ax_a.boxplot(hr_data_100, positions=x - w/2, widths=0.28, patch_artist=True,
                   boxprops=dict(facecolor='#4575b4', alpha=0.8), medianprops=dict(color='black', linewidth=1.5))
bp2 = ax_a.boxplot(hr_data_500, positions=x + w/2, widths=0.28, patch_artist=True,
                   boxprops=dict(facecolor='#f46d43', alpha=0.8), medianprops=dict(color='black', linewidth=1.5))

ax_a.set_xticks(x)
ax_a.set_xticklabels(phase_labels, fontsize=11, fontweight='bold')
ax_a.set_ylabel('Mean Heart Rate (BPM)', fontsize=11, fontweight='bold')
ax_a.set_title('A: Phase-Stratified Heart Rate Distributions (100 Hz vs 500 Hz)', fontsize=12, fontweight='bold', pad=10)
# Custom legend
from matplotlib.patches import Patch
ax_a.legend([Patch(facecolor='#4575b4', alpha=0.8), Patch(facecolor='#f46d43', alpha=0.8)],
            ['100 Hz', '500 Hz'], loc='upper right', frameon=True)

# Panel B: Selected Rhythm Lead Distribution
ax_b = axes[0, 1]
lead_counts = valid_df['selected_rhythm_lead'].value_counts()
colors_b = ['#2b5c8f', '#41b6c4', '#7fcdbb', '#df65b0']
bars_b = ax_b.bar(lead_counts.index, lead_counts.values, color=colors_b[:len(lead_counts)], edgecolor='black', width=0.5)
ax_b.set_xlabel('Selected Primary Rhythm Lead', fontsize=11, fontweight='bold')
ax_b.set_ylabel('Number of Recordings', fontsize=11, fontweight='bold')
ax_b.set_title('B: Rhythm Lead Hierarchy Utilization (N=480)', fontsize=12, fontweight='bold', pad=10)
for bar in bars_b:
    h = bar.get_height()
    ax_b.annotate(f'{h} ({h/len(valid_df)*100:.1f}%)',
                  xy=(bar.get_x() + bar.get_width()/2, h), xytext=(0, 3),
                  textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')

# Panel C: Implausible Beat Outlier Percentage by Phase
ax_c = axes[1, 0]
imp_data = [valid_df[valid_df['phase'] == p]['rhythm_implausible_pct'].dropna() for p in phases_order]
bp_c = ax_c.boxplot(imp_data, tick_labels=phase_labels, patch_artist=True,
                    boxprops=dict(facecolor='#abd9e9', alpha=0.9), medianprops=dict(color='red', linewidth=1.5))
ax_c.set_ylabel('Implausible RR Interval Ratio (%)', fontsize=11, fontweight='bold')
ax_c.set_title('C: Physiological Outlier Rate (< 300 ms or > 2000 ms)', fontsize=12, fontweight='bold', pad=10)
ax_c.set_ylim(-0.05, max(valid_df['rhythm_implausible_pct'].max() * 1.2, 0.5))

# Panel D: Lead V6 Peak Count vs Selected Rhythm Lead Peak Count
ax_d = axes[1, 1]
is_shortened = valid_df['lead_v6_sample_count'] < (valid_df['sampling_rate'] * 350)
ax_d.scatter(valid_df[~is_shortened]['rhythm_peak_count'], valid_df[~is_shortened]['lead_v6_peak_count'],
             color='#1f77b4', alpha=0.6, s=25, label='Normal Lead V6 (N=395)')
ax_d.scatter(valid_df[is_shortened]['rhythm_peak_count'], valid_df[is_shortened]['lead_v6_peak_count'],
             color='#d62728', alpha=0.8, s=35, marker='^', label='Shortened Lead V6 (N=85)')
# Identity line
min_p = min(valid_df['rhythm_peak_count'].min(), valid_df['lead_v6_peak_count'].min())
max_p = max(valid_df['rhythm_peak_count'].max(), valid_df['lead_v6_peak_count'].max())
ax_d.plot([min_p, max_p], [min_p, max_p], 'k--', alpha=0.5, label='1:1 Line of Identity')
ax_d.set_xlabel('Selected Rhythm Lead R-Peak Count', fontsize=11, fontweight='bold')
ax_d.set_ylabel('Lead V6 Detected R-Peak Count', fontsize=11, fontweight='bold')
ax_d.set_title('D: Precordial Lead V6 Peak Concordance & Deficit Validation', fontsize=12, fontweight='bold', pad=10)
ax_d.legend(loc='lower right', frameon=True)

plt.tight_layout()
plt.savefig(target_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Successfully generated: {target_path} ({target_path.stat().st_size} bytes)")

