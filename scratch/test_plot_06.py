import json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

project_root = Path.cwd()
summary_path = project_root / 'reports' / 'hrv' / '06_patient_hrv_summary.csv'
df_patient_summary = pd.read_csv(summary_path)

fig_out_dir = project_root / 'outputs' / 'figures'
fig_out_dir.mkdir(parents=True, exist_ok=True)
target_path = fig_out_dir / '06_heart_rate_recovery_dynamics.png'

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, axes = plt.subplots(2, 2, figsize=(16, 11), dpi=300)

# Filter for labeled cohort
df_labeled = df_patient_summary[df_patient_summary['label_status'].isin(['POSITIVE', 'NEGATIVE'])].copy()

# Panel A: Rest -> Stress -> Recovery HR Trajectories
ax_a = axes[0, 0]
phases = [0, 1, 2]
phase_names = ['Rest\n(360s)', 'Peak Stress\n(720s)', 'Recovery\n(360s)']

for grp, color, label in [('NEGATIVE', '#1f77b4', 'MSIMI-Negative (N=62)'),
                          ('POSITIVE', '#d62728', 'MSIMI-Positive (N=16)')]:
    sub = df_labeled[df_labeled['label_status'] == grp]
    hr_mat = sub[['rest_hr_100hz', 'stress_hr_100hz', 'recover_hr_100hz']].dropna().values
    means = np.mean(hr_mat, axis=0)
    sems = np.std(hr_mat, axis=0, ddof=1) / np.sqrt(len(hr_mat))
    
    # Plot individual faint trajectories
    for row in hr_mat:
        ax_a.plot(phases, row, color=color, alpha=0.15, linewidth=0.8)
    # Plot group mean with errorbar
    ax_a.errorbar(phases, means, yerr=sems, color=color, linewidth=2.5, marker='o',
                  markersize=7, capsize=5, label=f"{label} (mean +/- SEM)")

ax_a.set_xticks(phases)
ax_a.set_xticklabels(phase_names, fontsize=11, fontweight='bold')
ax_a.set_ylabel('Mean Heart Rate (BPM)', fontsize=11, fontweight='bold')
ax_a.set_title('A: Protocol Heart Rate Dynamics by MSIMI Status', fontsize=12, fontweight='bold', pad=10)
ax_a.legend(loc='upper right', frameon=True)

# Panel B: HRR 6-min Distribution by Clinical Subgroup
ax_b = axes[0, 1]
hrr_neg = df_labeled[df_labeled['label_status'] == 'NEGATIVE']['hrr_6min_100hz'].dropna()
hrr_pos = df_labeled[df_labeled['label_status'] == 'POSITIVE']['hrr_6min_100hz'].dropna()

bp = ax_b.boxplot([hrr_neg, hrr_pos], tick_labels=['MSIMI-Negative\n(N=62)', 'MSIMI-Positive\n(N=16)'],
                  patch_artist=True, medianprops=dict(color='black', linewidth=1.5))
bp['boxes'][0].set_facecolor('#abd9e9')
bp['boxes'][1].set_facecolor('#fdae61')

# Add jittered points
ax_b.scatter(np.random.normal(1, 0.04, size=len(hrr_neg)), hrr_neg, color='#1f77b4', alpha=0.6, s=30)
ax_b.scatter(np.random.normal(2, 0.04, size=len(hrr_pos)), hrr_pos, color='#d62728', alpha=0.6, s=30)
ax_b.axhline(12.0, color='gray', linestyle='--', linewidth=1, label='Normal Recovery Threshold (12 BPM)')
ax_b.set_ylabel('HRR_6min = HR_stress - HR_recover (BPM)', fontsize=11, fontweight='bold')
ax_b.set_title('B: 6-Minute Heart Rate Recovery (HRR_6min)', fontsize=12, fontweight='bold', pad=10)
ax_b.legend(loc='upper right', frameon=True)

# Panel C: Stress Delta HR vs Recovery HRR Scatter
ax_c = axes[1, 0]
for grp, color, marker, label in [('NEGATIVE', '#1f77b4', 'o', 'MSIMI-Negative'),
                                  ('POSITIVE', '#d62728', 's', 'MSIMI-Positive')]:
    sub = df_labeled[df_labeled['label_status'] == grp]
    ax_c.scatter(sub['delta_hr_stress_rest_100hz'], sub['hrr_6min_100hz'],
                 color=color, marker=marker, s=50, alpha=0.75, label=label, edgecolor='black', linewidth=0.5)

ax_c.axvline(0, color='gray', linestyle=':', alpha=0.6)
ax_c.axhline(0, color='gray', linestyle=':', alpha=0.6)
ax_c.set_xlabel('Stress Delta HR = HR_stress - HR_rest (BPM)', fontsize=11, fontweight='bold')
ax_c.set_ylabel('HRR_6min = HR_stress - HR_recover (BPM)', fontsize=11, fontweight='bold')
ax_c.set_title('C: Autonomic Reactivity vs Recovery Decoupling', fontsize=12, fontweight='bold', pad=10)
ax_c.legend(loc='lower right', frameon=True)

# Panel D: 100 Hz vs 500 Hz HRR Concordance
ax_d = axes[1, 1]
valid_both = df_labeled.dropna(subset=['hrr_6min_100hz', 'hrr_6min_500hz'])
x_hrr = valid_both['hrr_6min_100hz']
y_hrr = valid_both['hrr_6min_500hz']
ax_d.scatter(x_hrr, y_hrr, color='#2ca02c', alpha=0.75, s=40, edgecolor='black', linewidth=0.5)

# 1:1 Identity Line
min_val = min(x_hrr.min(), y_hrr.min()) - 2
max_val = max(x_hrr.max(), y_hrr.max()) + 2
ax_d.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.6, label='1:1 Line of Identity')

r_corr = np.corrcoef(x_hrr, y_hrr)[0, 1]
ax_d.annotate(f'Pearson r = {r_corr:.3f}\n(N={len(valid_both)} Paired Subjects)',
              xy=(0.05, 0.85), xycoords='axes fraction', fontsize=10, fontweight='bold',
              bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='gray', alpha=0.8))

ax_d.set_xlabel('HRR_6min at 100 Hz (BPM)', fontsize=11, fontweight='bold')
ax_d.set_ylabel('HRR_6min at 500 Hz (BPM)', fontsize=11, fontweight='bold')
ax_d.set_title('D: 100 Hz vs 500 Hz Sampling Concordance', fontsize=12, fontweight='bold', pad=10)
ax_d.legend(loc='lower right', frameon=True)

plt.tight_layout()
plt.savefig(target_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Successfully generated: {target_path} ({target_path.stat().st_size} bytes)")

