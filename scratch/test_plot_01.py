import json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

project_root = Path.cwd()
df_rec_qc = pd.read_csv(project_root / 'reports' / 'qc' / '03_ecg_recording_quality_audit.csv')

# In 01_dataset_exploration.ipynb, df_records is defined in Cell 06 with columns:
# ['subject_id', 'phase', 'rate', 'lead_count', 'duration_sec', 'nominal_sec', 'has_unequal_leads', 'lead_lengths', 'is_dur_irregular']
df_records = pd.DataFrame({
    'subject_id': df_rec_qc['subject_id'],
    'phase': df_rec_qc['phase'],
    'rate': df_rec_qc['sampling_rate'],
    'lead_count': df_rec_qc['lead_count'],
    'duration_sec': df_rec_qc['max_duration_sec'],
    'nominal_sec': df_rec_qc['expected_duration_sec'],
    'has_unequal_leads': df_rec_qc['has_unequal_lead_lengths'],
    'is_dur_irregular': df_rec_qc['is_duration_irregular']
})

fig_out_dir = project_root / 'outputs' / 'figures'
fig_out_dir.mkdir(parents=True, exist_ok=True)
target_path = fig_out_dir / '01_recording_completeness_and_sampling_distribution.png'

# Visual styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, axes = plt.subplots(2, 2, figsize=(15, 11), dpi=300)

palette_phase = {'rest': '#1f77b4', 'stress': '#d62728', 'recover': '#2ca02c'}
phase_labels = {'rest': 'Rest (360s)', 'stress': 'Stress (720s)', 'recover': 'Recovery (360s)'}

# Panel A: Recording Count per Phase stratified by Sampling Frequency
ax_a = axes[0, 0]
counts_100 = [len(df_records[(df_records['rate'] == 100) & (df_records['phase'] == p)]) for p in ['rest', 'stress', 'recover']]
counts_500 = [len(df_records[(df_records['rate'] == 500) & (df_records['phase'] == p)]) for p in ['rest', 'stress', 'recover']]
x = np.arange(3)
width = 0.35
rects1 = ax_a.bar(x - width/2, counts_100, width, label='100 Hz', color='#4575b4', edgecolor='black', linewidth=0.8)
rects2 = ax_a.bar(x + width/2, counts_500, width, label='500 Hz', color='#f46d43', edgecolor='black', linewidth=0.8)
ax_a.set_ylabel('Number of Verified Recordings', fontsize=11, fontweight='bold')
ax_a.set_title('A: Protocol Recording Completeness (N=480 Total)', fontsize=12, fontweight='bold', pad=10)
ax_a.set_xticks(x)
ax_a.set_xticklabels(['Rest', 'Stress', 'Recover'], fontsize=11)
ax_a.set_ylim(0, 100)
ax_a.axhline(80, color='gray', linestyle='--', linewidth=1, alpha=0.7, label='Target Cohort (N=80)')
for bar in rects1 + rects2:
    h = bar.get_height()
    ax_a.annotate(f'{h}', xy=(bar.get_x() + bar.get_width()/2, h), xytext=(0, 3),
                  textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')
ax_a.legend(loc='lower right', frameon=True)

# Panel B: Duration Distribution Across Protocol Phases
ax_b = axes[0, 1]
box_data = [df_records[df_records['phase'] == p]['duration_sec'].dropna() for p in ['rest', 'stress', 'recover']]
bplot = ax_b.boxplot(box_data, tick_labels=['Rest', 'Stress', 'Recover'], patch_artist=True,
                     medianprops=dict(color='black', linewidth=1.5))
colors = ['#abd9e9', '#fdae61', '#a6d96a']
for patch, color in zip(bplot['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_edgecolor('black')
# Plot nominal lines
ax_b.scatter([1, 2, 3], [360, 720, 360], color='red', marker='D', s=50, zorder=5, label='Nominal Duration')
ax_b.set_ylabel('Acquired Duration (seconds)', fontsize=11, fontweight='bold')
ax_b.set_title('B: Protocol Duration Fidelity by Phase', fontsize=12, fontweight='bold', pad=10)
ax_b.legend(loc='upper right', frameon=True)

# Panel C: Duration Delta Distribution (Acquired - Nominal)
ax_c = axes[1, 0]
df_records['dur_delta'] = df_records['duration_sec'] - df_records['nominal_sec']
for p, color in palette_phase.items():
    sub_deltas = df_records[df_records['phase'] == p]['dur_delta']
    ax_c.hist(sub_deltas, bins=25, alpha=0.6, color=color, label=f"{phase_labels[p]} (mean: {sub_deltas.mean():+.1f}s)", edgecolor='black', linewidth=0.5)
ax_c.axvline(0, color='black', linestyle='--', linewidth=1.2, label='Nominal Zero Delta')
ax_c.set_xlabel('Duration Delta: Acquired - Nominal (seconds)', fontsize=11, fontweight='bold')
ax_c.set_ylabel('Recording Frequency', fontsize=11, fontweight='bold')
ax_c.set_title('C: Duration Variance from Protocol Specification', fontsize=12, fontweight='bold', pad=10)
ax_c.legend(loc='upper left', frameon=True, fontsize=9)

# Panel D: 12-Lead Acquisition Compliance & Length Uniformity
ax_d = axes[1, 1]
compliance_categories = [
    '12-Lead Complete\n(100% of 480)',
    'Standard Duration\n(98.3% of 480)',
    'Equal Lead Lengths\n(82.3% of 480)',
    'Shortened Lead V6\n(17.7% of 480)'
]
n_total = len(df_records)
n_12lead = (df_records['lead_count'] == 12).sum()
n_std_dur = (df_records['is_dur_irregular'] != True).sum()
n_unequal = (df_records['has_unequal_leads'] == True).sum()
n_equal = n_total - n_unequal

percentages = [
    n_12lead / n_total * 100,
    n_std_dur / n_total * 100,
    n_equal / n_total * 100,
    n_unequal / n_total * 100
]
bar_colors = ['#2ca02c', '#1f77b4', '#17becf', '#ff7f0e']
bars = ax_d.barh(compliance_categories[::-1], percentages[::-1], color=bar_colors[::-1], edgecolor='black', linewidth=0.8)
ax_d.set_xlabel('Cohort Compliance (%)', fontsize=11, fontweight='bold')
ax_d.set_title('D: 12-Lead Signal Architecture & Integrity Profile', fontsize=12, fontweight='bold', pad=10)
ax_d.set_xlim(0, 115)
for bar in bars:
    w = bar.get_width()
    ax_d.annotate(f'{w:.1f}% ({int(round(w * n_total / 100))}/{n_total})',
                  xy=(w, bar.get_y() + bar.get_height()/2), xytext=(5, 0),
                  textcoords="offset points", ha='left', va='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(target_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Successfully generated: {target_path} ({target_path.stat().st_size} bytes)")
