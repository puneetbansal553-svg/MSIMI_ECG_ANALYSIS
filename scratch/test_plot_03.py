import json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

project_root = Path.cwd()
df_rec_qc = pd.read_csv(project_root / 'reports' / 'qc' / '03_ecg_recording_quality_audit.csv')
df_lead_qc = pd.read_csv(project_root / 'reports' / 'qc' / '03_ecg_lead_level_quality_audit.csv')

fig_out_dir = project_root / 'outputs' / 'figures'
fig_out_dir.mkdir(parents=True, exist_ok=True)
target_path = fig_out_dir / '03_lead_quality_and_signal_stability_matrix.png'

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=300)

leads_order = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

# Panel A: 12-Lead Amplitude Stability (Mean +/- Std in ADC units)
ax_a = axes[0, 0]
mean_amps = []
std_amps = []
for lead in leads_order:
    sub = df_lead_qc[df_lead_qc['lead_name'] == lead]
    mean_amps.append(sub['mean_adc'].mean())
    std_amps.append(sub['std_adc'].mean())

x_leads = np.arange(len(leads_order))
ax_a.bar(x_leads, mean_amps, yerr=std_amps, capsize=4, color='#2b5c8f', edgecolor='black', alpha=0.85, label='Mean ADC +/- Std ADC')
ax_a.axhline(2048, color='red', linestyle='--', linewidth=1.2, label='Nominal Baseline (ADC=2048)')
ax_a.set_xticks(x_leads)
ax_a.set_xticklabels(leads_order, fontsize=10, fontweight='bold')
ax_a.set_ylabel('Signal Level (Raw ADC Counts)', fontsize=11, fontweight='bold')
ax_a.set_title('A: 12-Lead Baseline Centering & Dynamic Spread', fontsize=12, fontweight='bold', pad=10)
ax_a.legend(loc='lower left', frameon=True)

# Panel B: Lead-Level Clipping Frequency (High & Low Clipping in ADC)
ax_b = axes[0, 1]
clip_low_counts = []
clip_high_counts = []
for lead in leads_order:
    sub = df_lead_qc[df_lead_qc['lead_name'] == lead]
    clip_low_counts.append(sub['clip_low_count'].sum())
    clip_high_counts.append(sub['clip_high_count'].sum())

w = 0.35
ax_b.bar(x_leads - w/2, clip_low_counts, w, label='Low Saturation (<= 5 ADC)', color='#3182bd', edgecolor='black')
ax_b.bar(x_leads + w/2, clip_high_counts, w, label='High Saturation (>= 4090 ADC)', color='#de2d26', edgecolor='black')
ax_b.set_xticks(x_leads)
ax_b.set_xticklabels(leads_order, fontsize=10, fontweight='bold')
ax_b.set_ylabel('Total Saturated Samples Across Cohort', fontsize=11, fontweight='bold')
ax_b.set_title('B: Lead-Specific ADC Satiation / Clipping Frequency', fontsize=12, fontweight='bold', pad=10)
ax_b.legend(loc='upper right', frameon=True)

# Panel C: Sampling Frequency & Unequal Lead Length Incidence (V6 Deficit)
ax_c = axes[1, 0]
# Count unequal lead occurrences by sampling rate and phase
unequal_cross = pd.crosstab(
    df_rec_qc['phase'],
    df_rec_qc['sampling_rate'],
    values=df_rec_qc['has_unequal_lead_lengths'].fillna(False).astype(int),
    aggfunc='sum'
).reindex(['rest', 'stress', 'recover'])

phases = ['Rest', 'Stress', 'Recover']
x_c = np.arange(3)
rects1 = ax_c.bar(x_c - w/2, unequal_cross[100].values, w, label='100 Hz (N=80/phase)', color='#4575b4', edgecolor='black')
rects2 = ax_c.bar(x_c + w/2, unequal_cross[500].values, w, label='500 Hz (N=80/phase)', color='#f46d43', edgecolor='black')
ax_c.set_xticks(x_c)
ax_c.set_xticklabels(phases, fontsize=10, fontweight='bold')
ax_c.set_ylabel('Recordings with Shortened Lead V6', fontsize=11, fontweight='bold')
ax_c.set_title('C: Unequal Lead Length (Shortened Lead V6) Incidence', fontsize=12, fontweight='bold', pad=10)
ax_c.legend(loc='upper left', frameon=True)
for bar in rects1 + rects2:
    h = bar.get_height()
    if h > 0:
        ax_c.annotate(f'{int(h)} ({h/80*100:.1f}%)',
                      xy=(bar.get_x() + bar.get_width()/2, h), xytext=(0, 3),
                      textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')

# Panel D: Recording Quality Summary & Integrity Gate Status
ax_d = axes[1, 1]
qc_categories = [
    'Finite Numerical Values\n(0 NaN / 0 Inf)',
    'Zero Flatline Leads\n(0 Flatline Channels)',
    'Lead Integrity Compliance\n(12 Leads Present)',
    'Clipping Safety\n(< 0.05% of Samples)',
    'Synchronous Lengths\n(82.3% Perfect Sync)'
]
n_total_recs = len(df_rec_qc)
n_finite = (df_rec_qc['all_finite'] == True).sum()
n_noflat = (df_rec_qc['flatline_lead_count'] == 0).sum()
n_12lead = (df_rec_qc['lead_count_valid'] == True).sum()
n_clip_safe = (df_rec_qc['clipping_pct_overall'] < 0.05).sum()
n_sync = (df_rec_qc['has_unequal_lead_lengths'] == False).sum()

qc_pcts = [
    n_finite / n_total_recs * 100,
    n_noflat / n_total_recs * 100,
    n_12lead / n_total_recs * 100,
    n_clip_safe / n_total_recs * 100,
    n_sync / n_total_recs * 100
]
bar_colors_d = ['#2ca02c', '#2ca02c', '#2ca02c', '#1f77b4', '#ff7f0e']
bars_d = ax_d.barh(qc_categories[::-1], qc_pcts[::-1], color=bar_colors_d[::-1], edgecolor='black', linewidth=0.8)
ax_d.set_xlabel('Cohort Compliance (%)', fontsize=11, fontweight='bold')
ax_d.set_title('D: Quality Control Acceptance Gate Metrics (N=480)', fontsize=12, fontweight='bold', pad=10)
ax_d.set_xlim(0, 115)
for bar in bars_d:
    w_val = bar.get_width()
    ax_d.annotate(f'{w_val:.1f}% ({int(round(w_val * n_total_recs / 100))}/{n_total_recs})',
                  xy=(w_val, bar.get_y() + bar.get_height()/2), xytext=(5, 0),
                  textcoords="offset points", ha='left', va='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(target_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Successfully generated: {target_path} ({target_path.stat().st_size} bytes)")
