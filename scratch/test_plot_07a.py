import json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

project_root = Path.cwd()
manifest_path = project_root / 'reports' / 'transitions' / '07a_inter_phase_transition_manifest.csv'
df_transitions = pd.read_csv(manifest_path)

fig_out_dir = project_root / 'outputs' / 'figures'
fig_out_dir.mkdir(parents=True, exist_ok=True)
target_path = fig_out_dir / '07a_hrr60_hrri_and_morphology_distributions.png'

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, axes = plt.subplots(2, 2, figsize=(16, 11), dpi=300)

# Filter for T2 (Stress -> Recover) for HRR60 and HRRI
df_t2 = df_transitions[df_transitions['transition_type'] == 'stress_to_recover'].copy()
df_t1 = df_transitions[df_transitions['transition_type'] == 'rest_to_stress'].copy()

# Panel A: HRR60 Distribution by Sampling Rate
ax_a = axes[0, 0]
hrr60_100 = df_t2[df_t2['sampling_rate'] == 100]['hrr60_bpm'].dropna()
hrr60_500 = df_t2[df_t2['sampling_rate'] == 500]['hrr60_bpm'].dropna()

bp_a = ax_a.boxplot([hrr60_100, hrr60_500], tick_labels=['100 Hz (N=80)', '500 Hz (N=64)'],
                    patch_artist=True, medianprops=dict(color='black', linewidth=1.5))
bp_a['boxes'][0].set_facecolor('#4575b4')
bp_a['boxes'][1].set_facecolor('#f46d43')
ax_a.axhline(12.0, color='red', linestyle='--', linewidth=1.2, label='Clinical Blunting Cutoff (12 BPM)')
ax_a.scatter(np.random.normal(1, 0.04, size=len(hrr60_100)), hrr60_100, color='#2b5c8f', alpha=0.5, s=25)
ax_a.scatter(np.random.normal(2, 0.04, size=len(hrr60_500)), hrr60_500, color='#d73027', alpha=0.5, s=25)
ax_a.set_ylabel('HRR_60 (HR_end_stress - HR_rec_60s) [BPM]', fontsize=11, fontweight='bold')
ax_a.set_title('A: 1-Minute Heart Rate Recovery (HRR_60) Distribution', fontsize=12, fontweight='bold', pad=10)
ax_a.legend(loc='upper right', frameon=True)

# Panel B: HRRI (%) vs HRR60 (BPM)
ax_b = axes[0, 1]
valid_hrri = df_t2.dropna(subset=['hrr60_bpm', 'hrri_pct'])
ax_b.scatter(valid_hrri['hrr60_bpm'], valid_hrri['hrri_pct'], color='#756bb1', alpha=0.7, s=35, edgecolor='black', linewidth=0.5)
ax_b.axvline(12.0, color='gray', linestyle=':', alpha=0.7)
ax_b.axhline(0.0, color='gray', linestyle=':', alpha=0.7)
ax_b.set_xlabel('HRR_60 (BPM)', fontsize=11, fontweight='bold')
ax_b.set_ylabel('Heart Rate Recovery Index (HRRI %)', fontsize=11, fontweight='bold')
ax_b.set_title('B: Recovery Index (HRRI) vs Absolute HR Drop (HRR_60)', fontsize=12, fontweight='bold', pad=10)

# Panel C: Frobenius Morphology Dissimilarity (Rest->Stress vs Stress->Recover)
ax_c = axes[1, 0]
frob_t1 = df_t1['frobenius_dissimilarity_mv'].dropna()
frob_t2 = df_t2['frobenius_dissimilarity_mv'].dropna()

bp_c = ax_c.boxplot([frob_t1, frob_t2], tick_labels=['Rest -> Stress (T1)', 'Stress -> Recover (T2)'],
                    patch_artist=True, medianprops=dict(color='black', linewidth=1.5))
bp_c['boxes'][0].set_facecolor('#fdae61')
bp_c['boxes'][1].set_facecolor('#a6d96a')
ax_c.scatter(np.random.normal(1, 0.04, size=len(frob_t1)), frob_t1, color='#e6550d', alpha=0.4, s=20)
ax_c.scatter(np.random.normal(2, 0.04, size=len(frob_t2)), frob_t2, color='#31a354', alpha=0.4, s=20)
ax_c.set_ylabel('12-Lead Frobenius Dissimilarity (ADC Counts)', fontsize=11, fontweight='bold')
ax_c.set_title('C: Global 12-Lead Median Beat Morphology Distortion', fontsize=12, fontweight='bold', pad=10)

# Panel D: Lead V5 ST60 Shift Across Transitions
ax_d = axes[1, 1]
st60_t1 = df_t1['delta_st60_lead_v5_mv'].dropna()
st60_t2 = df_t2['delta_st60_lead_v5_mv'].dropna()

bp_d = ax_d.boxplot([st60_t1, st60_t2], tick_labels=['Rest -> Stress (T1)', 'Stress -> Recover (T2)'],
                    patch_artist=True, medianprops=dict(color='black', linewidth=1.5))
bp_d['boxes'][0].set_facecolor('#f781bf')
bp_d['boxes'][1].set_facecolor('#999999')
ax_d.axhline(0, color='black', linestyle='--', linewidth=1.0)
ax_d.set_ylabel('Delta ST60 in Lead V5 (Raw ADC Counts)', fontsize=11, fontweight='bold')
ax_d.set_title('D: Precordial Lead V5 Repolarization Shift (ST60 Delta)', fontsize=12, fontweight='bold', pad=10)

plt.tight_layout()
plt.savefig(target_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Successfully generated: {target_path} ({target_path.stat().st_size} bytes)")

