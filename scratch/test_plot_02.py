import json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

project_root = Path.cwd()
manifest_path = project_root / 'data' / 'manifests' / 'cohort_manifest.csv'
df_candidate_manifest = pd.read_csv(manifest_path)

fig_out_dir = project_root / 'outputs' / 'figures'
fig_out_dir.mkdir(parents=True, exist_ok=True)
target_path = fig_out_dir / '02_cohort_eligibility_and_completeness_waterfall.png'

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=300)

# Panel A: Cohort Verification & Eligibility Flow
ax_a = axes[0]
stages = [
    'Enrolled Cohort\n(sub_001 - sub_080)',
    'Confirmed ANOCA\n(100% Match)',
    'Limb & Metadata Files\n(14/14 Present)',
    'Definitive Ground Truth\n(78 Labeled, 2 Unspec)'
]
values = [80, 80, 80, 78]
bar_colors = ['#2b5c8f', '#41b6c4', '#7fcdbb', '#2ca02c']
bars_a = ax_a.bar(stages, values, color=bar_colors, edgecolor='black', width=0.55)
ax_a.set_ylabel('Number of Patients (N)', fontsize=11, fontweight='bold')
ax_a.set_title('A: Cohort Eligibility & Verification Funnel', fontsize=12, fontweight='bold', pad=10)
ax_a.set_ylim(0, 95)
for bar in bars_a:
    h = bar.get_height()
    ax_a.annotate(f'N = {h}\n(100%)' if h == 80 else f'N = {h}\n(97.5%)',
                  xy=(bar.get_x() + bar.get_width()/2, h), xytext=(0, 4),
                  textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')

# Panel B: Ground Truth Target Class Balance (MSIMI vs Non-MSIMI)
ax_b = axes[1]
n_total = len(df_candidate_manifest)
n_pos = int((df_candidate_manifest['label_status'] == 'POSITIVE').sum())
n_neg = int((df_candidate_manifest['label_status'] == 'NEGATIVE').sum())
n_unspec = int((df_candidate_manifest['label_status'] == 'UNSPECIFIED').sum())

labels_pie = [f'MSIMI-Negative\n(N={n_neg}, {n_neg/n_total*100:.1f}%)',
              f'MSIMI-Positive\n(N={n_pos}, {n_pos/n_total*100:.1f}%)',
              f'Unspecified Holdout\n(N={n_unspec}, {n_unspec/n_total*100:.1f}%)']
sizes = [n_neg, n_pos, n_unspec]
colors_pie = ['#3182bd', '#de2d26', '#bdbdbd']
explode = (0.03, 0.08, 0.05)

wedges, texts, autotexts = ax_b.pie(
    sizes, explode=explode, labels=labels_pie, colors=colors_pie,
    autopct='%1.1f%%', startangle=140, pctdistance=0.6,
    wedgeprops=dict(edgecolor='black', linewidth=1.2),
    textprops=dict(fontsize=10, fontweight='bold')
)
ax_b.set_title('B: Ground-Truth Target Label Distribution\n(Imbalance Ratio ~ 1 : 3.88)', fontsize=12, fontweight='bold', pad=10)

# Panel C: Stratified Modality Completeness across Positive, Negative, Unspecified
ax_c = axes[2]
categories = ['MSIMI-\n(N=62)', 'MSIMI+\n(N=16)', 'Holdout\n(N=2)']
# In our cohort: All 80 have 100 Hz recordings (62 neg, 16 pos, 2 unspec)
# For 500 Hz matched: 62 subjects total have 500 Hz (48 neg, 14 pos, 0 unspec or similar)
# Let's compute exact matched counts from cohort manifest and files
rates = [100, 500]
x_c = np.arange(3)
w_c = 0.35
# Count by group
def get_group_counts(rate):
    c_neg = len([s for s in df_candidate_manifest[df_candidate_manifest['label_status'] == 'NEGATIVE']['subject_id']
                 if (project_root / 'data' / 'raw' / f"{s}_baseline-data_rest-ecg-{rate}hz.csv").exists()])
    c_pos = len([s for s in df_candidate_manifest[df_candidate_manifest['label_status'] == 'POSITIVE']['subject_id']
                 if (project_root / 'data' / 'raw' / f"{s}_baseline-data_rest-ecg-{rate}hz.csv").exists()])
    c_uns = len([s for s in df_candidate_manifest[df_candidate_manifest['label_status'] == 'UNSPECIFIED']['subject_id']
                 if (project_root / 'data' / 'raw' / f"{s}_baseline-data_rest-ecg-{rate}hz.csv").exists()])
    return [c_neg, c_pos, c_uns]

counts_100hz = get_group_counts(100)
counts_500hz = get_group_counts(500)

b1 = ax_c.bar(x_c - w_c/2, counts_100hz, w_c, label='100 Hz Acquisition', color='#4575b4', edgecolor='black')
b2 = ax_c.bar(x_c + w_c/2, counts_500hz, w_c, label='500 Hz Acquisition', color='#f46d43', edgecolor='black')
ax_c.set_ylabel('Number of Available Patients', fontsize=11, fontweight='bold')
ax_c.set_title('C: Modality Availability by Clinical Subgroup', fontsize=12, fontweight='bold', pad=10)
ax_c.set_xticks(x_c)
ax_c.set_xticklabels(categories, fontsize=10, fontweight='bold')
ax_c.set_ylim(0, 75)
ax_c.legend(loc='upper right', frameon=True)
for bar in b1 + b2:
    h = bar.get_height()
    ax_c.annotate(f'{h}', xy=(bar.get_x() + bar.get_width()/2, h), xytext=(0, 3),
                  textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(target_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Successfully generated: {target_path} ({target_path.stat().st_size} bytes)")

