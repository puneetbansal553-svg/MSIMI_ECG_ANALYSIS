import json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, auc

project_root = Path.cwd()
fig_out_dir = project_root / 'outputs' / 'figures'
fig_out_dir.mkdir(parents=True, exist_ok=True)

# Ingest read-only frozen data artifacts
df_feat = pd.read_csv(project_root / 'data' / 'features' / '08_patient_phase_features.csv')
df_splits = pd.read_csv(project_root / 'reports' / 'validation' / '09_patient_validation_splits.csv')
df_m10 = pd.read_csv(project_root / 'reports' / 'models' / '10_model_evaluation_metrics.csv')
df_p10 = pd.read_csv(project_root / 'reports' / 'models' / '10_patient_predictions.csv')
df_manifest = pd.read_csv(project_root / 'data' / 'manifests' / 'cohort_manifest.csv')

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

# =========================================================================
# Figure 1: 12_stage08_st_and_autonomic_distributions.png
# =========================================================================
fig1_path = fig_out_dir / '12_stage08_st_and_autonomic_distributions.png'
fig, axes = plt.subplots(2, 2, figsize=(16, 11), dpi=300)

df_feat_merged = pd.merge(df_feat, df_manifest[['subject_id', 'label_status']], on='subject_id')
df_feat_labeled = df_feat_merged[df_feat_merged['label_status'].isin(['POSITIVE', 'NEGATIVE'])].copy()

# Panel A: Minimum ST60 Global (ADC counts) by Phase and Group (at 500 Hz)
ax_a = axes[0, 0]
sub_500 = df_feat_labeled[df_feat_labeled['sampling_rate'] == 500]
rest_st_neg = sub_500[sub_500['label_status'] == 'NEGATIVE']['rest_min_st60_global_adc'].dropna()
rest_st_pos = sub_500[sub_500['label_status'] == 'POSITIVE']['rest_min_st60_global_adc'].dropna()
stress_st_neg = sub_500[sub_500['label_status'] == 'NEGATIVE']['stress_min_st60_global_adc'].dropna() if 'stress_min_st60_global_adc' in sub_500.columns else rest_st_neg
stress_st_pos = sub_500[sub_500['label_status'] == 'POSITIVE']['stress_min_st60_global_adc'].dropna() if 'stress_min_st60_global_adc' in sub_500.columns else rest_st_pos

bp_a = ax_a.boxplot([rest_st_neg, rest_st_pos], tick_labels=['MSIMI-Negative\n(Rest)', 'MSIMI-Positive\n(Rest)'],
                    patch_artist=True, medianprops=dict(color='black', linewidth=1.5))
bp_a['boxes'][0].set_facecolor('#abd9e9')
bp_a['boxes'][1].set_facecolor('#fdae61')
ax_a.set_ylabel('Global Min ST60 Level (Raw ADC Counts)', fontsize=11, fontweight='bold')
ax_a.set_title('A: Global ST60 Depression Level (500 Hz Cohort)', fontsize=12, fontweight='bold', pad=10)

# Panel B: Autonomic HRV Feature Distributions (Rest RMSSD)
ax_b = axes[0, 1]
rmssd_col = [c for c in sub_500.columns if 'rmssd' in c.lower()][0]
rmssd_neg = sub_500[sub_500['label_status'] == 'NEGATIVE'][rmssd_col].dropna()
rmssd_pos = sub_500[sub_500['label_status'] == 'POSITIVE'][rmssd_col].dropna()

bp_b = ax_b.boxplot([rmssd_neg, rmssd_pos], tick_labels=['MSIMI-Negative\n(N=48)', 'MSIMI-Positive\n(N=14)'],
                    patch_artist=True, medianprops=dict(color='black', linewidth=1.5))
bp_b['boxes'][0].set_facecolor('#abd9e9')
bp_b['boxes'][1].set_facecolor('#fdae61')
ax_b.set_ylabel(f'{rmssd_col} (ms)', fontsize=11, fontweight='bold')
ax_b.set_title('B: Baseline Parasympathetic Tone (Rest RMSSD)', fontsize=12, fontweight='bold', pad=10)

# Panel C: Feature Correlation Scatter: ST Depression vs Autonomic Tone
ax_c = axes[1, 0]
for grp, color, marker, label in [('NEGATIVE', '#1f77b4', 'o', 'MSIMI-Negative'),
                                  ('POSITIVE', '#d62728', 's', 'MSIMI-Positive')]:
    sub = sub_500[sub_500['label_status'] == grp]
    ax_c.scatter(sub['rest_min_st60_global_adc'], sub[rmssd_col],
                 color=color, marker=marker, s=50, alpha=0.75, label=label, edgecolor='black', linewidth=0.5)
ax_c.set_xlabel('Global Min ST60 Level (ADC counts)', fontsize=11, fontweight='bold')
ax_c.set_ylabel(f'{rmssd_col} (ms)', fontsize=11, fontweight='bold')
ax_c.set_title('C: ST-Depression vs Vagal Modulation Covariance', fontsize=12, fontweight='bold', pad=10)
ax_c.legend(loc='upper right', frameon=True)

# Panel D: Sampling Frequency Feature Invariance (100 Hz vs 500 Hz)
ax_d = axes[1, 1]
feat_100 = df_feat_labeled[df_feat_labeled['sampling_rate'] == 100].set_index('subject_id')
feat_500 = df_feat_labeled[df_feat_labeled['sampling_rate'] == 500].set_index('subject_id')
common_subs = feat_100.index.intersection(feat_500.index)
x_st = feat_100.loc[common_subs, 'rest_min_st60_global_adc']
y_st = feat_500.loc[common_subs, 'rest_min_st60_global_adc']

ax_d.scatter(x_st, y_st, color='#2ca02c', alpha=0.75, s=40, edgecolor='black')
min_v = min(x_st.min(), y_st.min()) - 10
max_v = max(x_st.max(), y_st.max()) + 10
ax_d.plot([min_v, max_v], [min_v, max_v], 'k--', alpha=0.6, label='1:1 Line')
ax_d.set_xlabel('100 Hz Rest Min ST60 (ADC)', fontsize=11, fontweight='bold')
ax_d.set_ylabel('500 Hz Rest Min ST60 (ADC)', fontsize=11, fontweight='bold')
ax_d.set_title('D: Feature Sampling Invariance (100 Hz vs 500 Hz)', fontsize=12, fontweight='bold', pad=10)
ax_d.legend(loc='lower right', frameon=True)

plt.tight_layout()
plt.savefig(fig1_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Generated: {fig1_path}")

# =========================================================================
# Figure 2: 12_stage08_feature_correlation_and_missingness.png
# =========================================================================
fig2_path = fig_out_dir / '12_stage08_feature_correlation_and_missingness.png'
fig, axes = plt.subplots(1, 2, figsize=(16, 7), dpi=300)

# Panel A: Feature Correlation Matrix (Subset of representative features)
ax_a = axes[0]
sample_cols = [c for c in sub_500.columns if any(k in c for k in ['st60', 'qrs', 'rr_mean', 'rmssd', 't_amp'])][:10]
corr_mat = sub_500[sample_cols].corr()
im = ax_a.imshow(corr_mat, cmap='coolwarm', vmin=-1, vmax=1)
ax_a.set_xticks(range(len(sample_cols)))
ax_a.set_yticks(range(len(sample_cols)))
ax_a.set_xticklabels([c.replace('rest_', '') for c in sample_cols], rotation=45, ha='right', fontsize=9)
ax_a.set_yticklabels([c.replace('rest_', '') for c in sample_cols], fontsize=9)
plt.colorbar(im, ax=ax_a, fraction=0.046, pad=0.04)
ax_a.set_title('A: Cross-Feature Correlation Matrix (Stage 08)', fontsize=12, fontweight='bold', pad=10)

# Panel B: Feature Missingness Audit Across Strata
ax_b = axes[1]
strata_names = ['All Subjects 100 Hz', 'Matched Both 100 Hz', 'Matched Both 500 Hz']
sub_100_all = df_feat[df_feat['sampling_rate'] == 100]
sub_100_mat = df_feat[(df_feat['sampling_rate'] == 100) & (df_feat['subject_id'].isin(common_subs))]
sub_500_mat = df_feat[(df_feat['sampling_rate'] == 500) & (df_feat['subject_id'].isin(common_subs))]

numeric_cols = df_feat.select_dtypes(include=[np.number]).columns
pct_missing = [
    sub_100_all[numeric_cols].isna().mean().mean() * 100,
    sub_100_mat[numeric_cols].isna().mean().mean() * 100,
    sub_500_mat[numeric_cols].isna().mean().mean() * 100
]
bars_b = ax_b.bar(strata_names, pct_missing, color=['#4575b4', '#74add1', '#fdae61'], edgecolor='black', width=0.45)
ax_b.set_ylabel('Mean Feature Missingness (%)', fontsize=11, fontweight='bold')
ax_b.set_title('B: Pre-Imputation Feature Store Completeness', fontsize=12, fontweight='bold', pad=10)
ax_b.set_ylim(0, 15)
for bar in bars_b:
    h = bar.get_height()
    ax_b.annotate(f'{h:.2f}%', xy=(bar.get_x() + bar.get_width()/2, h), xytext=(0, 3),
                  textcoords="offset points", ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(fig2_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Generated: {fig2_path}")

# =========================================================================
# Figure 3: 12_stage09_patient_fold_assignment_heatmap.png
# =========================================================================
fig3_path = fig_out_dir / '12_stage09_patient_fold_assignment_heatmap.png'
fig, ax = plt.subplots(figsize=(16, 8), dpi=300)

# Build fold assignment matrix: Patients (rows) x 5 Repeats (columns)
repeat_cols = [f'repeat_{r}_fold' for r in range(5)]
df_splits_sorted = df_splits.sort_values('subject_id')
fold_mat = df_splits_sorted[repeat_cols].values  # shape: (80, 5)

# Mask holdouts (-1)
cmap = plt.colormaps['Set3']
im_split = ax.imshow(fold_mat.T, aspect='auto', cmap=cmap, interpolation='none')

ax.set_yticks(range(5))
ax.set_yticklabels([f'Repeat {r}' for r in range(5)], fontsize=11, fontweight='bold')
ax.set_xlabel('Patient Cohort (sub_001 to sub_080)', fontsize=11, fontweight='bold')
ax.set_title('Stage 09: Frozen 5-Repeat x 5-Fold Patient-Level Partition Heatmap (N=80)', fontsize=13, fontweight='bold', pad=12)

# Highlight holdouts sub_005 and sub_049
holdout_indices = df_splits_sorted[df_splits_sorted['subject_id'].isin(['sub_005', 'sub_049'])].index
for idx in holdout_indices:
    ax.axvline(idx, color='red', linestyle='--', linewidth=1.5, alpha=0.8)

plt.colorbar(im_split, ax=ax, fraction=0.02, pad=0.03, ticks=range(5), label='Fold Assignment (0 - 4)')
plt.tight_layout()
plt.savefig(fig3_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Generated: {fig3_path}")

# =========================================================================
# Figure 4: 12_stage09_fold_class_distribution_and_holdouts.png
# =========================================================================
fig4_path = fig_out_dir / '12_stage09_fold_class_distribution_and_holdouts.png'
fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=300)

# Panel A: Positive Patient Allocation across 25 Folds
ax_a = axes[0]
pos_counts = []
fold_labels = []
for r in range(5):
    for f in range(5):
        fold_subs = df_splits[df_splits[f'repeat_{r}_fold'] == f]
        n_pos_fold = (fold_subs['target_label'] == 1.0).sum()
        pos_counts.append(n_pos_fold)
        fold_labels.append(f'R{r}F{f}')

ax_a.bar(range(25), pos_counts, color='#2b5c8f', edgecolor='black', width=0.7)
ax_a.axhline(16/5, color='red', linestyle='--', linewidth=1.2, label='Expected Uniform Allocation (3.2 / fold)')
ax_a.set_ylabel('MSIMI-Positive Patients Count', fontsize=11, fontweight='bold')
ax_a.set_xlabel('Fold Partition Index (25 Total)', fontsize=11, fontweight='bold')
ax_a.set_title('A: MSIMI-Positive Class Balance Across 25 Folds', fontsize=12, fontweight='bold', pad=10)
ax_a.set_ylim(0, 6)
ax_a.legend(loc='upper right', frameon=True)

# Panel B: Holdout Isolation Profile
ax_b = axes[1]
categories_holdout = ['Validation Eligible\n(N=78 Patients)', 'Inference Holdout\n(sub_005, sub_049)']
counts_holdout = [78, 2]
bars_h = ax_b.bar(categories_holdout, counts_holdout, color=['#31a354', '#de2d26'], edgecolor='black', width=0.45)
ax_b.set_ylabel('Number of Patients', fontsize=11, fontweight='bold')
ax_b.set_title('B: Patient Validation Eligibility & Strict Holdout Isolation', fontsize=12, fontweight='bold', pad=10)
ax_b.set_ylim(0, 90)
for bar in bars_h:
    h = bar.get_height()
    ax_b.annotate(f'N = {h}\n({h/80*100:.1f}%)', xy=(bar.get_x() + bar.get_width()/2, h), xytext=(0, 4),
                  textcoords="offset points", ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(fig4_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Generated: {fig4_path}")

# =========================================================================
# Figure 5: 12_stage10_roc_and_prc_comparison_curves.png
# =========================================================================
fig5_path = fig_out_dir / '12_stage10_roc_and_prc_comparison_curves.png'
fig, axes = plt.subplots(1, 2, figsize=(16, 6.5), dpi=300)

ax_roc = axes[0]
ax_prc = axes[1]

# Plot ROC & PRC for Stage 10 Elastic Net across Feature Subsets (500 Hz matched_both)
palette_subsets = {
    'Subset_A_ST_Dynamics': '#1f77b4',
    'Subset_B_Autonomic': '#2ca02c',
    'Subset_C_Multimodal': '#d62728'
}
labels_subsets = {
    'Subset_A_ST_Dynamics': 'Subset A: ST Dynamics',
    'Subset_B_Autonomic': 'Subset B: Autonomic HRV',
    'Subset_C_Multimodal': 'Subset C: Multimodal Combined'
}

df_p10_matched = df_p10[(df_p10['cohort'] == 'matched_62') & (df_p10['sampling_rate'] == 500) & (df_p10['model_type'] == 'ridge') & (df_p10['is_holdout'] == False)]

ax_roc.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Chance (AUC=0.50)')
prevalence_10 = 7.0 / 62.0
ax_prc.axhline(prevalence_10, color='k', linestyle='--', alpha=0.5, label=f'Prevalence ({prevalence_10:.3f})')

for s_name, color in palette_subsets.items():
    sub_p = df_p10_matched[df_p10_matched['feature_subset'] == s_name]
    if len(sub_p) > 0:
        # Aggregate mean score per subject across repeats
        sub_agg = sub_p.groupby('subject_id').agg({'target_label': 'first', 'predicted_score': 'mean'}).dropna()
        fpr, tpr, _ = roc_curve(sub_agg['target_label'], sub_agg['predicted_score'])
        prec, rec, _ = precision_recall_curve(sub_agg['target_label'], sub_agg['predicted_score'])
        roc_val = auc(fpr, tpr)
        prc_val = auc(rec, prec)
        
        ax_roc.plot(fpr, tpr, color=color, linewidth=2.0, label=f"{labels_subsets[s_name]} (AUC={roc_val:.3f})")
        ax_prc.plot(rec, prec, color=color, linewidth=2.0, label=f"{labels_subsets[s_name]} (AP={prc_val:.3f})")

ax_roc.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=11, fontweight='bold')
ax_roc.set_ylabel('True Positive Rate (Sensitivity)', fontsize=11, fontweight='bold')
ax_roc.set_title('A: ROC Discrimination Curves (Stage 10 Elastic Net)', fontsize=12, fontweight='bold', pad=10)
ax_roc.legend(loc='lower right', frameon=True)

ax_prc.set_xlabel('Recall (Sensitivity)', fontsize=11, fontweight='bold')
ax_prc.set_ylabel('Precision (PPV)', fontsize=11, fontweight='bold')
ax_prc.set_title('B: Precision-Recall Curves (Stage 10 Elastic Net)', fontsize=12, fontweight='bold', pad=10)
ax_prc.legend(loc='upper right', frameon=True)

plt.tight_layout()
plt.savefig(fig5_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Generated: {fig5_path}")

# =========================================================================
# Figure 6: 12_stage10_model_comparison_and_confusion_matrices.png
# =========================================================================
fig6_path = fig_out_dir / '12_stage10_model_comparison_and_confusion_matrices.png'
fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=300)

# Panel A: Balanced Accuracy and Sensitivity/Specificity at Prevalence Threshold
ax_a = axes[0]
sub_m10_matched = df_m10[(df_m10['cohort'] == 'matched_62') & (df_m10['sampling_rate'] == 500) & (df_m10['model_type'] == 'ridge')].set_index('feature_subset')
subsets_order = ['Subset_A_ST_Dynamics', 'Subset_B_Autonomic', 'Subset_C_Multimodal']
labels_order = ['Subset A (ST)', 'Subset B (HRV)', 'Subset C (Multi)']

sens = sub_m10_matched.loc[subsets_order, 'sens_at_prev_mean'].values
spec = sub_m10_matched.loc[subsets_order, 'spec_at_prev_mean'].values
bacc = sub_m10_matched.loc[subsets_order, 'bacc_at_prev_mean'].values

x_m = np.arange(3)
w_m = 0.25
ax_a.bar(x_m - w_m, sens, w_m, label='Sensitivity', color='#e41a1c', edgecolor='black')
ax_a.bar(x_m, spec, w_m, label='Specificity', color='#377eb8', edgecolor='black')
ax_a.bar(x_m + w_m, bacc, w_m, label='Balanced Accuracy', color='#4daf4a', edgecolor='black')

ax_a.axhline(0.50, color='gray', linestyle='--', linewidth=1.2, label='Chance Baseline')
ax_a.set_xticks(x_m)
ax_a.set_xticklabels(labels_order, fontsize=11, fontweight='bold')
ax_a.set_ylabel('Validation Performance Metric', fontsize=11, fontweight='bold')
ax_a.set_title('A: Threshold-Dependent Performance at Prevalence Operating Point', fontsize=12, fontweight='bold', pad=10)
ax_a.set_ylim(0, 1.05)
ax_a.legend(loc='upper right', frameon=True)

# Panel B: Summary Metrics Table / Heatmap across Feature Subsets
ax_b = axes[1]
metrics_table = sub_m10_matched.loc[subsets_order, ['roc_auc_mean', 'average_precision_mean', 'bacc_at_prev_mean', 'f1_at_prev_mean']]
metrics_table.index = labels_order
metrics_table.columns = ['ROC-AUC', 'Average Precision', 'Balanced Accuracy', 'F1 Score']

im_t = ax_b.imshow(metrics_table.values, cmap='YlGnBu', vmin=0.1, vmax=0.8)
ax_b.set_xticks(range(len(metrics_table.columns)))
ax_b.set_yticks(range(len(labels_order)))
ax_b.set_xticklabels(metrics_table.columns, fontsize=10, fontweight='bold')
ax_b.set_yticklabels(labels_order, fontsize=10, fontweight='bold')
for i in range(len(labels_order)):
    for j in range(len(metrics_table.columns)):
        val = metrics_table.values[i, j]
        ax_b.annotate(f'{val:.3f}', xy=(j, i), ha='center', va='center', fontsize=11, fontweight='bold',
                      color='white' if val > 0.45 else 'black')
ax_b.set_title('B: Cross-Validation Summary Matrix (500 Hz Matched Both)', fontsize=12, fontweight='bold', pad=10)
plt.colorbar(im_t, ax=ax_b, fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig(fig6_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Generated: {fig6_path}")
print("ALL 6 FIGURES IN GALLERY TEST GENERATED SUCCESSFULLY!")
