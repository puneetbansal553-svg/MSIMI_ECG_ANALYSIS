import json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

project_root = Path.cwd()
metrics_csv_path = project_root / 'reports' / 'models' / '11_extended_model_evaluation_metrics.csv'
df_metrics = pd.read_csv(metrics_csv_path)

fig_out_dir = project_root / 'outputs' / 'figures'
fig_out_dir.mkdir(parents=True, exist_ok=True)
target_path_1 = fig_out_dir / '11_threshold_metrics_and_accuracy_baselines.png'
target_path_2 = fig_out_dir / '11_knn_sensitivity_and_cohort_strata_comparison.png'

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

# Figure 1: 11_threshold_metrics_and_accuracy_baselines.png
fig, axes = plt.subplots(2, 2, figsize=(16, 11), dpi=300)

# Filter for primary matched-both cohort
df_prim = df_metrics[(df_metrics['cohort'] == 'matched_both') & (df_metrics['sampling_rate'] == 500)].copy()

model_order = ['rf', 'xgb', 'svm_linear', 'svm_rbf', 'knn_k5']
model_labels = ['Random Forest', 'XGBoost', 'Linear SVM', 'RBF SVM', 'KNN (k=5)']
subsets = ['Subset_A_ST_Dynamics', 'Subset_B_Autonomic', 'Subset_C_Multimodal']
subset_labels = ['Subset A (ST)', 'Subset B (HRV)', 'Subset C (Multi)']

# Panel A: ROC-AUC Mean Across Models & Subsets
ax_a = axes[0, 0]
x = np.arange(len(model_order))
w = 0.25
colors = ['#1f77b4', '#2ca02c', '#d62728']

for s_idx, (s_name, s_lbl) in enumerate(zip(subsets, subset_labels)):
    sub = df_prim[df_prim['feature_subset'] == s_name].set_index('model_type').reindex(model_order)
    vals = sub['roc_auc_mean'].values
    errs = sub['roc_auc_sd'].values
    ax_a.bar(x + (s_idx - 1) * w, vals, w, yerr=errs, capsize=3, label=s_lbl, color=colors[s_idx], edgecolor='black')

ax_a.axhline(0.50, color='gray', linestyle='--', linewidth=1.2, label='Chance Baseline (0.50)')
ax_a.set_xticks(x)
ax_a.set_xticklabels(model_labels, fontsize=10, fontweight='bold')
ax_a.set_ylabel('Mean ROC-AUC (+/- SD)', fontsize=11, fontweight='bold')
ax_a.set_title('A: Cross-Validated Discrimination (ROC-AUC) Across Models', fontsize=12, fontweight='bold', pad=10)
ax_a.set_ylim(0.3, 0.95)
ax_a.legend(loc='lower right', frameon=True)

# Panel B: Average Precision Mean Across Models & Subsets
ax_b = axes[0, 1]
for s_idx, (s_name, s_lbl) in enumerate(zip(subsets, subset_labels)):
    sub = df_prim[df_prim['feature_subset'] == s_name].set_index('model_type').reindex(model_order)
    vals = sub['average_precision_mean'].values
    errs = sub['average_precision_sd'].values
    ax_b.bar(x + (s_idx - 1) * w, vals, w, yerr=errs, capsize=3, label=s_lbl, color=colors[s_idx], edgecolor='black')

prevalence = 7.0 / 62.0  # ~0.113 in matched_both
ax_b.axhline(prevalence, color='red', linestyle='--', linewidth=1.2, label=f'Prevalence Baseline ({prevalence:.3f})')
ax_b.set_xticks(x)
ax_b.set_xticklabels(model_labels, fontsize=10, fontweight='bold')
ax_b.set_ylabel('Mean Average Precision (+/- SD)', fontsize=11, fontweight='bold')
ax_b.set_title('B: Precision-Recall Performance (Average Precision)', fontsize=12, fontweight='bold', pad=10)
ax_b.set_ylim(0.0, 0.65)
ax_b.legend(loc='upper right', frameon=True)

# Panel C: Sampling Rate Concordance (100 Hz vs 500 Hz on Matched Both)
ax_c = axes[1, 0]
df_mb_100 = df_metrics[(df_metrics['cohort'] == 'matched_both') & (df_metrics['sampling_rate'] == 100) & (df_metrics['is_primary'] == True)]
df_mb_500 = df_metrics[(df_metrics['cohort'] == 'matched_both') & (df_metrics['sampling_rate'] == 500) & (df_metrics['is_primary'] == True)]
merged_mb = pd.merge(df_mb_100, df_mb_500, on=['model_type', 'feature_subset'], suffixes=('_100hz', '_500hz'))

ax_c.scatter(merged_mb['roc_auc_mean_100hz'], merged_mb['roc_auc_mean_500hz'], color='#756bb1', s=60, edgecolor='black', zorder=5)
for _, r in merged_mb.iterrows():
    ax_c.annotate(f"{r['model_type']}", (r['roc_auc_mean_100hz'] + 0.005, r['roc_auc_mean_500hz'] - 0.005), fontsize=8)

ax_c.plot([0.4, 0.85], [0.4, 0.85], 'k--', alpha=0.6, label='1:1 Line of Equality')
ax_c.set_xlabel('100 Hz Mean ROC-AUC', fontsize=11, fontweight='bold')
ax_c.set_ylabel('500 Hz Mean ROC-AUC', fontsize=11, fontweight='bold')
ax_c.set_title('C: Discretization Sensitivity (100 Hz vs 500 Hz ROC-AUC)', fontsize=12, fontweight='bold', pad=10)
ax_c.set_xlim(0.4, 0.85)
ax_c.set_ylim(0.4, 0.85)
ax_c.legend(loc='lower right', frameon=True)

# Panel D: Score Type & Calibration Architecture
ax_d = axes[1, 1]
score_types = df_metrics.groupby('model_type')['score_type'].first().reindex(model_order)
categories = ['Calibrated Probabilities\n(RF, XGB)', 'Raw Margin / Distance\n(SVM Lin, SVM RBF, KNN)']
model_types_desc = [
    'Random Forest: predict_proba() [0, 1]',
    'XGBoost: predict_proba() [0, 1]',
    'Linear SVM: decision_function() [-inf, +inf]',
    'RBF SVM: decision_function() [-inf, +inf]',
    'KNN (k=5): Distance-Weighted Voting [0, 1]'
]
y_pos = np.arange(len(model_types_desc))
bar_colors_d = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', '#8c564b']
ax_d.barh(y_pos[::-1], [1]*len(model_types_desc), color=bar_colors_d[::-1], edgecolor='black', height=0.6)
ax_d.set_yticks(y_pos[::-1])
ax_d.set_yticklabels(model_types_desc, fontsize=10, fontweight='bold')
ax_d.set_xlim(0, 1.2)
ax_d.set_xticks([])
ax_d.set_title('D: Model Family Output Taxonomy & Scoring Orientation', fontsize=12, fontweight='bold', pad=10)

plt.tight_layout()
plt.savefig(target_path_1, dpi=300, bbox_inches='tight')
plt.close()
print(f"Successfully generated: {target_path_1} ({target_path_1.stat().st_size} bytes)")

# Figure 2: 11_knn_sensitivity_and_cohort_strata_comparison.png
fig, axes = plt.subplots(2, 2, figsize=(16, 11), dpi=300)

# Panel A: KNN Sensitivity (k=3, 5, 7) on Subset A (ST Dynamics)
ax_a = axes[0, 0]
knn_models = ['knn_k3', 'knn_k5', 'knn_k7']
knn_labels = ['KNN (k=3)', 'KNN (k=5)', 'KNN (k=7)']
x_k = np.arange(3)
w_k = 0.35

sub_knn_100 = df_metrics[(df_metrics['cohort'] == 'matched_both') & (df_metrics['sampling_rate'] == 100) & (df_metrics['feature_subset'] == 'Subset_A_ST_Dynamics') & (df_metrics['model_type'].isin(knn_models))].set_index('model_type').reindex(knn_models)
sub_knn_500 = df_metrics[(df_metrics['cohort'] == 'matched_both') & (df_metrics['sampling_rate'] == 500) & (df_metrics['feature_subset'] == 'Subset_A_ST_Dynamics') & (df_metrics['model_type'].isin(knn_models))].set_index('model_type').reindex(knn_models)

b1 = ax_a.bar(x_k - w_k/2, sub_knn_100['roc_auc_mean'], w_k, yerr=sub_knn_100['roc_auc_sd'], capsize=4, label='100 Hz', color='#4575b4', edgecolor='black')
b2 = ax_a.bar(x_k + w_k/2, sub_knn_500['roc_auc_mean'], w_k, yerr=sub_knn_500['roc_auc_sd'], capsize=4, label='500 Hz', color='#f46d43', edgecolor='black')
ax_a.axhline(0.50, color='gray', linestyle='--', linewidth=1.2, label='Chance Baseline')
ax_a.set_xticks(x_k)
ax_a.set_xticklabels(knn_labels, fontsize=11, fontweight='bold')
ax_a.set_ylabel('Mean ROC-AUC (+/- SD)', fontsize=11, fontweight='bold')
ax_a.set_title('A: KNN Sensitivity Analysis (k=3 vs 5 vs 7 on Subset A)', fontsize=12, fontweight='bold', pad=10)
ax_a.set_ylim(0.4, 0.85)
ax_a.legend(loc='upper right', frameon=True)

# Panel B: KNN Average Precision Sensitivity
ax_b = axes[0, 1]
b1_ap = ax_b.bar(x_k - w_k/2, sub_knn_100['average_precision_mean'], w_k, yerr=sub_knn_100['average_precision_sd'], capsize=4, label='100 Hz', color='#4575b4', edgecolor='black')
b2_ap = ax_b.bar(x_k + w_k/2, sub_knn_500['average_precision_mean'], w_k, yerr=sub_knn_500['average_precision_sd'], capsize=4, label='500 Hz', color='#f46d43', edgecolor='black')
ax_b.axhline(prevalence, color='red', linestyle='--', linewidth=1.2, label='Prevalence Baseline')
ax_b.set_xticks(x_k)
ax_b.set_xticklabels(knn_labels, fontsize=11, fontweight='bold')
ax_b.set_ylabel('Mean Average Precision (+/- SD)', fontsize=11, fontweight='bold')
ax_b.set_title('B: KNN Average Precision Sensitivity (Subset A)', fontsize=12, fontweight='bold', pad=10)
ax_b.set_ylim(0.0, 0.5)
ax_b.legend(loc='upper right', frameon=True)

# Panel C: Cohort Strata Comparison Across Primary Models
ax_c = axes[1, 0]
strata = ['all_subjects', 'matched_500hz', 'matched_both']
strata_labels = ['All-Subjects (N=78)', 'Matched-500Hz (N=62)', 'Matched-Both (N=62)']
strata_colors = ['#66c2a5', '#fc8d62', '#8da0cb']
x_s = np.arange(len(model_order))
w_s = 0.25

for s_i, (strat, s_lbl) in enumerate(zip(strata, strata_labels)):
    sub_strat = df_metrics[(df_metrics['cohort'] == strat) & (df_metrics['feature_subset'] == 'Subset_A_ST_Dynamics') & (df_metrics['sampling_rate'] == (100 if strat == 'all_subjects' else 500))].set_index('model_type').reindex(model_order)
    ax_c.bar(x_s + (s_i - 1) * w_s, sub_strat['roc_auc_mean'], w_s, label=s_lbl, color=strata_colors[s_i], edgecolor='black')

ax_c.axhline(0.50, color='gray', linestyle='--', linewidth=1.2)
ax_c.set_xticks(x_s)
ax_c.set_xticklabels(model_labels, fontsize=10, fontweight='bold')
ax_c.set_ylabel('Mean ROC-AUC', fontsize=11, fontweight='bold')
ax_c.set_title('C: Cohort Partitioning Stability (Subset A ST Dynamics)', fontsize=12, fontweight='bold', pad=10)
ax_c.set_ylim(0.35, 0.85)
ax_c.legend(loc='lower right', frameon=True)

# Panel D: Feature Subsets Yield Comparison Across 5 Model Families
ax_d = axes[1, 1]
sub_a_means = df_prim[df_prim['feature_subset'] == 'Subset_A_ST_Dynamics'].set_index('model_type').reindex(model_order)['roc_auc_mean'].values
sub_b_means = df_prim[df_prim['feature_subset'] == 'Subset_B_Autonomic'].set_index('model_type').reindex(model_order)['roc_auc_mean'].values
sub_c_means = df_prim[df_prim['feature_subset'] == 'Subset_C_Multimodal'].set_index('model_type').reindex(model_order)['roc_auc_mean'].values

for m_i, m_lbl in enumerate(model_labels):
    ax_d.plot(['Subset A (ST)', 'Subset B (HRV)', 'Subset C (Multi)'],
              [sub_a_means[m_i], sub_b_means[m_i], sub_c_means[m_i]],
              marker='o', linewidth=2.0, markersize=8, label=m_lbl)

ax_d.axhline(0.50, color='gray', linestyle='--', linewidth=1.2, label='Chance Baseline')
ax_d.set_ylabel('Mean ROC-AUC', fontsize=11, fontweight='bold')
ax_d.set_title('D: Feature Domain Yield Trajectories Across Model Architectures', fontsize=12, fontweight='bold', pad=10)
ax_d.set_ylim(0.4, 0.85)
ax_d.legend(loc='lower right', frameon=True)

plt.tight_layout()
plt.savefig(target_path_2, dpi=300, bbox_inches='tight')
plt.close()
print(f"Successfully generated: {target_path_2} ({target_path_2.stat().st_size} bytes)")

