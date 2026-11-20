"""
05_stage09_11_ml_model_benchmarking.py

Generates:
- COMPARISON/08_VALIDATION_COMPARISON/stage09_validation_protocol_comparison.csv
- COMPARISON/09_MODEL_PERFORMANCE/model_comprehensive_benchmark_metrics.csv
- COMPARISON/09_MODEL_PERFORMANCE/model_operating_point_metrics.csv
- COMPARISON/09_MODEL_PERFORMANCE/model_confusion_matrices.csv
- Figures:
  - COMPARISON/figures/fig18_stage09_validation_splitting_scheme.png / .pdf
  - COMPARISON/figures/fig19_stage10_11_roc_curves_comparison.png / .pdf
  - COMPARISON/figures/fig20_stage10_11_pr_curves_comparison.png / .pdf
  - COMPARISON/figures/fig21_stage10_11_model_performance_heatmap.png / .pdf
  - COMPARISON/figures/fig22_stage10_11_operating_point_metrics_forest.png / .pdf
  - COMPARISON/figures/fig23_stage10_11_prediction_score_distributions.png / .pdf
  - COMPARISON/figures/fig24_stage10_11_confusion_matrices.png / .pdf
"""

import os
os.environ['MPLCONFIGDIR'] = os.path.abspath('scratch/.matplotlib')

import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, confusion_matrix
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.dpi'] = 300

os.makedirs('COMPARISON/08_VALIDATION_COMPARISON', exist_ok=True)
os.makedirs('COMPARISON/09_MODEL_PERFORMANCE', exist_ok=True)
os.makedirs('COMPARISON/figures', exist_ok=True)
os.makedirs('COMPARISON/tables', exist_ok=True)

print("--- Step 5: Processing Stages 09, 10, and 11 ML Benchmarking ---")

# --- Stage 09 Validation Protocol Comparison ---
stage09_comparison = [
    {
        "Validation_Protocol_Dimension": "Partitioning Strategy",
        "Our_MSIMI_Pipeline": "Repeated 5-fold cross-validation with 5 repeats (25 total folds); patient-stratified",
        "Literature_Standard": "Patient-level grouped / stratified cross-validation",
        "Benchmark_Reference": "Varoquaux, 2018 (NeuroImage, DOI: 10.1016/j.neuroimage.2017.06.061)",
        "Compliance_Evaluation": "Full compliance; prevents overly optimistic bias by averaging multiple partitionings."
    },
    {
        "Validation_Protocol_Dimension": "Data Leakage Prevention",
        "Our_MSIMI_Pipeline": "Patient-level partitioning; all preprocessing (imputation, scaling, weights) fitted strictly on training folds",
        "Literature_Standard": "Zero cross-patient contamination; preprocessing inside CV loops",
        "Benchmark_Reference": "Saeb et al., 2017 (GigaScience, DOI: 10.1093/gigascience/gix083)",
        "Compliance_Evaluation": "Rigorous compliance; zero validation-fold hyperparameter tuning or leakage."
    },
    {
        "Validation_Protocol_Dimension": "Holdout Sample Segregation",
        "Our_MSIMI_Pipeline": "sub_005 and sub_049 held out strictly for inference; completely excluded from training & metrics",
        "Literature_Standard": "Independent test set completely blinded during model development",
        "Benchmark_Reference": "Little et al., 2017 (Nat Biomed Eng)",
        "Compliance_Evaluation": "Enforced by automated integrity assertions; zero label contamination."
    },
    {
        "Validation_Protocol_Dimension": "Metric Selection under Class Imbalance",
        "Our_MSIMI_Pipeline": "Primary: Average Precision (PR-AUC) & ROC-AUC; Balanced Accuracy, Sensitivity, Specificity (raw accuracy never reported alone)",
        "Literature_Standard": "Avoid raw accuracy on imbalanced clinical datasets; emphasize PR-AUC and Balanced Acc",
        "Benchmark_Reference": "Saito & Rehmsmeier, 2015 (PLoS ONE); Al-Zaiti et al., 2023 (Nat Med)",
        "Compliance_Evaluation": "Fully compliant with international guidelines on imbalanced medical ML."
    }
]

df_s09 = pd.DataFrame(stage09_comparison)
df_s09.to_csv('COMPARISON/08_VALIDATION_COMPARISON/stage09_validation_protocol_comparison.csv', index=False)
df_s09.to_csv('COMPARISON/tables/stage09_validation_protocol_comparison.csv', index=False)
print("Saved stage09_validation_protocol_comparison.csv")

# --- Load Predictions from Stage 10 and Stage 11 ---
df_p10 = pd.read_csv('reports/models/10_patient_predictions.csv')
df_p11 = pd.read_csv('notebooks/reports/models/11_extended_patient_predictions.csv')

# Standardize column types
df_p10_oof = df_p10[df_p10['is_holdout'] == False].copy()
df_p11_oof = df_p11[df_p11['is_holdout'] == False].copy()

# Map model names to clean display labels
model_map = {
    'ridge': 'Logistic Regression (Ridge)',
    'lda': 'Linear Discriminant Analysis (LDA)',
    'rank_st': 'Rank-ST (Heuristic Benchmark)',
    'rf': 'Random Forest',
    'xgb': 'XGBoost',
    'svm_linear': 'Linear SVM',
    'svm_rbf': 'RBF SVM',
    'knn_k5': 'KNN (k=5, Primary)',
    'knn_k3': 'KNN (k=3, Sensitivity)',
    'knn_k7': 'KNN (k=7, Sensitivity)'
}

# Aggregate repeated CV predictions per patient to obtain true independent patient-level ensemble scores
def aggregate_patient_scores(df_preds):
    grouped = df_preds.groupby(['cohort', 'feature_subset', 'sampling_rate', 'model_type', 'subject_id']).agg({
        'target_label': 'first',
        'predicted_score': 'mean'
    }).reset_index()
    return grouped

df_p10_agg = aggregate_patient_scores(df_p10_oof)
df_p11_agg = aggregate_patient_scores(df_p11_oof)

all_preds_agg = pd.concat([df_p10_agg, df_p11_agg], ignore_index=True)
all_preds_agg['model_name'] = all_preds_agg['model_type'].map(model_map)

# Compute Comprehensive Benchmark Metrics across all configurations
benchmark_rows = []
operating_rows = []
cm_rows = []

configs = all_preds_agg.groupby(['cohort', 'feature_subset', 'sampling_rate', 'model_type'])

for (cohort, subset, sr, mtype), group in configs:
    y_true = group['target_label'].values.astype(int)
    y_score = group['predicted_score'].values
    n_total = len(y_true)
    n_pos = int(y_true.sum())
    n_neg = n_total - n_pos
    prevalence = n_pos / n_total if n_total > 0 else np.nan
    
    # Check if both classes present
    if len(np.unique(y_true)) < 2:
        continue
        
    fpr, tpr, roc_thresh = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    
    precision, recall, pr_thresh = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    
    # Youden's J statistic for optimal threshold
    j_scores = tpr - fpr
    opt_idx = np.argmax(j_scores)
    opt_thresh = roc_thresh[opt_idx]
    
    y_pred = (y_score >= opt_thresh).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn, fp, fn, tp = 0, 0, 0, 0
        
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    bal_acc = (sens + spec) / 2.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * (prec * sens) / (prec + sens) if (prec + sens) > 0 else 0.0
    raw_acc = (tp + tn) / n_total
    majority_acc = n_neg / n_total # majority negative class baseline
    
    benchmark_rows.append({
        'Cohort': cohort,
        'Sampling_Rate_Hz': sr,
        'Feature_Subset': subset,
        'Model_Type': mtype,
        'Model_Name': model_map.get(mtype, mtype),
        'N_Patients': n_total,
        'N_Positive': n_pos,
        'N_Negative': n_neg,
        'Prevalence': round(prevalence, 4),
        'ROC_AUC': round(roc_auc, 4),
        'Average_Precision': round(ap, 4),
        'Prevalence_Baseline_AP': round(prevalence, 4),
        'ROC_AUC_Baseline': 0.5000,
        'Balanced_Accuracy': round(bal_acc, 4),
        'Sensitivity': round(sens, 4),
        'Specificity': round(spec, 4),
        'Precision_PPV': round(prec, 4),
        'F1_Score': round(f1, 4),
        'Raw_Accuracy': round(raw_acc, 4),
        'Majority_Class_Baseline_Accuracy': round(majority_acc, 4)
    })
    
    operating_rows.append({
        'Cohort': cohort,
        'Sampling_Rate_Hz': sr,
        'Feature_Subset': subset,
        'Model_Type': mtype,
        'Model_Name': model_map.get(mtype, mtype),
        'Optimal_Threshold': round(opt_thresh, 4),
        'Sensitivity': round(sens, 4),
        'Specificity': round(spec, 4),
        'Balanced_Accuracy': round(bal_acc, 4),
        'PPV': round(prec, 4),
        'NPV': round(tn / (tn + fn) if (tn + fn) > 0 else 0.0, 4),
        'F1_Score': round(f1, 4),
        'Youden_J': round(float(tpr[opt_idx] - fpr[opt_idx]), 4)
    })
    
    cm_rows.append({
        'Cohort': cohort,
        'Sampling_Rate_Hz': sr,
        'Feature_Subset': subset,
        'Model_Type': mtype,
        'Model_Name': model_map.get(mtype, mtype),
        'True_Negative': tn,
        'False_Positive': fp,
        'False_Negative': fn,
        'True_Positive': tp
    })

df_bench = pd.DataFrame(benchmark_rows)
df_oper = pd.DataFrame(operating_rows)
df_cm = pd.DataFrame(cm_rows)

df_bench.to_csv('COMPARISON/09_MODEL_PERFORMANCE/model_comprehensive_benchmark_metrics.csv', index=False)
df_bench.to_csv('COMPARISON/tables/model_comprehensive_benchmark_metrics.csv', index=False)

df_oper.to_csv('COMPARISON/09_MODEL_PERFORMANCE/model_operating_point_metrics.csv', index=False)
df_oper.to_csv('COMPARISON/tables/model_operating_point_metrics.csv', index=False)

df_cm.to_csv('COMPARISON/09_MODEL_PERFORMANCE/model_confusion_matrices.csv', index=False)
df_cm.to_csv('COMPARISON/tables/model_confusion_matrices.csv', index=False)

print("Saved all 3 ML benchmark tables!")

# --- PLOTTING ML FIGURES (18 to 24) ---
print("Generating ML figures 18 through 24...")

# Figure 18: Validation Splitting Scheme Diagram
fig, ax = plt.subplots(figsize=(11, 5.5))
ax.axis('off')
ax.set_xlim(0, 10)
ax.set_ylim(0, 8)

# Title
ax.text(5, 7.5, 'Figure 18: Patient-Level Stratified Cross-Validation Architecture', ha='center', fontsize=12, fontweight='bold')

# Scheme boxes
box_props = dict(boxstyle='round,pad=0.5', facecolor='#2b5c8f', edgecolor='black', alpha=0.9)
box_leak = dict(boxstyle='round,pad=0.5', facecolor='#d95f02', edgecolor='black', alpha=0.9)
box_card = dict(boxstyle='round,pad=0.4', facecolor='#f0f0f0', edgecolor='gray')

ax.text(2.5, 6.2, 'A: Common Leaky Validation (Flawed)', ha='center', fontsize=11, fontweight='bold', color='#d95f02')
ax.text(2.5, 5.0, 'Recordings split randomly across folds\n• Same patient in Train & Validation\n• Artificially inflated accuracy\n• Memorizes patient-specific ECG signature',
        ha='center', va='center', bbox=box_card, fontsize=9)

ax.text(7.5, 6.2, 'B: Our Patient-Level Stratified Design (Rigorous)', ha='center', fontsize=11, fontweight='bold', color='#2b5c8f')
ax.text(7.5, 5.0, 'Grouped by Subject ID across 25 folds (5x5 CV)\n• Zero intra-patient leakage across splits\n• Preprocessing fitted on training folds only\n• Unlabelled holdouts (sub_005, sub_049) isolated\n• Metrics reported on independent patient OOF',
        ha='center', va='center', bbox=box_card, fontsize=9)

# Bottom summary box
ax.text(5, 2.2, 'Evaluation Cohort Stratification:\n• full_78 Cohort: N=78 (9 Pos, 69 Neg, Prevalence 11.54%, Ratio 1:7.67)\n• matched_62 Cohort: N=62 (7 Pos, 55 Neg, Prevalence 11.29%, Ratio 1:7.86)\n• Inference Holdouts: sub_005, sub_049 (N=2, strictly excluded from CV metrics)',
        ha='center', va='center', bbox=dict(boxstyle='round,pad=0.6', facecolor='#e6f2ff', edgecolor='#2b5c8f', linewidth=1.5), fontsize=9.5)

plt.tight_layout()
fig.savefig('COMPARISON/figures/fig18_stage09_validation_splitting_scheme.png', bbox_inches='tight')
fig.savefig('COMPARISON/figures/fig18_stage09_validation_splitting_scheme.pdf', bbox_inches='tight')
plt.close(fig)
print("Saved fig18")

# Figure 19: ROC Curves (Primary models on full_78, Subset A)
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

plot_models = ['rf', 'xgb', 'svm_linear', 'svm_rbf', 'knn_k5', 'ridge', 'lda', 'rank_st']
palette = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#a65628', '#f781bf', '#999999']

for ax_idx, (coh, title_str) in enumerate([('full_78', 'A: full_78 Cohort (N=78, 500 Hz)'), ('matched_62', 'B: matched_62 Cohort (N=62, 500 Hz)')]):
    for mtype, color in zip(plot_models, palette):
        sub = all_preds_agg[(all_preds_agg['cohort'] == coh) &
                            (all_preds_agg['feature_subset'] == 'Subset_A_ST_Dynamics') &
                            (all_preds_agg['sampling_rate'] == 500) &
                            (all_preds_agg['model_type'] == mtype)]
        if len(sub) > 0 and len(np.unique(sub['target_label'])) == 2:
            fpr, tpr, _ = roc_curve(sub['target_label'], sub['predicted_score'])
            score_auc = auc(fpr, tpr)
            axes[ax_idx].plot(fpr, tpr, color=color, linewidth=2.0, label=f'{model_map.get(mtype, mtype)[:16]} (AUC={score_auc:.3f})')
            
    axes[ax_idx].plot([0, 1], [0, 1], 'k--', linewidth=1.2, label='Chance (AUC=0.500)')
    axes[ax_idx].set_title(title_str, fontsize=11, fontweight='bold')
    axes[ax_idx].set_xlabel('False Positive Rate (1 - Specificity)', fontsize=10)
    axes[ax_idx].set_ylabel('True Positive Rate (Sensitivity)', fontsize=10)
    axes[ax_idx].grid(linestyle=':', alpha=0.6)
    axes[ax_idx].legend(loc='lower right', fontsize=8, frameon=True, facecolor='white')

plt.suptitle('Figure 19: Out-of-Fold Receiver Operating Characteristic (ROC) Benchmark Curves (Subset A)', fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
fig.savefig('COMPARISON/figures/fig19_stage10_11_roc_curves_comparison.png', bbox_inches='tight')
fig.savefig('COMPARISON/figures/fig19_stage10_11_roc_curves_comparison.pdf', bbox_inches='tight')
plt.close(fig)
print("Saved fig19")

# Figure 20: Precision-Recall Curves (with prevalence baseline)
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax_idx, (coh, prev_val, title_str) in enumerate([
    ('full_78', 0.1154, 'A: full_78 Cohort (Prevalence = 11.5%)'),
    ('matched_62', 0.1129, 'B: matched_62 Cohort (Prevalence = 11.3%)')
]):
    for mtype, color in zip(plot_models, palette):
        sub = all_preds_agg[(all_preds_agg['cohort'] == coh) &
                            (all_preds_agg['feature_subset'] == 'Subset_A_ST_Dynamics') &
                            (all_preds_agg['sampling_rate'] == 500) &
                            (all_preds_agg['model_type'] == mtype)]
        if len(sub) > 0 and len(np.unique(sub['target_label'])) == 2:
            prec, rec, _ = precision_recall_curve(sub['target_label'], sub['predicted_score'])
            ap_val = average_precision_score(sub['target_label'], sub['predicted_score'])
            axes[ax_idx].plot(rec, prec, color=color, linewidth=2.0, label=f'{model_map.get(mtype, mtype)[:16]} (AP={ap_val:.3f})')
            
    axes[ax_idx].axhline(prev_val, color='black', linestyle='--', linewidth=1.5, label=f'Prevalence Baseline ({prev_val*100:.1f}%)')
    axes[ax_idx].set_title(title_str, fontsize=11, fontweight='bold')
    axes[ax_idx].set_xlabel('Recall (Sensitivity)', fontsize=10)
    axes[ax_idx].set_ylabel('Precision (Positive Predictive Value)', fontsize=10)
    axes[ax_idx].set_ylim(-0.02, 1.02)
    axes[ax_idx].grid(linestyle=':', alpha=0.6)
    axes[ax_idx].legend(loc='upper right', fontsize=8, frameon=True, facecolor='white')

plt.suptitle('Figure 20: Out-of-Fold Precision-Recall (PR) Curves Relative to Empirical Prevalence Baseline', fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
fig.savefig('COMPARISON/figures/fig20_stage10_11_pr_curves_comparison.png', bbox_inches='tight')
fig.savefig('COMPARISON/figures/fig20_stage10_11_pr_curves_comparison.pdf', bbox_inches='tight')
plt.close(fig)
print("Saved fig20")

# Figure 21: Model Performance Heatmap (ROC-AUC & Average Precision)
df_sub500 = df_bench[(df_bench['Cohort'] == 'full_78') & (df_bench['Sampling_Rate_Hz'] == 500)]
piv_auc = df_sub500.pivot(index='Model_Name', columns='Feature_Subset', values='ROC_AUC')
piv_ap = df_sub500.pivot(index='Model_Name', columns='Feature_Subset', values='Average_Precision')

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

im0 = axes[0].imshow(piv_auc, cmap='YlGnBu', vmin=0.4, vmax=0.85)
axes[0].set_xticks(range(len(piv_auc.columns)))
axes[0].set_yticks(range(len(piv_auc.index)))
axes[0].set_xticklabels(piv_auc.columns, rotation=30, ha='right', fontsize=9, fontweight='bold')
axes[0].set_yticklabels(piv_auc.index, fontsize=9, fontweight='bold')
axes[0].set_title('A: ROC-AUC Across Feature Subsets (full_78)', fontsize=11, fontweight='bold')
fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

for i in range(len(piv_auc)):
    for j in range(len(piv_auc.columns)):
        val = piv_auc.iloc[i, j]
        axes[0].text(j, i, f'{val:.3f}' if np.isfinite(val) else 'N/A', ha='center', va='center',
                     fontsize=9, color='white' if val > 0.65 else 'black', fontweight='bold')

im1 = axes[1].imshow(piv_ap, cmap='YlOrRd', vmin=0.10, vmax=0.45)
axes[1].set_xticks(range(len(piv_ap.columns)))
axes[1].set_yticks(range(len(piv_ap.index)))
axes[1].set_xticklabels(piv_ap.columns, rotation=30, ha='right', fontsize=9, fontweight='bold')
axes[1].set_yticklabels(piv_ap.index, fontsize=9, fontweight='bold')
axes[1].set_title('B: Average Precision Across Feature Subsets (full_78)', fontsize=11, fontweight='bold')
fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

for i in range(len(piv_ap)):
    for j in range(len(piv_ap.columns)):
        val = piv_ap.iloc[i, j]
        axes[1].text(j, i, f'{val:.3f}' if np.isfinite(val) else 'N/A', ha='center', va='center',
                     fontsize=9, color='white' if val > 0.30 else 'black', fontweight='bold')

plt.tight_layout()
fig.savefig('COMPARISON/figures/fig21_stage10_11_model_performance_heatmap.png', bbox_inches='tight')
fig.savefig('COMPARISON/figures/fig21_stage10_11_model_performance_heatmap.pdf', bbox_inches='tight')
plt.close(fig)
print("Saved fig21")

# Figure 22: Operating Point Metrics (Sensitivity, Specificity, Balanced Accuracy)
fig, ax = plt.subplots(figsize=(12, 6))

df_plot_sub = df_bench[(df_bench['Cohort'] == 'full_78') &
                       (df_bench['Feature_Subset'] == 'Subset_A_ST_Dynamics') &
                       (df_bench['Sampling_Rate_Hz'] == 500)].copy()

y_positions = np.arange(len(df_plot_sub))
bar_width = 0.25

ax.barh(y_positions - bar_width, df_plot_sub['Sensitivity'], height=bar_width, color='#e41a1c', label='Sensitivity (Recall)', edgecolor='black')
ax.barh(y_positions, df_plot_sub['Specificity'], height=bar_width, color='#377eb8', label='Specificity', edgecolor='black')
ax.barh(y_positions + bar_width, df_plot_sub['Balanced_Accuracy'], height=bar_width, color='#4daf4a', label='Balanced Accuracy', edgecolor='black')

ax.axvline(0.5, color='gray', linestyle=':', label='Chance (0.50)')
ax.set_yticks(y_positions)
ax.set_yticklabels(df_plot_sub['Model_Name'], fontsize=9, fontweight='bold')
ax.set_xlabel('Metric Value (0.0 to 1.0)', fontsize=10, fontweight='bold')
ax.set_title('Figure 22: Operating Point Metrics at Optimal Youden Cutoff (full_78, Subset A)', fontsize=12, fontweight='bold')
ax.set_xlim(0, 1.05)
ax.grid(axis='x', linestyle=':', alpha=0.6)
ax.legend(loc='lower right', frameon=True, facecolor='white', edgecolor='gray')

plt.tight_layout()
fig.savefig('COMPARISON/figures/fig22_stage10_11_operating_point_metrics_forest.png', bbox_inches='tight')
fig.savefig('COMPARISON/figures/fig22_stage10_11_operating_point_metrics_forest.pdf', bbox_inches='tight')
plt.close(fig)
print("Saved fig22")

# Figure 23: Prediction Score Distributions (Positive vs Negative)
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

dist_models = [('rf', 'Random Forest'), ('xgb', 'XGBoost'), ('svm_linear', 'Linear SVM'),
               ('svm_rbf', 'RBF SVM'), ('knn_k5', 'KNN (k=5)'), ('ridge', 'Ridge Regression')]

for idx, (mtype, mlabel) in enumerate(dist_models):
    sub = all_preds_agg[(all_preds_agg['cohort'] == 'full_78') &
                        (all_preds_agg['feature_subset'] == 'Subset_A_ST_Dynamics') &
                        (all_preds_agg['sampling_rate'] == 500) &
                        (all_preds_agg['model_type'] == mtype)]
    
    pos_scores = sub[sub['target_label'] == 1]['predicted_score']
    neg_scores = sub[sub['target_label'] == 0]['predicted_score']
    
    axes[idx].hist(neg_scores, bins=15, alpha=0.6, color='#2b5c8f', edgecolor='black', label=f'MSIMI- (N={len(neg_scores)})')
    axes[idx].hist(pos_scores, bins=10, alpha=0.8, color='#d95f02', edgecolor='black', label=f'MSIMI+ (N={len(pos_scores)})')
    axes[idx].set_title(mlabel, fontsize=11, fontweight='bold')
    axes[idx].set_xlabel('Ensemble OOF Predicted Score', fontsize=9)
    axes[idx].set_ylabel('Patient Count', fontsize=9)
    axes[idx].grid(axis='y', linestyle=':', alpha=0.6)
    axes[idx].legend(loc='upper right', fontsize=8, frameon=True, facecolor='white')

plt.suptitle('Figure 23: Out-of-Fold Model Score Distributions Stratified by True Label (full_78, Subset A)', fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
fig.savefig('COMPARISON/figures/fig23_stage10_11_prediction_score_distributions.png', bbox_inches='tight')
fig.savefig('COMPARISON/figures/fig23_stage10_11_prediction_score_distributions.pdf', bbox_inches='tight')
plt.close(fig)
print("Saved fig23")

# Figure 24: Confusion Matrix Grid
fig, axes = plt.subplots(2, 3, figsize=(13, 8))
axes = axes.flatten()

for idx, (mtype, mlabel) in enumerate(dist_models):
    row_cm = df_cm[(df_cm['Cohort'] == 'full_78') &
                   (df_cm['Feature_Subset'] == 'Subset_A_ST_Dynamics') &
                   (df_cm['Sampling_Rate_Hz'] == 500) &
                   (df_cm['Model_Type'] == mtype)]
    
    if len(row_cm) > 0:
        r = row_cm.iloc[0]
        matrix = np.array([[r['True_Negative'], r['False_Positive']],
                           [r['False_Negative'], r['True_Positive']]])
        
        im = axes[idx].imshow(matrix, cmap='Blues', interpolation='nearest')
        axes[idx].set_xticks([0, 1])
        axes[idx].set_yticks([0, 1])
        axes[idx].set_xticklabels(['Pred Neg', 'Pred Pos'], fontsize=9, fontweight='bold')
        axes[idx].set_yticklabels(['True Neg', 'True Pos'], fontsize=9, fontweight='bold')
        axes[idx].set_title(mlabel, fontsize=11, fontweight='bold')
        
        for i in range(2):
            for j in range(2):
                val = matrix[i, j]
                axes[idx].text(j, i, str(val), ha='center', va='center',
                               fontsize=14, fontweight='bold', color='white' if val > 20 else 'black')

plt.suptitle('Figure 24: Binary Classification Confusion Matrices at Optimal Youden Cutoff (full_78, Subset A)', fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
fig.savefig('COMPARISON/figures/fig24_stage10_11_confusion_matrices.png', bbox_inches='tight')
fig.savefig('COMPARISON/figures/fig24_stage10_11_confusion_matrices.pdf', bbox_inches='tight')
plt.close(fig)
print("Saved fig24")

print("Stages 09, 10, and 11 completed successfully!")
