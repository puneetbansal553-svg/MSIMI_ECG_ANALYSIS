"""
03_stage05_rest_stress_recovery_comprehensive.py

Performs comprehensive paired REST vs STRESS vs RECOVERY statistical analysis
across all ECG and HRV features using actual patient-level recording data.

Outputs:
- COMPARISON/05_REST_STRESS_RECOVERY/rest_stress_recovery_comprehensive_stats.csv
- COMPARISON/05_REST_STRESS_RECOVERY/rest_stress_recovery_paired_differences.csv
- COMPARISON/05_REST_STRESS_RECOVERY/rest_stress_recovery_percentage_changes.csv
- COMPARISON/05_REST_STRESS_RECOVERY/rest_stress_recovery_hypothesis_tests.csv
- COMPARISON/figures/fig07_rest_stress_recovery_boxplots.png / .pdf
- COMPARISON/figures/fig08_rest_stress_recovery_violins.png / .pdf
- COMPARISON/figures/fig09_rest_stress_recovery_mean_ci.png / .pdf
- COMPARISON/figures/fig10_rest_stress_recovery_median_comparison.png / .pdf
- COMPARISON/figures/fig11_individual_patient_trajectories.png / .pdf
- COMPARISON/figures/fig12_paired_differences_distributions.png / .pdf
- COMPARISON/figures/fig13_feature_transition_heatmaps.png / .pdf
- COMPARISON/figures/fig14_sampling_rate_100hz_vs_500hz_comparison.png / .pdf
- COMPARISON/figures/fig15_statistical_significance_volcano.png / .pdf
"""

import os
os.environ['MPLCONFIGDIR'] = os.path.abspath('scratch/.matplotlib')

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.dpi'] = 300

os.makedirs('COMPARISON/05_REST_STRESS_RECOVERY', exist_ok=True)
os.makedirs('COMPARISON/figures', exist_ok=True)
os.makedirs('COMPARISON/tables', exist_ok=True)

print("--- Step 3: Loading Feature Store and Preparing Paired Cohort ---")
df_rec = pd.read_csv('data/features/08_recording_level_features.csv')
df_p06 = pd.read_csv('reports/hrv/06_rr_hrv_manifest.csv')

# Merge empirical HRV and Heart Rate from Stage 06 manifest
hrv_cols = ['recording_id', 'gated_mean_hr_bpm', 'raw_detector_rr_median_rr_ms',
            'gated_sdnn_ms', 'gated_rmssd_ms', 'gated_pnn50_pct', 'exploratory_lf_hf_ratio']
df_rec = pd.merge(df_rec, df_p06[hrv_cols], on='recording_id', how='left')

# Define feature list for analysis
leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
st60_features = [f'{l}_st60_adc' for l in leads]
st80_features = [f'{l}_st80_adc' for l in leads]
global_st_features = [
    'min_st60_global_adc', 'min_st60_inferior_adc', 'min_st60_anterior_adc', 'min_st60_lateral_adc',
    'min_st80_global_adc', 'min_st80_inferior_adc', 'min_st80_anterior_adc', 'min_st80_lateral_adc'
]
hrv_autonomic_features = [
    'gated_mean_hr_bpm', 'raw_detector_rr_median_rr_ms', 'gated_sdnn_ms',
    'gated_rmssd_ms', 'gated_pnn50_pct', 'exploratory_lf_hf_ratio'
]
morphology_features = [
    'II_qrs_dur_ms', 'V5_qrs_dur_ms', 'II_t_amp_adc', 'V5_t_amp_adc',
    'II_qt_ms', 'V5_qt_ms', 'qrs_axis_degrees', 'contiguous_depressed_leads_count'
]

all_features = global_st_features + st60_features + st80_features + hrv_autonomic_features + morphology_features

# Feature metadata: unit and display name
def get_feature_meta(f):
    if 'adc' in f:
        unit = 'ADC counts'
    elif 'ms' in f:
        unit = 'ms'
    elif 'bpm' in f:
        unit = 'BPM'
    elif 'pct' in f:
        unit = '%'
    elif 'degrees' in f:
        unit = 'degrees'
    elif 'ratio' in f:
        unit = 'ratio'
    elif 'count' in f:
        unit = 'leads'
    else:
        unit = 'units'
    return unit

# Use 500 Hz as primary patient-level analysis (N=80 patients complete)
df_500 = df_rec[df_rec['sampling_rate'] == 500].copy()

# Ensure we pivot by patient and phase
phases = ['rest', 'stress', 'recover']
phase_labels = {'rest': 'REST', 'stress': 'STRESS', 'recover': 'RECOVERY'}

comprehensive_stats = []
paired_diff_rows = []
percentage_change_rows = []
hypothesis_test_rows = []

print("Computing statistics across", len(all_features), "features...")

for feat in all_features:
    piv = df_500.pivot(index='subject_id', columns='phase', values=feat)
    
    # Check if all phases exist
    for p in phases:
        if p not in piv.columns:
            piv[p] = np.nan
            
    rest_vals = piv['rest'].dropna()
    stress_vals = piv['stress'].dropna()
    rec_vals = piv['recover'].dropna()
    
    # Compute descriptive stats for each phase
    for p, vals in [('REST', rest_vals), ('STRESS', stress_vals), ('RECOVERY', rec_vals)]:
        n_obs = len(vals)
        mean_val = float(vals.mean()) if n_obs > 0 else np.nan
        std_val = float(vals.std()) if n_obs > 1 else np.nan
        median_val = float(vals.median()) if n_obs > 0 else np.nan
        min_val = float(vals.min()) if n_obs > 0 else np.nan
        max_val = float(vals.max()) if n_obs > 0 else np.nan
        q1_val = float(vals.quantile(0.25)) if n_obs > 0 else np.nan
        q3_val = float(vals.quantile(0.75)) if n_obs > 0 else np.nan
        iqr_val = q3_val - q1_val if n_obs > 0 else np.nan
        
        # Mode for discrete/low cardinality
        mode_val = np.nan
        if feat in ['contiguous_depressed_leads_count'] and n_obs > 0:
            m = vals.mode()
            if len(m) > 0:
                mode_val = float(m.iloc[0])
                
        comprehensive_stats.append({
            'Feature': feat,
            'Unit': get_feature_meta(feat),
            'Phase': p,
            'N_Valid': n_obs,
            'Mean': round(mean_val, 4),
            'SD': round(std_val, 4),
            'Median': round(median_val, 4),
            'Min': round(min_val, 4),
            'Max': round(max_val, 4),
            'Q1': round(q1_val, 4),
            'Q3': round(q3_val, 4),
            'IQR': round(iqr_val, 4),
            'Mode': mode_val
        })
        
    # Paired patient-level comparisons (patients with both phases present)
    piv_paired = piv.dropna(subset=['rest', 'stress', 'recover'])
    n_paired = len(piv_paired)
    
    if n_paired > 0:
        d_stress_rest = piv_paired['stress'] - piv_paired['rest']
        d_rec_stress = piv_paired['recover'] - piv_paired['stress']
        d_rec_rest = piv_paired['recover'] - piv_paired['rest']
        
        # Absolute differences
        for comp_name, d_series in [
            ('STRESS - REST', d_stress_rest),
            ('RECOVERY - STRESS', d_rec_stress),
            ('RECOVERY - REST', d_rec_rest)
        ]:
            paired_diff_rows.append({
                'Feature': feat,
                'Unit': get_feature_meta(feat),
                'Comparison': comp_name,
                'N_Paired': len(d_series),
                'Mean_Diff': round(float(d_series.mean()), 4),
                'SD_Diff': round(float(d_series.std()), 4),
                'Median_Diff': round(float(d_series.median()), 4),
                'Min_Diff': round(float(d_series.min()), 4),
                'Max_Diff': round(float(d_series.max()), 4),
                'Q1_Diff': round(float(d_series.quantile(0.25)), 4),
                'Q3_Diff': round(float(d_series.quantile(0.75)), 4),
                'IQR_Diff': round(float(d_series.quantile(0.75) - d_series.quantile(0.25)), 4)
            })
            
        # Percentage changes where mathematically meaningful
        def calc_pct(post, pre):
            denom = np.abs(pre)
            mask = denom > 1e-3
            pct = np.full_like(post, np.nan, dtype=float)
            pct[mask] = ((post[mask] - pre[mask]) / denom[mask]) * 100.0
            return pd.Series(pct[mask])

        pct_stress_rest = calc_pct(piv_paired['stress'].values, piv_paired['rest'].values)
        pct_rec_stress = calc_pct(piv_paired['recover'].values, piv_paired['stress'].values)
        pct_rec_rest = calc_pct(piv_paired['recover'].values, piv_paired['rest'].values)

        for comp_name, pct_series in [
            ('Pct_Change_STRESS_vs_REST', pct_stress_rest),
            ('Pct_Change_RECOVERY_vs_STRESS', pct_rec_stress),
            ('Pct_Change_RECOVERY_vs_REST', pct_rec_rest)
        ]:
            if len(pct_series) > 0 and np.isfinite(pct_series).any():
                percentage_change_rows.append({
                    'Feature': feat,
                    'Comparison': comp_name,
                    'N_Valid_Pct': int(np.isfinite(pct_series).sum()),
                    'Mean_Pct_Change': round(float(pct_series.mean()), 2),
                    'SD_Pct_Change': round(float(pct_series.std()), 2),
                    'Median_Pct_Change': round(float(pct_series.median()), 2),
                    'IQR_Pct_Change': round(float(pct_series.quantile(0.75) - pct_series.quantile(0.25)), 2),
                    'Min_Pct_Change': round(float(pct_series.min()), 2),
                    'Max_Pct_Change': round(float(pct_series.max()), 2)
                })

        # Paired Hypothesis Testing (t-test and Wilcoxon)
        for comp_name, pre_col, post_col in [
            ('STRESS vs REST', 'rest', 'stress'),
            ('RECOVERY vs STRESS', 'stress', 'recover'),
            ('RECOVERY vs REST', 'rest', 'recover')
        ]:
            x = piv_paired[pre_col].values
            y = piv_paired[post_col].values
            diff = y - x
            
            # Paired t-test
            t_stat, t_pval = stats.ttest_rel(y, x, nan_policy='omit')
            
            # Wilcoxon signed-rank test
            try:
                diff_nonzero = diff[diff != 0]
                if len(diff_nonzero) >= 5:
                    w_stat, w_pval = stats.wilcoxon(y, x)
                else:
                    w_stat, w_pval = np.nan, np.nan
            except Exception:
                w_stat, w_pval = np.nan, np.nan
                
            hypothesis_test_rows.append({
                'Feature': feat,
                'Comparison': comp_name,
                'N_Paired': n_paired,
                'Mean_Pre': round(float(np.mean(x)), 4),
                'Mean_Post': round(float(np.mean(y)), 4),
                'Mean_Difference': round(float(np.mean(diff)), 4),
                'Paired_T_Stat': round(float(t_stat), 4) if np.isfinite(t_stat) else np.nan,
                'Paired_T_PValue': float(t_pval) if np.isfinite(t_pval) else np.nan,
                'Wilcoxon_W_Stat': round(float(w_stat), 4) if np.isfinite(w_stat) else np.nan,
                'Wilcoxon_PValue': float(w_pval) if np.isfinite(w_pval) else np.nan
            })

# Convert to DataFrames
df_comp_stats = pd.DataFrame(comprehensive_stats)
df_paired_diff = pd.DataFrame(paired_diff_rows)
df_pct_change = pd.DataFrame(percentage_change_rows)
df_hypo = pd.DataFrame(hypothesis_test_rows)

# FDR Correction (Benjamini-Hochberg) for hypothesis tests
def fdr_correction(p_vals):
    p = np.array(p_vals, dtype=float)
    valid_idx = np.where(np.isfinite(p))[0]
    p_valid = p[valid_idx]
    n = len(p_valid)
    if n == 0:
        return p
    sorted_order = np.argsort(p_valid)
    sorted_p = p_valid[sorted_order]
    fdr = np.zeros(n)
    fdr[-1] = sorted_p[-1]
    for i in range(n - 2, -1, -1):
        fdr[i] = min(fdr[i + 1], sorted_p[i] * n / (i + 1))
    fdr_out = np.full_like(p, np.nan)
    fdr_out[valid_idx[sorted_order]] = fdr
    return fdr_out

df_hypo['Paired_T_FDR_PValue'] = np.nan
df_hypo['Wilcoxon_FDR_PValue'] = np.nan

for comp in ['STRESS vs REST', 'RECOVERY vs STRESS', 'RECOVERY vs REST']:
    idx = df_hypo[df_hypo['Comparison'] == comp].index
    df_hypo.loc[idx, 'Paired_T_FDR_PValue'] = fdr_correction(df_hypo.loc[idx, 'Paired_T_PValue'].values)
    df_hypo.loc[idx, 'Wilcoxon_FDR_PValue'] = fdr_correction(df_hypo.loc[idx, 'Wilcoxon_PValue'].values)

# Save CSVs
df_comp_stats.to_csv('COMPARISON/05_REST_STRESS_RECOVERY/rest_stress_recovery_comprehensive_stats.csv', index=False)
df_comp_stats.to_csv('COMPARISON/tables/rest_stress_recovery_comprehensive_stats.csv', index=False)

df_paired_diff.to_csv('COMPARISON/05_REST_STRESS_RECOVERY/rest_stress_recovery_paired_differences.csv', index=False)
df_paired_diff.to_csv('COMPARISON/tables/rest_stress_recovery_paired_differences.csv', index=False)

df_pct_change.to_csv('COMPARISON/05_REST_STRESS_RECOVERY/rest_stress_recovery_percentage_changes.csv', index=False)
df_pct_change.to_csv('COMPARISON/tables/rest_stress_recovery_percentage_changes.csv', index=False)

df_hypo.to_csv('COMPARISON/05_REST_STRESS_RECOVERY/rest_stress_recovery_hypothesis_tests.csv', index=False)
df_hypo.to_csv('COMPARISON/tables/rest_stress_recovery_hypothesis_tests.csv', index=False)

print("Saved all 4 comprehensive comparison tables!")

# --- PLOTTING SECTION ---
print("Generating figures 07 through 15...")

key_biomarkers = [
    ('min_st60_global_adc', 'Global Min ST60 (ADC counts)', 'ADC counts'),
    ('min_st60_inferior_adc', 'Inferior Min ST60 (ADC counts)', 'ADC counts'),
    ('gated_mean_hr_bpm', 'Heart Rate (BPM)', 'BPM'),
    ('gated_rmssd_ms', 'Vagal Tone RMSSD (ms)', 'ms'),
    ('II_st60_adc', 'Lead II ST60 (ADC counts)', 'ADC counts'),
    ('V5_st60_adc', 'Lead V5 ST60 (ADC counts)', 'ADC counts')
]

colors = ['#2b5c8f', '#d95f02', '#7570b3']

# --- Figure 07: Boxplots ---
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
axes = axes.flatten()

for idx, (feat, title, unit) in enumerate(key_biomarkers):
    data = [df_500[df_500['phase'] == p][feat].dropna().values for p in phases]
    box = axes[idx].boxplot(data, tick_labels=['REST', 'STRESS', 'RECOVERY'], patch_artist=True)
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
    axes[idx].set_title(title, fontsize=11, fontweight='bold')
    axes[idx].set_ylabel(f'Amplitude / Value ({unit})', fontsize=10)
    axes[idx].grid(axis='y', linestyle=':', alpha=0.6)

plt.suptitle('Figure 07: REST vs STRESS vs RECOVERY Primary Biomarker Boxplots (N=80 Patients)', fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
fig.savefig('COMPARISON/figures/fig07_rest_stress_recovery_boxplots.png', bbox_inches='tight')
fig.savefig('COMPARISON/figures/fig07_rest_stress_recovery_boxplots.pdf', bbox_inches='tight')
plt.close(fig)
print("Saved fig07")

# --- Figure 08: Violin Plots ---
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
axes = axes.flatten()

for idx, (feat, title, unit) in enumerate(key_biomarkers):
    data = [df_500[df_500['phase'] == p][feat].dropna().values for p in phases]
    if all(len(d) > 0 for d in data):
        parts = axes[idx].violinplot(data, positions=[1, 2, 3], showmeans=True, showmedians=True)
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(colors[i])
            pc.set_edgecolor('black')
            pc.set_alpha(0.7)
    axes[idx].set_xticks([1, 2, 3])
    axes[idx].set_xticklabels(['REST', 'STRESS', 'RECOVERY'], fontweight='bold')
    axes[idx].set_title(title, fontsize=11, fontweight='bold')
    axes[idx].set_ylabel(f'{unit}', fontsize=10)
    axes[idx].grid(axis='y', linestyle=':', alpha=0.6)

plt.suptitle('Figure 08: Kernel Density Violin Distributions Across Experimental Phases', fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
fig.savefig('COMPARISON/figures/fig08_rest_stress_recovery_violins.png', bbox_inches='tight')
fig.savefig('COMPARISON/figures/fig08_rest_stress_recovery_violins.pdf', bbox_inches='tight')
plt.close(fig)
print("Saved fig08")

# --- Figure 09: Mean with 95% Confidence Interval ---
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
axes = axes.flatten()

for idx, (feat, title, unit) in enumerate(key_biomarkers):
    means = []
    cis = []
    for p in phases:
        vals = df_500[df_500['phase'] == p][feat].dropna().values
        n = len(vals)
        if n > 1:
            m = np.mean(vals)
            se = np.std(vals, ddof=1) / np.sqrt(n)
            ci = 1.96 * se
        else:
            m, ci = np.nan, np.nan
        means.append(m)
        cis.append(ci)
    
    axes[idx].errorbar(['REST', 'STRESS', 'RECOVERY'], means, yerr=cis, fmt='-o',
                       color='#2b5c8f', ecolor='#d95f02', elinewidth=2, capsize=5, capthick=2, markersize=7)
    axes[idx].set_title(title, fontsize=11, fontweight='bold')
    axes[idx].set_ylabel(f'Mean ± 95% CI ({unit})', fontsize=10)
    axes[idx].grid(linestyle=':', alpha=0.6)

plt.suptitle('Figure 09: Parametric Phase Trajectories (Mean with 95% Confidence Interval)', fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
fig.savefig('COMPARISON/figures/fig09_rest_stress_recovery_mean_ci.png', bbox_inches='tight')
fig.savefig('COMPARISON/figures/fig09_rest_stress_recovery_mean_ci.pdf', bbox_inches='tight')
plt.close(fig)
print("Saved fig09")

# --- Figure 10: Median with Interquartile Range ---
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
axes = axes.flatten()

for idx, (feat, title, unit) in enumerate(key_biomarkers):
    medians = []
    yerr_lower = []
    yerr_upper = []
    for p in phases:
        vals = df_500[df_500['phase'] == p][feat].dropna().values
        if len(vals) > 0:
            med = np.median(vals)
            q1 = np.percentile(vals, 25)
            q3 = np.percentile(vals, 75)
            medians.append(med)
            yerr_lower.append(med - q1)
            yerr_upper.append(q3 - med)
        else:
            medians.append(np.nan)
            yerr_lower.append(0)
            yerr_upper.append(0)
        
    axes[idx].errorbar(['REST', 'STRESS', 'RECOVERY'], medians, yerr=[yerr_lower, yerr_upper],
                       fmt='-s', color='#7570b3', ecolor='#1b9e77', elinewidth=2, capsize=5, capthick=2, markersize=7)
    axes[idx].set_title(title, fontsize=11, fontweight='bold')
    axes[idx].set_ylabel(f'Median [Q1, Q3] ({unit})', fontsize=10)
    axes[idx].grid(linestyle=':', alpha=0.6)

plt.suptitle('Figure 10: Non-Parametric Phase Trajectories (Median with IQR)', fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
fig.savefig('COMPARISON/figures/fig10_rest_stress_recovery_median_comparison.png', bbox_inches='tight')
fig.savefig('COMPARISON/figures/fig10_rest_stress_recovery_median_comparison.pdf', bbox_inches='tight')
plt.close(fig)
print("Saved fig10")

# --- Figure 11: Individual Patient Trajectory Plots (Spaghetti Plots) ---
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
axes = axes.flatten()

cohort_df = pd.read_csv('data/manifests/cohort_manifest.csv')
pos_subjects = set(cohort_df[cohort_df['label_status'] == 'POSITIVE']['subject_id'])

for idx, (feat, title, unit) in enumerate(key_biomarkers):
    piv = df_500.pivot(index='subject_id', columns='phase', values=feat)[['rest', 'stress', 'recover']].dropna()
    
    # Plot negative subjects in semi-transparent grey
    for subj_id in piv.index:
        y_vals = piv.loc[subj_id].values
        if subj_id in pos_subjects:
            axes[idx].plot(['REST', 'STRESS', 'RECOVERY'], y_vals, color='#d95f02', alpha=0.9, linewidth=1.8, zorder=5)
        else:
            axes[idx].plot(['REST', 'STRESS', 'RECOVERY'], y_vals, color='#999999', alpha=0.35, linewidth=0.9, zorder=2)
            
    # Add mean trendlines
    neg_subjs = [s for s in piv.index if s not in pos_subjects]
    pos_subjs = [s for s in piv.index if s in pos_subjects]
    if len(neg_subjs) > 0:
        mean_neg = piv.loc[neg_subjs].mean().values
        axes[idx].plot(['REST', 'STRESS', 'RECOVERY'], mean_neg, color='#2b5c8f', linewidth=3.0, linestyle='--', label='Neg Mean', zorder=6)
    if len(pos_subjs) > 0:
        mean_pos = piv.loc[pos_subjs].mean().values
        axes[idx].plot(['REST', 'STRESS', 'RECOVERY'], mean_pos, color='#d95f02', linewidth=3.0, linestyle='-', label='Pos Mean', zorder=7)

    axes[idx].set_title(title, fontsize=11, fontweight='bold')
    axes[idx].set_ylabel(f'{unit}', fontsize=10)
    axes[idx].grid(linestyle=':', alpha=0.6)
    if idx == 0:
        axes[idx].legend(frameon=True, facecolor='white', edgecolor='gray', loc='upper right')

plt.suptitle('Figure 11: Individual Patient Trajectories (Spaghetti Plot: Orange=MSIMI+, Grey=MSIMI-)', fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
fig.savefig('COMPARISON/figures/fig11_individual_patient_trajectories.png', bbox_inches='tight')
fig.savefig('COMPARISON/figures/fig11_individual_patient_trajectories.pdf', bbox_inches='tight')
plt.close(fig)
print("Saved fig11")

# --- Figure 12: Paired Differences Distributions ---
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
axes = axes.flatten()

for idx, (feat, title, unit) in enumerate(key_biomarkers):
    piv = df_500.pivot(index='subject_id', columns='phase', values=feat)[['rest', 'stress', 'recover']].dropna()
    d_stress_rest = piv['stress'] - piv['rest']
    d_rec_stress = piv['recover'] - piv['stress']
    
    axes[idx].hist(d_stress_rest, bins=20, alpha=0.6, color='#d95f02', edgecolor='black', label='Δ(Stress - Rest)')
    axes[idx].hist(d_rec_stress, bins=20, alpha=0.5, color='#7570b3', edgecolor='black', label='Δ(Rec - Stress)')
    axes[idx].axvline(0, color='black', linestyle='--', linewidth=1.2)
    
    axes[idx].set_title(title, fontsize=11, fontweight='bold')
    axes[idx].set_xlabel(f'Paired Difference ({unit})', fontsize=10)
    axes[idx].set_ylabel('Patient Count', fontsize=10)
    axes[idx].grid(axis='y', linestyle=':', alpha=0.6)
    if idx == 0:
        axes[idx].legend(frameon=True, facecolor='white', edgecolor='gray')

plt.suptitle('Figure 12: Paired Within-Patient Differences Across Experimental Transitions', fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
fig.savefig('COMPARISON/figures/fig12_paired_differences_distributions.png', bbox_inches='tight')
fig.savefig('COMPARISON/figures/fig12_paired_differences_distributions.pdf', bbox_inches='tight')
plt.close(fig)
print("Saved fig12")

# --- Figure 13: Feature Transition Correlation Heatmap ---
piv_st = df_500.pivot(index='subject_id', columns='phase')
delta_st60_leads = pd.DataFrame(index=piv_st.index)
for l in leads:
    delta_st60_leads[f'{l}_ΔST60'] = piv_st[(f'{l}_st60_adc', 'stress')] - piv_st[(f'{l}_st60_adc', 'rest')]

corr_matrix = delta_st60_leads.corr()

fig, ax = plt.subplots(figsize=(10, 8.5))
cax = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
ax.set_xticks(range(len(corr_matrix.columns)))
ax.set_yticks(range(len(corr_matrix.columns)))
ax.set_xticklabels(corr_matrix.columns, rotation=45, ha='right', fontsize=9, fontweight='bold')
ax.set_yticklabels(corr_matrix.columns, fontsize=9, fontweight='bold')
cbar = fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('Pearson Correlation Coefficient (r)', fontsize=10, fontweight='bold')

for i in range(len(corr_matrix)):
    for j in range(len(corr_matrix)):
        val = corr_matrix.iloc[i, j]
        ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=7, color='white' if abs(val) > 0.5 else 'black')

ax.set_title('Figure 13: Inter-Lead ST60 Stress Response Correlation Matrix (Δ Stress - Rest)', fontsize=12, fontweight='bold')
plt.tight_layout()
fig.savefig('COMPARISON/figures/fig13_feature_transition_heatmaps.png', bbox_inches='tight')
fig.savefig('COMPARISON/figures/fig13_feature_transition_heatmaps.pdf', bbox_inches='tight')
plt.close(fig)
print("Saved fig13")

# --- Figure 14: Sampling Rate Comparison: 100 Hz vs 500 Hz ---
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

compare_metrics = [
    ('min_st60_global_adc', 'Global Min ST60', 'ADC counts'),
    ('gated_mean_hr_bpm', 'Heart Rate', 'BPM'),
    ('gated_rmssd_ms', 'RMSSD (Vagal Tone)', 'ms'),
    ('II_st60_adc', 'Lead II ST60', 'ADC counts')
]

for idx, (feat, title, unit) in enumerate(compare_metrics):
    piv_sr = df_rec[df_rec['phase'] == 'rest'].pivot(index='subject_id', columns='sampling_rate', values=feat).dropna()
    x = piv_sr[100].values
    y = piv_sr[500].values
    
    axes[idx].scatter(x, y, color='#2b5c8f', edgecolors='black', alpha=0.75, s=45)
    lims = [min(x.min(), y.min()), max(x.max(), y.max())]
    axes[idx].plot(lims, lims, 'r--', linewidth=1.5, label='Identity Line (y = x)')
    
    r_val, _ = stats.pearsonr(x, y)
    axes[idx].text(0.05, 0.90, f'r = {r_val:.4f}\nN = {len(x)} matched', transform=axes[idx].transAxes,
                   fontsize=10, fontweight='bold', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    axes[idx].set_title(f'{title} (Resting Baseline)', fontsize=11, fontweight='bold')
    axes[idx].set_xlabel(f'100 Hz Acquisition ({unit})', fontsize=10)
    axes[idx].set_ylabel(f'500 Hz Acquisition ({unit})', fontsize=10)
    axes[idx].grid(linestyle=':', alpha=0.6)
    axes[idx].legend(loc='lower right', frameon=True, facecolor='white')

plt.suptitle('Figure 14: Cross-Frequency Fidelity (100 Hz vs 500 Hz Matched Recordings)', fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
fig.savefig('COMPARISON/figures/fig14_sampling_rate_100hz_vs_500hz_comparison.png', bbox_inches='tight')
fig.savefig('COMPARISON/figures/fig14_sampling_rate_100hz_vs_500hz_comparison.pdf', bbox_inches='tight')
plt.close(fig)
print("Saved fig14")

# --- Figure 15: Statistical Significance Volcano Plot ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

for idx, comp in enumerate(['STRESS vs REST', 'RECOVERY vs STRESS']):
    sub_df = df_hypo[df_hypo['Comparison'] == comp].copy()
    sub_df['neg_log10_p'] = -np.log10(np.maximum(sub_df['Paired_T_FDR_PValue'], 1e-15))
    
    sig_mask = sub_df['Paired_T_FDR_PValue'] < 0.05
    axes[idx].scatter(sub_df.loc[~sig_mask, 'Mean_Difference'], sub_df.loc[~sig_mask, 'neg_log10_p'],
                      color='#999999', alpha=0.6, s=40, label='Not Significant (FDR >= 0.05)')
    axes[idx].scatter(sub_df.loc[sig_mask, 'Mean_Difference'], sub_df.loc[sig_mask, 'neg_log10_p'],
                      color='#d95f02', alpha=0.85, s=55, edgecolor='black', label='Significant (FDR < 0.05)')
    
    axes[idx].axhline(-np.log10(0.05), color='red', linestyle='--', linewidth=1.2, label='FDR = 0.05 threshold')
    axes[idx].axvline(0, color='black', linestyle=':', linewidth=1.0)
    
    top_sig = sub_df.sort_values('neg_log10_p', ascending=False).head(4)
    for _, row in top_sig.iterrows():
        axes[idx].annotate(row['Feature'][:15], (row['Mean_Difference'], row['neg_log10_p']),
                           textcoords="offset points", xytext=(0, 6), ha='center', fontsize=8, fontweight='bold')
        
    axes[idx].set_title(f'{comp} Hypothesis Testing Volcano Plot', fontsize=11, fontweight='bold')
    axes[idx].set_xlabel('Mean Paired Difference (Post - Pre)', fontsize=10)
    axes[idx].set_ylabel('-log10(FDR-adjusted p-value)', fontsize=10)
    axes[idx].grid(linestyle=':', alpha=0.6)
    axes[idx].legend(loc='upper right', frameon=True, facecolor='white', edgecolor='none')

plt.suptitle('Figure 15: Benjamini-Hochberg FDR Statistical Significance Volcano Plots', fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
fig.savefig('COMPARISON/figures/fig15_statistical_significance_volcano.png', bbox_inches='tight')
fig.savefig('COMPARISON/figures/fig15_statistical_significance_volcano.pdf', bbox_inches='tight')
plt.close(fig)
print("Saved fig15")

print("Stage 05 Comprehensive REST vs STRESS vs RECOVERY completed successfully!")
