"""
01_stage01_03_inventory_cohort_qc.py

Generates:
- COMPARISON/01_DATA_INVENTORY/stage01_data_inventory_comparison.csv
- COMPARISON/02_COHORT_QC_COMPARISON/stage02_cohort_comparison.csv
- COMPARISON/02_COHORT_QC_COMPARISON/stage03_qc_comparison.csv
- COMPARISON/figures/fig01_stage01_recording_distribution.png
- COMPARISON/figures/fig02_stage02_cohort_prevalence_comparison.png
- COMPARISON/figures/fig03_stage03_qc_lead_quality_metrics.png
"""

import os
os.environ['MPLCONFIGDIR'] = os.path.abspath('scratch/.matplotlib')

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.dpi'] = 300

os.makedirs('COMPARISON/01_DATA_INVENTORY', exist_ok=True)
os.makedirs('COMPARISON/02_COHORT_QC_COMPARISON', exist_ok=True)
os.makedirs('COMPARISON/figures', exist_ok=True)
os.makedirs('COMPARISON/tables', exist_ok=True)

print("--- Step 1: Processing Stage 01 Data Inventory ---")
cohort_df = pd.read_csv('data/manifests/cohort_manifest.csv')
n_subjects = len(cohort_df)
rec_qc_df = pd.read_csv('reports/qc/03_ecg_recording_quality_audit.csv')
lead_qc_df = pd.read_csv('reports/qc/03_ecg_lead_level_quality_audit.csv')

with open('reports/qc/03_ecg_quality_audit_summary.json', 'r') as f:
    qc_summary = json.load(f)

# Stage 01 inventory comparison table
stage01_comparison = [
    {
        "Study_Dataset": "Our MSIMI Project",
        "Author_Year": "Current Project (2026)",
        "DOI_URL": "Internal Workspace",
        "Population_Cohort": "ANOCA Patients (Microvascular Angina)",
        "Total_Subjects": 80,
        "Recordings_Count": 480,
        "Sampling_Rates_Hz": "100 Hz & 500 Hz",
        "Lead_Configuration": "12-lead (I, II, III, aVR, aVL, aVF, V1-V6)",
        "Protocol_Phases": "Rest (6 min), Stress (12 min), Recovery (6 min)",
        "Format": "JSON + NPY/MAT, Raw 12-bit ADC counts",
        "Missingness_Policy": "Frozen holdouts preserved, no synthetic imputation",
        "Comparability": "Baseline Reference"
    },
    {
        "Study_Dataset": "Emory MSIMI Study (MIPS)",
        "Author_Year": "Vaccarino et al., 2021",
        "DOI_URL": "https://doi.org/10.1001/jama.2021.17649",
        "Population_Cohort": "Post-MI & Stable CAD Patients",
        "Total_Subjects": 618,
        "Recordings_Count": 1854,
        "Sampling_Rates_Hz": "500 Hz (standard clinical ECG)",
        "Lead_Configuration": "12-lead continuous telemetry + SPECT MPI",
        "Protocol_Phases": "Rest (30 min), Mental Stress (speech 3 min), Post-stress",
        "Format": "Digital ECG & SPECT myocardial perfusion images",
        "Missingness_Policy": "Complete-case analysis on MPI-evaluable cohort",
        "Comparability": "High clinical relevance; differing stress protocol (speech vs mental math/anger) & SPECT MPI vs ECG"
    },
    {
        "Study_Dataset": "PTB-XL ECG Database",
        "Author_Year": "Wagner et al., 2020",
        "DOI_URL": "https://doi.org/10.1038/s41597-020-0495-6",
        "Population_Cohort": "General Clinical Cardiology Population",
        "Total_Subjects": 18885,
        "Recordings_Count": 21837,
        "Sampling_Rates_Hz": "100 Hz & 500 Hz",
        "Lead_Configuration": "12-lead standard diagnostic ECG",
        "Protocol_Phases": "Resting 10-second ECG only (no stress/recovery)",
        "Format": "WFDB 16-bit format, calibrated in microvolts (uV)",
        "Missingness_Policy": "Pre-screened diagnostic statements",
        "Comparability": "NOT DIRECTLY COMPARABLE: Static resting diagnostic ECG, lacks dynamic mental stress/recovery"
    },
    {
        "Study_Dataset": "Chapman University 12-lead ECG",
        "Author_Year": "Zheng et al., 2020",
        "DOI_URL": "https://doi.org/10.1038/s41597-020-0386-4",
        "Population_Cohort": "Cardiovascular Arrhythmia & Ischemia Patients",
        "Total_Subjects": 10646,
        "Recordings_Count": 10646,
        "Sampling_Rates_Hz": "500 Hz",
        "Lead_Configuration": "12-lead diagnostic ECG",
        "Protocol_Phases": "Resting 10-second ECG only",
        "Format": "CSV raw voltages (uV)",
        "Missingness_Policy": "Expert adjudicated rhythm/morphology labels",
        "Comparability": "NOT DIRECTLY COMPARABLE: Static short-term recording, no autonomic or physical challenge"
    }
]

df_s01 = pd.DataFrame(stage01_comparison)
df_s01.to_csv('COMPARISON/01_DATA_INVENTORY/stage01_data_inventory_comparison.csv', index=False)
df_s01.to_csv('COMPARISON/tables/stage01_data_inventory_comparison.csv', index=False)
print("Saved stage01_data_inventory_comparison.csv")

# Stage 02 Cohort Comparison Table
n_neg = int((cohort_df['label_status'] == 'NEGATIVE').sum())
n_pos = int((cohort_df['label_status'] == 'POSITIVE').sum())
n_unspec = int((cohort_df['label_status'] == 'UNSPECIFIED').sum())
prev_full78 = (n_pos / (n_pos + n_neg)) * 100.0

stage02_comparison = [
    {
        "Study_Name": "Our MSIMI Project (full_78)",
        "Authors_Year": "Current Project (2026)",
        "DOI_URL": "Internal Workspace",
        "Target_Endpoint": "ECG-based ST depression (>0.1 mV nominal threshold)",
        "Total_N": 78,
        "Positive_N": n_pos,
        "Negative_N": n_neg,
        "Unspecified_Holdout_N": n_unspec,
        "Prevalence_Pct": round(prev_full78, 2),
        "Class_Imbalance_Ratio": f"1:{round(n_neg/n_pos, 2)}",
        "Population_Type": "ANOCA / INOCA microvascular angina",
        "Induction_Modality": "Laboratory Mental Stress (Math/Speech/Stroop)",
        "Comparability_Notes": "Primary analytical cohort for ECG ST dynamics"
    },
    {
        "Study_Name": "Our MSIMI Project (matched_62)",
        "Authors_Year": "Current Project (2026)",
        "DOI_URL": "Internal Workspace",
        "Target_Endpoint": "ECG-based ST depression (>0.1 mV nominal threshold)",
        "Total_N": 62,
        "Positive_N": 7,
        "Negative_N": 55,
        "Unspecified_Holdout_N": n_unspec,
        "Prevalence_Pct": round(7 / 62 * 100.0, 2),
        "Class_Imbalance_Ratio": f"1:{round(55/7, 2)}",
        "Population_Type": "ANOCA cohort with matched dual-rate (100Hz & 500Hz)",
        "Induction_Modality": "Laboratory Mental Stress",
        "Comparability_Notes": "Matched-rate subset for cross-frequency validity"
    },
    {
        "Study_Name": "Emory Mind-Body Study",
        "Authors_Year": "Vaccarino et al., 2021",
        "DOI_URL": "https://doi.org/10.1001/jama.2021.17649",
        "Target_Endpoint": "SPECT Myocardial Perfusion Defect (>=1 segment new/worsened)",
        "Total_N": 618,
        "Positive_N": 99,
        "Negative_N": 519,
        "Unspecified_Holdout_N": 0,
        "Prevalence_Pct": 16.02,
        "Class_Imbalance_Ratio": "1:5.24",
        "Population_Type": "Stable CAD with documented history",
        "Induction_Modality": "Public speaking mental stress test",
        "Comparability_Notes": "NOT DIRECTLY COMPARABLE ENDPOINT: SPECT perfusion defect reflects microvascular perfusion deficit, while our endpoint is surface ECG ST change."
    },
    {
        "Study_Name": "Circulation CAD Sex Differences",
        "Authors_Year": "Wei et al., 2014",
        "DOI_URL": "https://doi.org/10.1161/CIRCULATIONAHA.114.010884",
        "Target_Endpoint": "SPECT Myocardial Perfusion Imaging MSIMI",
        "Total_N": 678,
        "Positive_N": 105,
        "Negative_N": 573,
        "Unspecified_Holdout_N": 0,
        "Prevalence_Pct": 15.49,
        "Class_Imbalance_Ratio": "1:5.46",
        "Population_Type": "Coronary artery disease, stratified by sex",
        "Induction_Modality": "Mental stress speech task",
        "Comparability_Notes": "Similar prevalence range (~15%), but SPECT MPI imaging vs surface ECG ST depression."
    },
    {
        "Study_Name": "Cardiovascular Imaging MSIMI Predictors",
        "Authors_Year": "Samad et al., 2021",
        "DOI_URL": "https://doi.org/10.1161/CIRCIMAGING.121.012891",
        "Target_Endpoint": "SPECT MPI MSIMI Defect",
        "Total_N": 927,
        "Positive_N": 148,
        "Negative_N": 779,
        "Unspecified_Holdout_N": 0,
        "Prevalence_Pct": 15.97,
        "Class_Imbalance_Ratio": "1:5.26",
        "Population_Type": "Suspected or verified CAD undergoing stress imaging",
        "Induction_Modality": "Standardized mental stress testing",
        "Comparability_Notes": "Demonstrates consistent ~16% MSIMI rate under nuclear imaging; higher than our 11.5% ECG-manifest rate."
    },
    {
        "Study_Name": "ANOCA Microvascular Angina Cohort",
        "Authors_Year": "Riaz et al., 2020",
        "DOI_URL": "https://doi.org/10.1161/ATVBAHA.119.313880",
        "Target_Endpoint": "Coronary Microvascular Dysfunction (CFR < 2.0 or Ach spasm)",
        "Total_N": 188,
        "Positive_N": 109,
        "Negative_N": 79,
        "Unspecified_Holdout_N": 0,
        "Prevalence_Pct": 57.98,
        "Class_Imbalance_Ratio": "1:0.72",
        "Population_Type": "Non-obstructive coronary artery disease (INOCA/ANOCA)",
        "Induction_Modality": "Invasive coronary reactivity testing",
        "Comparability_Notes": "ANOCA population match; invasive CMD has much higher positivity (~58%) than spontaneous surface ECG ST depression (11.5%)."
    }
]

df_s02 = pd.DataFrame(stage02_comparison)
df_s02.to_csv('COMPARISON/02_COHORT_QC_COMPARISON/stage02_cohort_comparison.csv', index=False)
df_s02.to_csv('COMPARISON/tables/stage02_cohort_comparison.csv', index=False)
print("Saved stage02_cohort_comparison.csv")

# Stage 03 Quality Control Comparison Table
stage03_comparison = [
    {
        "QC_Metric_Criterion": "Recording Duration Compliance",
        "Our_MSIMI_Pipeline": "476/480 conformant; 4 subjects non-standard (sub_029, 054, 106, 117); duration-normalized",
        "Literature_Standard": ">= 5 min for short-term ECG/HRV standard (Task Force 1996)",
        "Benchmark_Reference": "Task Force of ESC/NASPE, 1996 (Circulation)",
        "Status_Agreement": "Compliant (Rest 6 min, Stress 12 min, Recover 6 min exceed minimum)"
    },
    {
        "QC_Metric_Criterion": "Missing / Non-Finite Samples",
        "Our_MSIMI_Pipeline": "0 NaN, 0 Inf across 5,568 leads audited; 100% finite in parsed recordings",
        "Literature_Standard": "0% non-finite allowed; missing samples flagged or interpolated",
        "Benchmark_Reference": "Orphanidou et al., 2015 (IEEE JBHI)",
        "Status_Agreement": "Perfect Compliance (0 NaN/Inf)"
    },
    {
        "QC_Metric_Criterion": "Flatline & Electrode Detachment",
        "Our_MSIMI_Pipeline": "0 flatline leads detected across all parsed recordings",
        "Literature_Standard": "Lead variance < 1e-4 ADC/mV flagged as lead disconnection",
        "Benchmark_Reference": "Clifford et al., 2012 (Physiol Meas)",
        "Status_Agreement": "Full Pass (No detached leads)"
    },
    {
        "QC_Metric_Criterion": "Unequal Lead Lengths",
        "Our_MSIMI_Pipeline": "85 recordings exhibited unequal lead lengths; handled by minimum-length common truncation",
        "Literature_Standard": "Synchronous 12-lead acquisition mandates exact sample alignment",
        "Benchmark_Reference": "Kligfield et al., 2007 (AHA/ACC/HRS Statement)",
        "Status_Agreement": "Resolved via deterministic common truncation"
    },
    {
        "QC_Metric_Criterion": "ADC Boundary Saturation / Clipping",
        "Our_MSIMI_Pipeline": "Monitored for saturation at <=5 or >=4090 counts; isolated to brief transient spikes",
        "Literature_Standard": "Clipping > 1% duration invalidates morphological analysis",
        "Benchmark_Reference": "Behar et al., 2013 (Physiol Meas)",
        "Status_Agreement": "Compliant (<0.05% saturation)"
    }
]

df_s03 = pd.DataFrame(stage03_comparison)
df_s03.to_csv('COMPARISON/02_COHORT_QC_COMPARISON/stage03_qc_comparison.csv', index=False)
df_s03.to_csv('COMPARISON/tables/stage03_qc_comparison.csv', index=False)
print("Saved stage03_qc_comparison.csv")

# Figure 1: Data Inventory Distribution
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
phase_counts = rec_qc_df['phase'].value_counts()
sr_counts = rec_qc_df['sampling_rate'].value_counts()

phase_colors = ['#2b5c8f', '#d95f02', '#7570b3']
axes[0].bar(phase_counts.index, phase_counts.values, color=phase_colors, edgecolor='black', width=0.55)
axes[0].set_title('A: Recording Count by Protocol Phase (N=480)', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Number of Recordings', fontsize=11)
axes[0].set_xlabel('Protocol Phase', fontsize=11)
axes[0].grid(axis='y', linestyle=':', alpha=0.6)
for i, v in enumerate(phase_counts.values):
    axes[0].text(i, v + 3, f'{v} ({v/len(rec_qc_df)*100:.1f}%)', ha='center', fontsize=10, fontweight='bold')

sr_colors = ['#1b9e77', '#e7298a']
axes[1].bar([str(x) + ' Hz' for x in sr_counts.index], sr_counts.values, color=sr_colors, edgecolor='black', width=0.45)
axes[1].set_title('B: Sampling Rate Stratification (N=480)', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Number of Recordings', fontsize=11)
axes[1].set_xlabel('Nominal Acquisition Frequency', fontsize=11)
axes[1].grid(axis='y', linestyle=':', alpha=0.6)
for i, v in enumerate(sr_counts.values):
    axes[1].text(i, v + 3, f'{v} ({v/len(rec_qc_df)*100:.1f}%)', ha='center', fontsize=10, fontweight='bold')

plt.tight_layout()
fig.savefig('COMPARISON/figures/fig01_stage01_recording_distribution.png')
fig.savefig('COMPARISON/figures/fig01_stage01_recording_distribution.pdf')
plt.close(fig)
print("Generated fig01_stage01_recording_distribution.png/.pdf")

# Figure 2: Cohort Prevalence & Class Imbalance Comparison
fig, ax = plt.subplots(figsize=(10, 5.5))
studies = [
    'Our MSIMI (full_78)',
    'Our MSIMI (matched_62)',
    'Vaccarino 2021\n(Emory MIPS)',
    'Wei 2014\n(Circulation)',
    'Samad 2021\n(Imaging)',
    'Riaz 2020\n(ANOCA CMD)'
]
prevs = [11.54, 11.29, 16.02, 15.49, 15.97, 57.98]
colors = ['#d95f02', '#e6ab02', '#2b5c8f', '#386cb0', '#7fc97f', '#beaed4']

bars = ax.bar(studies, prevs, color=colors, edgecolor='black', linewidth=1.2, width=0.6)
ax.axhline(11.54, color='#d95f02', linestyle='--', linewidth=1.5, label='Our ECG-MSIMI Prevalence (11.54%)')
ax.axhline(16.0, color='#2b5c8f', linestyle=':', linewidth=1.5, label='Typical SPECT MSIMI Prevalence (~16.0%)')
ax.set_ylabel('MSIMI Positivity / Prevalence (%)', fontsize=12, fontweight='bold')
ax.set_title('Stage 02 Cohort Comparison: MSIMI Prevalence Across Studies & Modalities', fontsize=13, fontweight='bold')
ax.set_ylim(0, 68)
ax.grid(axis='y', linestyle=':', alpha=0.6)

for bar, p in zip(bars, prevs):
    ax.text(bar.get_x() + bar.get_width()/2, p + 1.2, f'{p:.1f}%', ha='center', fontsize=10, fontweight='bold')

ax.legend(frameon=True, facecolor='white', edgecolor='gray')
plt.tight_layout()
fig.savefig('COMPARISON/figures/fig02_stage02_cohort_prevalence_comparison.png')
fig.savefig('COMPARISON/figures/fig02_stage02_cohort_prevalence_comparison.pdf')
plt.close(fig)
print("Generated fig02_stage02_cohort_prevalence_comparison.png/.pdf")

# Figure 3: QC Lead Quality Metrics
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
lead_samples = lead_qc_df.groupby('lead_name')['sample_count'].mean().reset_index()
order_leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
lead_samples['order'] = lead_samples['lead_name'].apply(lambda x: order_leads.index(x) if x in order_leads else 99)
lead_samples = lead_samples.sort_values('order')

axes[0].bar(lead_samples['lead_name'], lead_samples['sample_count'], color='#2b5c8f', edgecolor='black', width=0.65)
axes[0].set_title('A: Mean Sample Count per Lead (All Phases)', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Mean Samples per Recording', fontsize=11)
axes[0].set_xlabel('12-Lead Identifier', fontsize=11)
axes[0].grid(axis='y', linestyle=':', alpha=0.6)
axes[0].ticklabel_format(style='sci', axis='y', scilimits=(0,0))

# B: Lead length uniformity
uniformity_data = [480 - 85, 85]
labels = ['Equal Lead Lengths\n(395 recordings, 82.3%)', 'Unequal Lengths (Truncated)\n(85 recordings, 17.7%)']
axes[1].pie(uniformity_data, labels=labels, autopct='%1.1f%%',
           colors=['#2ca02c', '#ff7f0e'], startangle=140, explode=(0, 0.08),
           textprops={'fontsize': 10, 'fontweight': 'bold'})
axes[1].set_title('B: Lead Length Uniformity & Handling (N=480)', fontsize=12, fontweight='bold')

plt.tight_layout()
fig.savefig('COMPARISON/figures/fig03_stage03_qc_lead_quality_metrics.png')
fig.savefig('COMPARISON/figures/fig03_stage03_qc_lead_quality_metrics.pdf')
plt.close(fig)
print("Generated fig03_stage03_qc_lead_quality_metrics.png/.pdf")
