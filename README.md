# MSIMI ECG Analysis

## 1. Project Overview

This repository contains an ECG analysis pipeline developed to investigate **Mental Stress-Induced Myocardial Ischemia (MSIMI)** in people with **Angina with Non-Obstructive Coronary Arteries (ANOCA)**.

The workflow analyses standard 12-lead ECG recordings collected during three phases: **Rest, Stress, and Recovery**. It includes dataset exploration, cohort and label verification, signal quality control, ECG preprocessing, R-peak detection, RR interval analysis, synchronized beat-window segmentation, feature extraction, and patient-level validation workflows.

This is an academic research project. It is not a clinically validated diagnostic system.

## 2. Project Objectives

- Inspect and organize the ECG dataset while preserving the original recordings.
- Verify the selected cohort and retain the original diagnostic labels.
- Assess signal quality at recording and lead level.
- Preprocess ECG signals using the configured pipeline.
- Detect R-peaks and calculate RR intervals and heart-rate-related measurements.
- Extract synchronized beat windows across all 12 ECG leads.
- Compare ECG morphology across Rest, Stress, and Recovery.
- Prepare patient-level data for statistical and machine-learning analysis.

## 3. Dataset Description

The selected analysis cohort contains **80 ANOCA subjects** (`sub_001` to `sub_080`).

| Property | Description |
|---|---|
| Subjects | 80 |
| ECG configuration | Standard 12-lead ECG |
| Leads | I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6 |
| Sampling frequencies | 100 Hz and 500 Hz |
| Recording phases | Rest, Stress, Recovery |
| Expected ECG files | 480 (80 subjects × 3 phases × 2 sampling rates) |
| Data formats | ECG CSV files and JSON metadata |

### 3.1 Recording protocol

| Phase | Expected duration |
|---|---:|
| Rest | 360 seconds (6 minutes) |
| Stress | 720 seconds (12 minutes) |
| Recovery | 360 seconds (6 minutes) |

### 3.2 MSIMI label distribution

The verified label distribution in the selected cohort is:

| Label status | Number of subjects |
|---|---:|
| MSIMI-positive | 9 |
| MSIMI-negative | 69 |
| Unspecified | 2 |
| **Total** | **80** |

Unspecified labels are retained as unknown/missing values. They must not be automatically converted to negative labels. The labelled cohort is imbalanced, so class imbalance must be considered during model development and evaluation.

## 4. Processing Workflow

Run the notebooks in sequence because later stages depend on outputs from earlier stages.

| Stage | Notebook / module | Purpose |
|---|---|---|
| 01 | `01_dataset_exploration.ipynb` | Explore dataset structure, file availability, recording phases, sampling rates, and lead configuration. |
| 02 | `02_cohort_and_label_verification.ipynb` | Verify the selected cohort and extract/check MSIMI labels from source metadata. |
| 03 | `03_ecg_quality_control.ipynb` | Audit lead count, signal lengths, recording duration, NaN/Inf values, flatlines, and potential clipping. |
| 04 | `04_ecg_preprocessing.ipynb` | Apply the configured ECG filtering and preprocessing steps. |
| 05 | `05_rpeak_detection.ipynb` | Detect R-peaks using the project's Pan–Tompkins-based QRS detection implementation and evaluate candidate rhythm leads. |
| 06 | `06_rr_hrv_analysis.ipynb` | Calculate RR intervals and heart-rate/HRV-related measurements with quality flags. |
| 07 | `07_beat_window_segmentation.ipynb` | Extract synchronized 12-lead beat windows, check boundaries, analyse overlap, and generate morphology plots. |
| 07a | `07a_inter_phase_transition_analysis.ipynb` | Analyse transitions between recording phases. |
| 08 | `08_ecg_feature_extraction.ipynb` | Generate ECG feature tables for downstream analysis. |
| 09 | `09_patient_level_validation.ipynb` | Prepare and evaluate patient-level validation workflows. |
| Additional | `newmodel_feature_selection.ipynb` | Experimental feature-selection and model-development workflow. |

Supplementary scripts and comparative analyses are also present in the repository. Check their input paths, output paths, and dependencies before running them.

## 5. Key Methodology

### 5.1 ECG quality control

The quality-control stage audits:

- Availability of the expected 12 leads.
- Unequal lead lengths.
- Actual versus expected recording duration.
- NaN and infinite values.
- Flatline signals.
- Potential low- and high-end ADC clipping.

In the reported audit, **480 ECG files were present**, **464 were successfully parsed**, and **16 produced parsing errors**. Among the 464 successfully parsed recordings, **85 had unequal lead lengths**.

These counts describe different checks: a file can exist on disk but still fail to parse.

The QC stage is a non-destructive audit. It does not itself filter, pad, truncate, or otherwise modify the raw ECG signals.

### 5.2 R-peak detection and rhythm-lead selection

The project uses a Pan–Tompkins-based QRS detection approach. Candidate rhythm leads are evaluated using:

- Detection status.
- Number of detected R-peaks.
- Mean heart rate.
- Percentage of physiologically implausible RR intervals.

The implemented rhythm-lead selection priority is:

1. **Lead II** — primary candidate.
2. **Lead V5** — first fallback.
3. **Lead I** — final fallback.

A candidate is selected only if it passes the configured quality checks. If none of these candidates passes, the recording is marked as having no acceptable rhythm lead. The other leads remain part of the synchronized 12-lead beat extraction; this hierarchy is specifically for selecting the rhythm lead used for R-peak/RR analysis.

### 5.3 RR interval calculation

For two consecutive R-peaks at sample indices \(r_i\) and \(r_{i+1}\), the RR interval is calculated as:

\[
RR_i\;(\mathrm{ms}) =
\frac{r_{i+1}-r_i}{f_s}\times 1000
\]

where:

- \(r_i\) and \(r_{i+1}\) are consecutive R-peak sample indices.
- \(f_s\) is the sampling frequency in samples per second.
- Multiplication by 1000 converts seconds to milliseconds.

Heart rate can be derived from the RR interval:

\[
HR\;(\mathrm{BPM}) = \frac{60\,000}{RR_i\;(\mathrm{ms})}
\]

Algorithmically derived RR intervals should not automatically be described as clinically adjudicated normal-to-normal (NN) intervals.

### 5.4 Beat-window segmentation

The current segmentation configuration uses a fixed **800 ms window** around each R-peak.

| Window component | Duration | Relative time |
|---|---:|---:|
| Pre-R | 250 ms | −250 ms to 0 ms |
| Post-R | 550 ms | 0 ms to +550 ms |
| **Total window** | **800 ms** | **−250 ms to +550 ms** |

The R-peak is the alignment point at **0 ms**.

The number of samples depends on the sampling frequency:

| Sampling rate | Pre-R samples | Post-R samples | Total samples per lead |
|---|---:|---:|---:|
| 100 Hz | 25 | 55 | 80 |
| 500 Hz | 125 | 275 | 400 |

Each accepted beat is extracted synchronously across all 12 leads. Boundary checks ensure that the requested window fits within the available signal. The checks also account for recordings where one lead, such as V6, is shorter than the others.

### 5.5 Window-overlap analysis

Two consecutive 800 ms beat windows overlap when the time between their R-peaks is shorter than 800 ms:

\[
RR_i < 800\ \mathrm{ms}
\]

Overlap is a consequence of the fixed window length and the observed RR intervals. It does not, by itself, indicate failed R-peak detection. However, overlapping windows are statistically dependent.

For downstream model evaluation, all recordings and beats belonging to one subject should remain in the same data partition. Randomly splitting individual beats across training and test sets can cause patient-level data leakage and overly optimistic performance estimates.

### 5.6 Median beat morphology

For diagnostic visualization, beat waveforms are grouped by phase and summarized using a sample-wise median waveform for each lead. This allows Rest, Stress, and Recovery morphology to be compared on a common R-peak-aligned time axis.

Representative output:

`outputs/figures/07_representative_12lead_beat_morphology.png`

## 6. Repository Structure

The following tree is a logical overview of the project. Some folders and generated files may appear only after their corresponding stages have been run. Adjust the tree to match the exact contents of the checked-out repository.

```text
MSIMI/
├── README.md
├── requirements.txt
├── .gitignore
├── run_sample_validation.py
├── configs/
│   ├── dataset_config.py
│   └── preprocessing.yaml
├── src/
│   ├── loader.py
│   ├── manifest.py
│   ├── qc.py
│   ├── preprocessing/
│   ├── peak_detection/
│   ├── evaluation/
│   ├── features/
│   └── utils/
├── notebooks/
│   ├── 01_dataset_exploration.ipynb
│   ├── 02_cohort_and_label_verification.ipynb
│   ├── 03_ecg_quality_control.ipynb
│   ├── 04_ecg_preprocessing.ipynb
│   ├── 05_rpeak_detection.ipynb
│   ├── 06_rr_hrv_analysis.ipynb
│   ├── 07_beat_window_segmentation.ipynb
│   ├── 07a_inter_phase_transition_analysis.ipynb
│   ├── 08_ecg_feature_extraction.ipynb
│   ├── 09_patient_level_validation.ipynb
│   └── newmodel_feature_selection.ipynb
├── data/
│   ├── cohort_manifest.csv
│   ├── cohort_manifest.json
│   ├── features/
│   └── interim/
│       └── preprocessed_ecg/
├── reports/
│   ├── qc/
│   ├── preprocessing/
│   ├── rpeaks/
│   ├── hrv/
│   ├── transitions/
│   └── validation/
├── outputs/
│   └── figures/
└── COMPARISON/
    ├── code/
    ├── figures/
    ├── 01_DATA_INVENTORY/
    ├── 02_COHORT_QC_COMPARISON/
    ├── 03_PREPROCESSING_COMPARISON/
    ├── 04_RPEAK_HRV_COMPARISON/
    ├── 05_REST_STRESS_RECOVERY/
    ├── 06_TRANSITION_COMPARISON/
    └── 07_FEATURE_COMPARISON/
```

## 7. Installation

### 7.1 Prerequisites

- Python 3.10 or later, subject to the version used by the project environment.
- Git.
- Jupyter Notebook or JupyterLab.
- Access to the original dataset.
- Sufficient disk space for raw and intermediate ECG data.

### 7.2 Windows (PowerShell)

```powershell
cd path\to\MSIMI

python -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks virtual-environment activation, use Command Prompt:

```bat
.venv\Scripts\activate.bat
```

### 7.3 Linux / macOS

```bash
cd path/to/MSIMI

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 8. Dataset Configuration

The original dataset is not assumed to be included in this Git repository.

Before running the notebooks:

1. Obtain access to the dataset from its official source.
2. Store the raw data in the expected local data directory.
3. Update dataset and project paths in `configs/dataset_config.py` or the path configuration actually used by the notebooks.
4. Review `configs/preprocessing.yaml` for preprocessing and detector parameters.
5. Confirm that subject IDs, file naming conventions, sampling rates, and lead names match the local dataset.

Do not commit restricted, private, or licensed patient data to a public repository. Follow the dataset's access and sharing conditions.

## 9. Running the Project

Start Jupyter from the repository root:

```bash
jupyter lab
```

Alternatively:

```bash
jupyter notebook
```

Run the notebooks in workflow order:

1. `01_dataset_exploration.ipynb`
2. `02_cohort_and_label_verification.ipynb`
3. `03_ecg_quality_control.ipynb`
4. `04_ecg_preprocessing.ipynb`
5. `05_rpeak_detection.ipynb`
6. `06_rr_hrv_analysis.ipynb`
7. `07_beat_window_segmentation.ipynb`
8. `07a_inter_phase_transition_analysis.ipynb` (if required)
9. `08_ecg_feature_extraction.ipynb`
10. `09_patient_level_validation.ipynb`
11. `newmodel_feature_selection.ipynb` (experimental workflow)

Run the sample validation script only after confirming its configured paths and dependencies:

```bash
python run_sample_validation.py
```

The scripts in `COMPARISON/code/` are supplementary. Inspect their input and output paths before execution.

## 10. Reproducibility and Data Integrity

- Preserve the original raw ECG recordings.
- Keep subject IDs and diagnostic labels linked to their source metadata.
- Retain unspecified labels as unknown.
- Record sampling frequency and recording phase for every processed signal.
- Preserve recording-level and lead-level QC outputs.
- Do not silently pad, truncate, or fabricate ECG samples.
- Keep all recordings and beats from one subject in the same data partition.
- Fit imputation, scaling, feature selection, and threshold-tuning steps using training data only within each validation split.
- Record software versions, configuration values, random seeds, and output paths for final experiments.

## 11. Limitations

1. **Small positive cohort:** Only 9 subjects in the selected cohort have a positive MSIMI label, so model estimates may have substantial uncertainty.
2. **Class imbalance:** Positive and negative labels are not evenly represented. Accuracy alone is not sufficient for evaluation.
3. **Unspecified labels:** Two subjects have unspecified labels and must not be treated as labelled negative cases.
4. **Recording irregularities:** Some recordings have parsing errors, shortened durations, or unequal lead lengths.
5. **Automated R-peaks:** Automated detection and RR screening are not equivalent to expert clinical annotation.
6. **Overlapping beat windows:** The fixed 800 ms segmentation creates overlap for many consecutive beats. This dependence must be considered in statistical analysis and model validation.
7. **Research scope:** ECG-derived features and model outputs require independent validation before any clinical interpretation or deployment.

## 12. Dataset Reference

Peng et al. (2024). **A multimodal physiological and psychological dataset for human with mental stress induced myocardial ischemia.** *Scientific Data*, 11, 704.

https://doi.org/10.1038/s41597-024-03462-2

Consult the original publication and dataset terms for the complete citation and data-use conditions.

## 13. Disclaimer

This repository is intended for academic and research purposes only. It is not a medical device, does not provide a clinical diagnosis, and must not be used to make patient-care decisions. Clinical interpretation must be performed by qualified healthcare professionals.
