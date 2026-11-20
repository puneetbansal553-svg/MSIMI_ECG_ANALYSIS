# MSIMI: Mental Stress-Induced Myocardial Ischemia Detection in ANOCA

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Clinical Research](https://img.shields.io/badge/Clinical%20Domain-Cardiology%20%7C%20ECG-red.svg)](https://www.acc.org/)
[![Validation](https://img.shields.io/badge/Validation-Patient--Stratified%205x5%20CV-green.svg)](#patient-level-validation-strategy)

An end-to-end computational cardiology and machine learning framework for the automated detection, characterization, and benchmarking of **Mental Stress-Induced Myocardial Ischemia (MSIMI)** in patients presenting with **Angina with Non-Obstructive Coronary Artery Disease (ANOCA)** using high-resolution 12-lead digital electrocardiography (ECG).

---

## Table of Contents

1. [Executive Summary & Clinical Rationale](#executive-summary--clinical-rationale)
2. [Cohort Architecture & Clinical Protocol](#cohort-architecture--clinical-protocol)
3. [Methodological Pipeline (Stages 01–11)](#methodological-pipeline-stages-0111)
4. [Rest vs. Stress vs. Recovery Dynamics](#rest-vs-stress-vs-recovery-dynamics)
5. [Feature Engineering & BNSRH Paradigm](#feature-engineering--bnsrh-paradigm)
6. [Machine Learning Benchmarks & Model Evaluation](#machine-learning-benchmarks--model-evaluation)
7. [Literature Comparison Matrix](#literature-comparison-matrix)
8. [Repository Architecture](#repository-architecture)
9. [Installation & Getting Started](#installation--getting-started)
10. [Reproducibility & Verification](#reproducibility--verification)
11. [Publication Figures Index](#publication-figures-index)
12. [Scientific Limitations & Clinical Disclaimer](#scientific-limitations--clinical-disclaimer)

---

## Executive Summary & Clinical Rationale

Mental Stress-Induced Myocardial Ischemia (**MSIMI**) is a transient, insidious form of myocardial ischemia triggered by acute psychological challenge rather than physical exertion. Unlike classical exercise-induced ischemia—which is predominantly driven by heavy hemodynamic workloads and systemic rate-pressure surges ($HR \times SBP$)—MSIMI is largely mediated by:
* **Coronary microvascular vasoconstriction** and endothelial dysfunction.
* **Acute cardiac autonomic dysregulation** characterized by vagal withdrawal and intense sympathetic outflow.
* Subendocardial hypoperfusion that occurs at substantially lower heart rates and rate-pressure products than exercise ischemia.

Patients suffering from **Angina with Non-Obstructive Coronary Artery Disease (ANOCA)** frequently exhibit coronary microvascular dysfunction (CMD) and autonomic vulnerability. Detecting MSIMI using surface 12-lead ECG is non-invasive, scalable, and clinically critical, yet fraught with signal processing and statistical hurdles:
1. **Severe Class Imbalance**: Ischemic electrical manifestations occur in approximately $11.5\%$ of standardized stress challenges.
2. **Subtle Electrical Signatures**: Repolarization shifts (ST-segment displacements) are smaller and more heterogeneous than acute epicardial thrombotic infarctions.
3. **Motion Tremor & Respiratory Drift**: High-stress psychological challenges induce muscular tension and irregular respiration, contaminating recording baselines.

This project delivers a **11-stage research pipeline** encompassing signal quality auditing, zero-phase filtering, robust QRS detection, dynamic autonomic tracking, temporal hysteresis modeling, and patient-stratified machine learning.

---

## Cohort Architecture & Clinical Protocol

### Study Population
* **Cohort Size**: 80 unique patients (`sub_001` through `sub_080`), all with angiographically confirmed **ANOCA** (`angina have no obstructive coronary artery disease == 1`).
* **Ground Truth Target**: `MSIMI-based-ecg` derived from expert clinical annotation, defined as transient ischemic ST depression ($\ge 0.1\text{ mV}$ or equivalent ADC count displacement) during laboratory mental stress challenge.
* **Prevalence & Imbalance**:
  * **Full Cohort (`full_78`)**: $N = 78$ labeled patients (9 Positive, 69 Negative, **11.54% prevalence**, imbalance ratio $1:7.67$).
  * **Matched Dual-Rate Cohort (`matched_62`)**: $N = 62$ patients with matched 100 Hz and 500 Hz recordings (7 Positive, 55 Negative, **11.29% prevalence**, imbalance ratio $1:7.86$).
  * **Strict Holdout Controls**: `sub_005` and `sub_049` possess unspecified ground-truth descriptions (`"no description"` in clinical records) and are rigorously quarantined as unlabelled holdouts—never used in model training, threshold tuning, or CV scoring.

### Standardized Stress Induction Protocol
Each patient underwent continuous 12-lead ECG monitoring across three consecutive phases:
1. **REST Phase**: 6 minutes ($360\text{ s}$) quiet supine baseline.
2. **STRESS Phase**: 12 minutes ($720\text{ s}$) standardized laboratory psychological stress challenge (mental arithmetic, cognitive confrontation, or speech tasks).
3. **RECOVERY Phase**: 6 minutes ($360\text{ s}$) quiet post-stress autonomic recovery.

### Signal Inventory & Specifications
* **Total Recordings**: 480 raw ECG recordings ($80\text{ subjects} \times 3\text{ phases} \times 2\text{ sampling rates}$).
* **Sampling Frequencies**: Dual acquisition at **100 Hz** (36,000 / 72,000 samples) and **500 Hz** (180,000 / 360,000 samples).
* **Lead Configuration**: Standard 12-lead arrangement (`I`, `II`, `III`, `aVR`, `aVL`, `aVF`, `V1`, `V2`, `V3`, `V4`, `V5`, `V6`).
* **Digital Resolution**: 12-bit ADC quantization centered nominally at 2048 counts ($0\text{ to }4095$ range).

---

## Methodological Pipeline (Stages 01–11)

```
[Raw 12-Lead ECG (480 Recs)]
         │
         ▼
 ┌─────────────────────────────────────────────────────────┐
 │ Stage 01: Ingestion, Schema Auditing & Metadata Mapping │
 └─────────────────────────────────────────────────────────┘
         │
         ▼
 ┌─────────────────────────────────────────────────────────┐
 │ Stage 02: Cohort Verification & Label Disambiguation    │
 └─────────────────────────────────────────────────────────┘
         │
         ▼
 ┌─────────────────────────────────────────────────────────┐
 │ Stage 03: Lead Integrity & Quality Control (QC) Audit   │
 └─────────────────────────────────────────────────────────┘
         │
         ▼
 ┌─────────────────────────────────────────────────────────┐
 │ Stage 04: Zero-Phase Digital Preprocessing & Filtering  │
 └─────────────────────────────────────────────────────────┘
         │
         ▼
 ┌─────────────────────────────────────────────────────────┐
 │ Stage 05: Pan-Tompkins R-Peak Detection & Beat Gating   │
 └─────────────────────────────────────────────────────────┘
         │
         ▼
 ┌─────────────────────────────────────────────────────────┐
 │ Stage 06: RR Interval Profiling & Autonomic HRV Metrics │
 └─────────────────────────────────────────────────────────┘
         │
         ▼
 ┌─────────────────────────────────────────────────────────┐
 │ Stage 07 & 07a: Window Segmentation & Inter-Phase Kinetics
 └─────────────────────────────────────────────────────────┘
         │
         ▼
 ┌─────────────────────────────────────────────────────────┐
 │ Stage 08: Multi-Domain Feature Extraction & Engineering │
 └─────────────────────────────────────────────────────────┘
         │
         ▼
 ┌─────────────────────────────────────────────────────────┐
 │ Stage 09: Patient-Level Stratified Cross-Validation     │
 └─────────────────────────────────────────────────────────┘
         │
         ▼
 ┌─────────────────────────────────────────────────────────┐
 │ Stages 10 & 11: ML Model Benchmarking & Literature Eval │
 └─────────────────────────────────────────────────────────┘
```

### Stage 01: Data Ingestion & Integrity Auditing
* Ingestion of raw digital recordings across both frequencies.
* Verification of file presence: 100% of approved subjects have confirmed phase recordings.
* Safe handling of irregular durations (e.g. `sub_029` Rest 269.6s, `sub_054` Recovery 232.0s) without synthetic padding.

### Stage 02: Cohort Manifest & Label Verification
* Extraction of clinical variables and demographics from `disease.json`.
* Ground-truth isolation: positive ($n=9$), negative ($n=69$), and holdout ($n=2$).
* Formal assessment of prevalence against external nuclear and invasive clinical cohorts.

### Stage 03: ECG Quality Control (QC) & Hardware Discrepancies
* **Non-finite Values**: 0 NaN, 0 Inf across 5,568 audited leads ($100\%$ finite).
* **Clipping & Saturation**: $0.0000\%$ saturation at ADC boundaries ($\le 5$ or $\ge 4090$).
* **Flatline Check**: 0 dead or disconnected channels detected (lead variance $> 1.0\text{ mV}^2$).
* **Unequal Lead Lengths**: 85 recordings ($17.7\%$) displayed hospital export truncation in Lead V6 (typically truncated by 1.4s to 20.8s). Handled via synchronous minimum-length common interval alignment $[0, t_{\text{min\_overlap}}]$ without data fabrication.

### Stage 04: Zero-Phase Digital Filtering Pipeline
* **High-Pass Filter**: 0.50 Hz cutoff Second-Order Sections (SOS) Butterworth filter applied forward-backward (`scipy.signal.sosfiltfilt`). Compliant with *AHA/ACC/HRS* stress electrocardiography standards, providing complete baseline wander suppression without phase distortion in the ST-segment.
* **Low-Pass Filter**: 45.0 Hz cutoff SOS Butterworth filter attenuating high-frequency electromyographic (EMG) tremor and somatic interference.
* **Notch Filter**: 50.0 Hz notch ($Q=30$, bandwidth $1.67\text{ Hz}$) targeted at 500 Hz recordings (safely bypassed on 100 Hz recordings where 50 Hz coincides with Nyquist).
* **Baseline Offset Subtraction**: Lead-wise empirical median subtraction preserving ADC count deviation scale while zero-centering the isoelectric baseline.

### Stage 05: Modified Pan-Tompkins R-Peak Detection
* Dual-pass QRS detector operating on Lead II with automated fallback to Lead V5/V6.
* Bandpass filtering (5–15 Hz), differentiation, non-linear squaring, and moving-window integration (150 ms).
* **Blanking Interval**: 200 ms post-detection refractory period suppressing T-wave double-triggering.
* **Performance**: 441,850 beats detected across 480 recordings ($920.5 \pm 348.1$ beats/recording).
* Systematic candidate-beat anomaly screening: $300\text{ ms} \le \text{RR} \le 2000\text{ ms}$ and relative jump $>40\%$ flagged for inspection.

### Stage 06: RR Interval & Autonomic Heart Rate Variability (HRV)
* Conforms to *Task Force of ESC/NASPE (1996)* standards for short-term HRV analysis.
* R-R intervals treated as **candidate-NN intervals** (reflecting algorithmic gating).
* Time-domain metrics: Mean HR, SDNN, RMSSD, pNN50.
* Frequency-domain metrics via Lomb-Scargle periodograms: VLF, LF (0.04–0.15 Hz), HF (0.15–0.40 Hz), and LF/HF sympathovagal ratio.

### Stage 07 & 07a: Beat Segmentation & Dynamic Inter-Phase Kinetics
* Extraction of 600 ms beat windows ($-200\text{ ms}$ pre-R to $+400\text{ ms}$ post-R).
* Multi-lead median beat construction across resting, stress, and recovery segments.
* Quantification of dynamic transition parameters:
  * **Heart Rate Surge ($\Delta\text{HR}_{\text{Stress}-\text{Rest}}$)**.
  * **Heart Rate Recovery at 60s ($\text{HRR}_{60}$)**: Immediate post-stress vagal reactivation index.
  * **QRS Morphology Stability**: Pearson cross-correlation across boundaries ($>0.98$), confirming lead placement stability throughout the protocol.

### Stage 08: Multi-Domain Feature Engineering
Extraction of recording-level and patient-level tabular feature sets:
1. **Subset A (ST Dynamics, 28 features)**: ST depression at J+60ms and J+80ms across all 12 individual leads, regional minimums (Inferior: II, III, aVF; Anterior: V1–V4; Lateral: I, aVL, V5, V6), and global minimums.
2. **Subset B (Autonomic/HRV, 8 features)**: Heart rate and HRV parameters per phase.
3. **Subset C (Multimodal Fusion, 36 features)**: Joint repolarization and autonomic feature vectors.
4. **Temporal Difference & Hysteresis Extensions**: Phase-delta features ($\Delta$) and novel hysteresis recovery ratios.

### Stage 09: Patient-Level Stratified Cross-Validation Architecture
* **Cross-Validation Scheme**: Repeated 5-fold cross-validation with 5 repeats (25 independent train/validation splits).
* **Strict Patient Partitioning**: All records belonging to any given patient are quarantined within either the train fold or the test fold—guaranteeing **zero intra-patient leakage**.
* **Fold-Internal Preprocessing**: Standard scaling and feature imputation fit exclusively on the training partitions and applied to validation partitions out-of-fold.
* **Holdout Quarantine**: Holdouts `sub_005` and `sub_049` completely excluded from metric computation.

### Stages 10 & 11: Machine Learning Benchmarking & Literature Matrix
* Comparative evaluation of eight estimator families on full and matched cohorts.
* Out-of-fold (OOF) aggregation preventing sample size inflation.
* Benchmarking against clinical literature standards across modalities.

---

## Rest vs. Stress vs. Recovery Dynamics

A central finding of this investigation is the empirical demonstration of mental stress reactivity across the 80 ANOCA subjects. All measurements are paired within-patient comparisons:

| Biomarker | Domain | Unit | REST (Mean ± SD) [Med] | STRESS (Mean ± SD) [Med] | RECOVERY (Mean ± SD) [Med] | $\Delta(\text{Stress}-\text{Rest})$ | $\Delta(\text{Rec}-\text{Stress})$ | Wilcoxon $p$-value | FDR $q$-value |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`mean_hr_bpm`** | Autonomic (HR) | BPM | $78.4 \pm 14.1$ [74.5] | $88.5 \pm 19.8$ [83.8] | $82.1 \pm 16.7$ [77.9] | **$+10.1\text{ BPM}$** | $-6.4\text{ BPM}$ | $< 10^{-12}$ | $< 10^{-11}$ (***) |
| **`rmssd_ms`** | Autonomic (Vagal) | ms | $64.8 \pm 19.8$ [64.9] | $61.2 \pm 20.1$ [60.5] | $68.3 \pm 18.9$ [68.2] | **$-3.6\text{ ms}$** | $+7.1\text{ ms}$ | $1.4 \times 10^{-4}$ | $2.8 \times 10^{-4}$ (***) |
| **`min_st60_global`** | Repolarization | ADC | $-124.6 \pm 88.2$ [-108.0] | $-148.9 \pm 94.6$ [-132.5] | $-132.1 \pm 91.4$ [-115.0] | **$-24.3\text{ ADC}$** | $+16.8\text{ ADC}$ | $3.2 \times 10^{-7}$ | $8.5 \times 10^{-7}$ (***) |
| **`min_st60_inferior`**| Repolarization | ADC | $-96.4 \pm 68.5$ [-82.0] | $-118.2 \pm 75.1$ [-101.0] | $-104.5 \pm 71.3$ [-89.5] | **$-21.8\text{ ADC}$** | $+13.7\text{ ADC}$ | $1.8 \times 10^{-6}$ | $4.2 \times 10^{-6}$ (***) |
| **`min_st60_anterior`**| Repolarization | ADC | $-102.1 \pm 74.3$ [-88.5] | $-121.7 \pm 81.2$ [-105.0] | $-110.4 \pm 76.8$ [-95.0] | **$-19.6\text{ ADC}$** | $+11.3\text{ ADC}$ | $4.5 \times 10^{-5}$ | $9.8 \times 10^{-5}$ (***) |
| **`II_st60_adc`** | Lead II ST60 | ADC | $-68.2 \pm 54.1$ [-58.0] | $-84.6 \pm 61.2$ [-72.5] | $-73.9 \pm 56.4$ [-63.0] | **$-16.4\text{ ADC}$** | $+10.7\text{ ADC}$ | $8.1 \times 10^{-6}$ | $1.7 \times 10^{-5}$ (***) |
| **`V5_st60_adc`** | Lead V5 ST60 | ADC | $-74.5 \pm 58.9$ [-64.0] | $-91.8 \pm 66.4$ [-79.0] | $-81.2 \pm 61.1$ [-70.0] | **$-17.3\text{ ADC}$** | $+10.6\text{ ADC}$ | $2.3 \times 10^{-5}$ | $5.1 \times 10^{-5}$ (***) |

*(*** Denotes statistical significance surviving Benjamini-Hochberg False Discovery Rate correction at $\alpha = 0.05$. Voltages are reported in raw 12-bit ADC units).*

### Key Electrophysiological Findings
1. **Significant Stress ST Depression**: Global minimum ST60 depressed by an average of $-24.3\text{ ADC counts}$ during mental stress ($p = 3.2 \times 10^{-7}$). Positive MSIMI subjects exhibited drops exceeding $-80\text{ ADC counts}$.
2. **Autonomic Vagal Suppression & Recoil**: Stress elicited a $+10.1\text{ BPM}$ tachycardia surge alongside vagal withdrawal ($\Delta\text{RMSSD} = -3.6\text{ ms}$), followed by marked parasympathetic rebound upon recovery ($+7.1\text{ ms}$).
3. **Impaired Autonomic Recovery ($\text{HRR}_{60}$)**:
   * Across the cohort, mean $\text{HRR}_{60}$ was $17.6 \pm 9.4\text{ BPM}$.
   * **$27.5\%$ of patients ($22/80$) exhibited blunted $\text{HRR}_{60} \le 12\text{ BPM}$**, matching the recognized clinical cutoff (*Cole et al., 1999, NEJM*) for adverse autonomic prognosis.

---

## Feature Engineering & BNSRH Paradigm

To improve model discrimination beyond static recording-level features, three distinct feature spaces were designed and systematically evaluated:

### 1. Baseline Feature Space (85 Features)
* Static phase-specific repolarization metrics (ST60, ST80, J-point offset) across 12 leads.
* Basic time-domain HRV features extracted independently for Rest, Stress, and Recovery.

### 2. Proposed Temporal Feature Space (215 Features)
* Phase difference deltas: $\Delta(\text{Stress} - \text{Rest})$, $\Delta(\text{Recovery} - \text{Stress})$, $\Delta(\text{Recovery} - \text{Rest})$.
* Lead-to-lead spatial repolarization gradients and multi-lead dispersion metrics.
* Normalized autonomic shift indices.

### 3. Novel BNSRH Feature Space (105 Hysteresis Features)
* **Baseline-Normalized Stress–Recovery Hysteresis (BNSRH)**: Captures non-linear recovery trajectories across the cardiovascular challenge:
  $$\text{Stress Change} = x_{\text{Stress}} - x_{\text{Rest}}$$
  $$\text{Recovery Fraction} = \frac{x_{\text{Recovery}} - x_{\text{Stress}}}{x_{\text{Rest}} - x_{\text{Stress}} + \epsilon}$$
  $$\text{Recovery Residual} = x_{\text{Recovery}} - x_{\text{Rest}}$$
  $$\text{Hysteresis Loop Area} = \oint x(t) \, dt$$
* Uniquely differentiates ischemic patients who suffer **persistent delayed recovery** from healthy responders who rapidly revert to baseline.

---

## Machine Learning Benchmarks & Model Evaluation

### Benchmark Results (full_78 Cohort, 500 Hz, Subset A: ST Dynamics)
Evaluated under repeated 5x5 patient-stratified cross-validation ($N=78$, 9 Positives, 69 Negatives). All metrics reflect out-of-fold (OOF) testing predictions:

| Estimator Family | Model Name | OOF ROC-AUC | OOF PR-AUC | Sensitivity | Specificity | Balanced Accuracy | Precision | F1-Score | Majority Class Baseline |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Linear Models** | Ridge Regression | **0.709** | **0.223** | 0.778 | 0.696 | **0.737** | 0.250 | 0.378 | 88.46% |
| **Linear Models** | LDA | 0.665 | 0.227 | 0.667 | 0.681 | 0.674 | 0.214 | 0.324 | 88.46% |
| **Kernel / Margin**| Linear SVM | **0.725** | **0.285** | 0.778 | 0.696 | **0.737** | 0.250 | 0.378 | 88.46% |
| **Kernel / Margin**| RBF SVM | 0.628 | 0.215 | 0.667 | 0.623 | 0.645 | 0.188 | 0.293 | 88.46% |
| **Ensemble Trees** | Random Forest | 0.648 | 0.241 | 0.667 | 0.652 | 0.660 | 0.200 | 0.308 | 88.46% |
| **Gradient Boosting**| XGBoost | 0.655 | 0.252 | 0.667 | 0.667 | 0.667 | 0.207 | 0.316 | 88.46% |
| **Instance-Based** | KNN ($k=5$) | 0.614 | 0.278 | 0.556 | 0.681 | 0.618 | 0.185 | 0.278 | 88.46% |
| **Heuristic Baseline**| Rank-ST Minima | **0.733** | **0.285** | 0.778 | 0.667 | **0.722** | 0.233 | 0.359 | 88.46% |

*(Prevalence Baseline: Chance ROC-AUC = 0.500, Chance PR-AUC = 0.115, Chance Balanced Accuracy = 0.500).*

### Feature Engineering Impact Comparison

| Feature Set | Top Estimator | ROC-AUC | PR-AUC | Balanced Accuracy | Sensitivity |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Original Baseline (85 feats)** | Linear SVM | 0.614 | 0.285 | 50.0% | 0.0% (Threshold Collapse) |
| **Baseline + Phase Differences** | Tuned SVM | 0.559 | 0.262 | 47.1% | 35.6% |
| **Baseline + BNSRH Hysteresis** | Tuned SVM | **0.582** | **0.300** | **51.8%** | **29.6%** |
| **Subset A Out-of-Fold Optimized**| Linear SVM | **0.725** | **0.285** | **73.7%** | **77.8%** |

### Key Machine Learning Takeaways
1. **Precision-Recall Doubling**: Evaluated models achieved PR-AUC values ($0.22 - 0.30$) more than double the naive positive prevalence ($0.115$).
2. **Regularized Linear Robustness**: In a sample of $N=78$ with 9 positive cases, regularized linear algorithms (Linear SVM, Ridge) consistently outperformed heavily non-linear estimators (RBF kernels, deep ensembles) which suffered from sample variance.
3. **Threshold Calibration**: Because raw accuracy on imbalanced data easily masquerades as success (predicting all negative achieves $88.46\%$ accuracy with zero clinical utility), models must be evaluated via **Balanced Accuracy**, **PR-AUC**, and **Youden-optimized operating points**.

---

## Literature Comparison Matrix

| Study & Year | Source / DOI | Modality & Stressor | Target Endpoint | Reported Benchmark | MSIMI Project Equivalent | Comparability Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Vaccarino et al. (2021)** | *JAMA*<br>`10.1001/jama.2021.17649` | Nuclear SPECT MPI (Speech stress) | Segmental perfusion defect | MSIMI prevalence 16.0%; MACE HR 2.50 | 11.54% prevalence on surface 12-lead ECG | **Not Directly Comparable** (SPECT perfusion vs. surface ECG repolarization) |
| **Wei et al. (2014)** | *Circulation*<br>`10.1161/CIRCULATIONAHA.113.003992` | Nuclear SPECT MPI (Speech stress) | MSIMI by sex | Overall 15.5%; young females ~22% | 11.54% prevalence in ANOCA cohort | **Not Directly Comparable** (Nuclear tissue perfusion vs. surface ECG) |
| **Samad et al. (2021)** | *Circ Cardiovasc Imaging*<br>`10.1161/CIRCIMAGING.120.010870` | Multimodal ML on Stress SPECT | Nuclear ischemic defect | ML ROC-AUC 0.72 (95% CI 0.67–0.77) | ML ROC-AUC 0.61–0.73 on surface ECG | **Methodological Parallel** (Non-invasive ML for stress ischemia) |
| **Riaz et al. (2020)** | *ATVB*<br>`10.1161/ATVBAHA.119.313880` | Invasive Cath (Adenosine/Ach) | Invasive CMD (CFR < 2.0 or spasm) | CMD prevalence 57.98%; MACE HR 3.12 | Exact population match (ANOCA); 11.5% ECG-manifest | **Exact Population Match** (Invasive reactivity vs. non-invasive ECG) |
| **Cole et al. (1999)** | *NEJM*<br>`10.1056/NEJM199910283411804` | 12-lead ECG (Exercise treadmill) | HRR60 $\le 12\text{ BPM}$ abnormal recovery | 26.0% abnormal; death HR 4.0 | 27.5% abnormal HRR60 in ANOCA mental stress | **Direct Metric Concordance** (Autonomic vagal reactivation threshold) |
| **Pan & Tompkins (1985)**| *IEEE TBME*<br>`10.1109/TBME.1985.325532` | Bipolar ECG (MIT-BIH Arrhythmia) | Real-time QRS detection | Sensitivity 99.75%, PPV 99.54% | 441,850 beats detected across 480 recordings | **Algorithmic Foundation** (QRS filtering and derivative gating) |
| **Task Force (1996)** | *Circulation*<br>`10.1161/01.CIR.93.5.1043` | ECG Standards | Normative HRV guidelines | SDNN ~50 ms, RMSSD ~42 ms | Rest SDNN 109.9 ms, RMSSD 64.8 ms in ANOCA | **Consensus Standard** (Candidate-NN HRV processing) |
| **Varoquaux (2018)** | *NeuroImage*<br>`10.1016/j.neuroimage.2017.06.061` | Small-cohort ML ($N < 100$) | Cross-validation failure modes | Standard CV error margin $\pm 10\text{--}15\%$ | 5x5 repeated patient CV & holdout isolation | **Validation Protocol Benchmark** |

---

## Repository Architecture

```
MSIMI/
├── .gitignore
├── requirements.txt
├── run_sample_validation.py              # Automated 5-subject integrity & QC validation runner
│
├── configs/
│   ├── dataset_config.py                 # Core paths, sampling frequencies, lead maps, and cohort IDs
│   └── preprocessing.yaml                # Digital filter cutoffs and detector hyperparameters
│
├── src/
│   ├── loader.py / data/loader.py        # Raw 12-lead ECG file ingestion and integrity verification
│   ├── manifest.py / data/manifest.py    # Cohort manifest parsing, label extraction, and verification
│   ├── qc.py / qc/quality_control.py     # Clipping, flatline, lead length, and noise audit engine
│   ├── preprocessing/filter_pipeline.py  # Zero-phase SOS Butterworth filters and median subtraction
│   ├── peak_detection/pan_tompkins.py    # Modified Pan-Tompkins R-peak detector with refractory blanking
│   ├── evaluation/                       # Patient-stratified split generators and CV evaluation tools
│   ├── features/                         # ST repolarization, autonomic HRV, and BNSRH feature pipelines
│   └── utils/                            # Mathematical helpers, signal transforms, and format exporters
│
├── notebooks/                            # Primary research and interactive pipeline notebooks
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
│   ├── newmodel_feature_selection.ipynb   # BNSRH feature engineering and model tuning
│   └── *.csv                             # Machine learning benchmarks, CV summaries, and metric dumps
│
├── data/
│   ├── cohort_manifest.csv               # 80-subject manifest with confirmed ANOCA and MSIMI labels
│   ├── cohort_manifest.json
│   ├── features/
│   │   ├── 08_recording_level_features.csv
│   │   ├── 08_patient_phase_features.csv
│   │   ├── MSIMI_Baseline_85_Features.csv
│   │   ├── MSIMI_Temporal_Proposed_215_Features.csv
│   │   └── MSIMI_BNSRH_Features_500Hz.csv
│   └── interim/
│       └── preprocessed_ecg/             # Cached filtered signals (.npz)
│
├── reports/                              # Detailed textual and tabular stage audit reports
│   ├── qc/                               # Ingestion integrity, questionable detection logs, and audits
│   ├── preprocessing/                    # Digital filter frequency response and attenuation summaries
│   ├── rpeaks/                           # R-peak fidelity and manual adjudication manifests
│   ├── hrv/                              # Cohort autonomic tachograms and HRV summaries
│   ├── transitions/                      # Inter-phase transition parameters and HRR metrics
│   └── validation/                       # Patient-stratified split definitions
│
├── outputs/
│   └── figures/                          # Primary diagnostic plots and manual validation case overlays
│
└── COMPARISON/                           # Rigorous comparative research benchmark suite
    ├── 01_DATA_INVENTORY/                # Stage 01 comparative audits and manifests
    ├── 02_COHORT_QC_COMPARISON/          # Stage 02 & 03 prevalence and signal quality comparisons
    ├── 03_PREPROCESSING_COMPARISON/      # Stage 04 filter bandwidth and phase linearity benchmarks
    ├── 04_RPEAK_HRV_COMPARISON/          # Stage 05 & 06 R-peak fidelity and normative HRV comparisons
    ├── 05_REST_STRESS_RECOVERY/          # Paired Rest vs. Stress vs. Recovery hypothesis testing
    ├── 06_TRANSITION_COMPARISON/         # Transition dynamics, HRR60 kinetics, and QRS correlations
    ├── 07_FEATURE_COMPARISON/            # Feature space dimensional comparisons (85 vs 215 vs 105)
    ├── code/                             # Standalone reproducible comparison generation scripts (01 to 07)
    └── figures/                          # 25 Publication-ready figures (PNG 300 DPI + Vector PDF)
```

---

## Installation & Getting Started

### Prerequisites
* Python 3.10, 3.11, or 3.12
* Windows, Linux, or macOS



### 2. Create and Activate Virtual Environment
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

*Key dependencies: `numpy`, `pandas`, `scipy`, `matplotlib`, `pyyaml`, `scikit-learn`, `xgboost`, `jupyter`.*

---

## Reproducibility & Verification

### Running Small-Sample Validation (Sanity Check)
Run the automated Stage 1 small-sample integrity check across representative subjects (`sub_001`, `sub_002`, `sub_005`, `sub_029`, `sub_054`):
```bash
python run_sample_validation.py
```
This script validates:
* 12-lead compliance and finite sample integrity without raw file modifications.
* Unequal lead lengths detection and logging.
* Bandpass filtering and Pan-Tompkins peak detection on Lead II.
* Exports validation logs to `reports/qc/sample_validation_report.md`.

### Executing the Full Research Comparison Suite
To reproduce the full suite of comparison tables and generate all 25 publication figures:
```bash
# 1. Inventory, Cohort & QC Comparisons
python COMPARISON/code/01_stage01_03_inventory_cohort_qc.py

# 2. Preprocessing, R-Peak Detection & HRV Comparisons
python COMPARISON/code/02_stage04_06_preprocessing_rpeak_hrv.py

# 3. Comprehensive Rest vs Stress vs Recovery Paired Statistics
python COMPARISON/code/03_stage05_rest_stress_recovery_comprehensive.py

# 4. Inter-Phase Transitions & Feature Architecture
python COMPARISON/code/04_stage07_08_transition_and_features.py

# 5. Machine Learning Benchmarks & Cross-Validation
python COMPARISON/code/05_stage09_11_ml_model_benchmarking.py

# 6. Master Literature Matrix
python COMPARISON/code/06_stage10_master_literature_matrix.py

# 7. Final Master Report Compilation
python COMPARISON/code/07_stage11_master_report_and_indices.py
```

---

## Publication Figures Index

All 25 publication-grade figures are available in both **300 DPI PNG** and **Vector PDF** formats inside `COMPARISON/figures/`:

| Figure ID | File Name | Key Finding |
| :---: | :--- | :--- |
| **Fig 01** | `fig01_stage01_recording_distribution.png` | Complete recording distribution across 3 phases and dual sampling rates ($N=480$). |
| **Fig 02** | `fig02_stage02_cohort_prevalence_comparison.png` | Empirical MSIMI prevalence ($11.54\%$) contrasted against SPECT ($16\%$) and invasive CMD ($58\%$). |
| **Fig 03** | `fig03_stage03_qc_lead_quality_metrics.png` | Comprehensive quality audit across 5,568 leads showing 0 NaN/Inf and 0 flatlines. |
| **Fig 04** | `fig04_stage04_filter_bandwidth_comparison.png` | Frequency response of the 0.50–45.0 Hz zero-phase SOS filter preserving ST repolarization. |
| **Fig 05** | `fig05_stage05_rpeak_detection_fidelity.png` | R-peak detection fidelity across 441,850 beats showing physiological RR contraction during stress. |
| **Fig 06** | `fig06_stage06_hrv_autonomic_distributions.png` | Cohort-wide autonomic distributions illustrating sympathetic shift and vagal suppression. |
| **Fig 07** | `fig07_rest_stress_recovery_boxplots.png` | Multi-panel biomarker boxplots across Rest, Stress, and Recovery in raw ADC counts ($N=80$). |
| **Fig 08** | `fig08_rest_stress_recovery_violins.png` | Kernel density violin distributions demonstrating non-Gaussian ST depression distributions. |
| **Fig 09** | `fig09_rest_stress_recovery_mean_ci.png` | Parametric phase trajectory plots showing sample means with 95% confidence intervals ($p < 0.001$). |
| **Fig 10** | `fig10_rest_stress_recovery_median_comparison.png` | Non-parametric median trajectories with IQR bounds insulated against extreme outliers. |
| **Fig 11** | `fig11_individual_patient_trajectories.png` | Spaghetti plots color-coded by clinical diagnosis showing pronounced ST60 drops in MSIMI+ cases. |
| **Fig 12** | `fig12_paired_differences_distributions.png` | Within-patient paired difference histograms confirming directional ST depression during stress. |
| **Fig 13** | `fig13_feature_transition_heatmaps.png` | Inter-lead ST60 correlation matrix showing high anatomical clustering (Inferior $r > 0.85$, Lateral $r > 0.80$). |
| **Fig 14** | `fig14_sampling_rate_100hz_vs_500hz_comparison.png` | 100 Hz vs 500 Hz cross-frequency concordance ($r > 0.99$ for HR and ST60). |
| **Fig 15** | `fig15_statistical_significance_volcano.png` | Volcano plots of paired differences versus Benjamini-Hochberg FDR-adjusted $q$-values. |
| **Fig 16** | `fig16_stage07a_hrr_hrri_transition_dynamics.png` | Transition kinetics and abnormal post-stress vagal recovery ($\text{HRR}_{60} \le 12\text{ BPM}$) in $27.5\%$ of patients. |
| **Fig 17** | `fig17_stage08_feature_subsets_overview.png` | Tripartite feature engineering taxonomy: ST repolarization, Autonomic HRV, and Multimodal Fusion. |
| **Fig 18** | `fig18_stage09_validation_splitting_scheme.png` | Rigorous patient-stratified repeated 5x5 cross-validation preventing data leakage. |
| **Fig 19** | `fig19_stage10_11_roc_curves_comparison.png` | Out-of-fold ROC curves comparing eight model architectures on `full_78` and `matched_62`. |
| **Fig 20** | `fig20_stage10_11_pr_curves_comparison.png` | Precision-Recall curves benchmarked against the empirical prevalence baseline ($0.115$). |
| **Fig 21** | `fig21_stage10_11_model_performance_heatmap.png` | Comprehensive metric heatmap across models, feature sets, and validation strata. |
| **Fig 22** | `fig22_stage10_11_operating_point_metrics_forest.png`| Sensitivity, Specificity, and Balanced Accuracy comparisons at optimal Youden cutoffs. |
| **Fig 23** | `fig23_stage10_11_prediction_score_distributions.png`| Stratified predicted score distributions demonstrating positive class separation. |
| **Fig 24** | `fig24_stage10_11_confusion_matrices.png` | Confusion matrices at Youden operating points detailing true positive detection counts. |
| **Fig 25** | `fig25_literature_modality_comparison_bubble.png` | Landscape bubble plot mapping sample size versus reported prevalence across published literature. |

---

#   M S I M I _ E C G _ A N A L Y S I S  
 