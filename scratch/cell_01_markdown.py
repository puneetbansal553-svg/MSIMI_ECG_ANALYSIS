# 10 - Model Development & Patient-Level Evaluation
**MSIMI Clinical Research Pipeline — Stage 10**

### Objective & Clinical Scope
Stage 10 executes the patient-level modeling and cross-validation evaluation for Mental Stress-Induced Myocardial Ischemia (MSIMI) in patients with Ischemia and Non-Obstructive Coronary Artery disease (ANOCA).

### Analytical Architecture & Governance Invariants
1. **Target Cohort Invariants**:
   - Total Registered Cohort: Exactly **80 subjects** (`sub_001` through `sub_080`).
   - CV-Eligible Labelled Cohort: Exactly **78 subjects** ($P=9$ positive, $N=69$ negative; prevalence = $11.54\%$).
   - Unlabelled Holdouts: Exactly **2 subjects** (`sub_005`, `sub_049`; `target_label = NaN`, `validation_eligible = False`, `repeat_*_fold = -1`).
   - Matched Dual-Rate Complete Sub-cohort: Exactly **62 labelled subjects** ($P=7, N=55$) with complete stress data at both 100 Hz and 500 Hz (Total matched cohort including holdouts: $N=64$).
   - Single-Rate Incomplete Sub-cohort: Exactly **16 labelled subjects** ($P=2, N=14$) with skipped 100 Hz stress (`SKIPPED_UPSTREAM_ERROR`).
2. **Zero Data Leakage Invariants**:
   - Patient-level grouping invariance: Both sampling-rate strata (100 Hz and 500 Hz) for patient $i$ reside in the identical fold.
   - Fold-local encapsulation: Every transformation (median imputation, `RobustScaler`) is fitted strictly on `X_train` within each fold.
   - Holdout isolation: `sub_005` and `sub_049` are strictly excluded from model training, validation metrics, and threshold tuning.
3. **Scientific Safeguards & Terminology Mandates**:
   - Raw ADC counts: ECG amplitude features are in raw ADC counts relative to local PR baseline; no physical voltage calibration ($\mu	ext{V}$ or $	ext{mV}$) is assumed.
   - Discrimination metrics primary: PR-AUC (Average Precision) and ROC-AUC are the primary continuous evaluation metrics.
   - Fixed reference score cutoffs: Secondary operating points are reported at $0.50$, $9/78 pprox 0.1154$, and $7/62 pprox 0.1129$. Outputs are uncalibrated discriminative scores; **no clinical risk probability interpretation is permitted**.
   - Holdout dispersion: Model variation across the 25 fold models is termed `between_model_prediction_dispersion` (descriptive variability across splits, not validated epistemic uncertainty).
   - Exploratory ST ranking: Single-feature ST-deviation ranking is an exploratory, non-clinical benchmark, not a validated clinical rule.
