### Section 11 & 12: Unlabelled Holdout Shadow Inference Protocol

**Holdout Governance Requirements**:
1. Blinding & Non-Contamination: `sub_005` and `sub_049` are strictly unlabelled (`target_label = NaN`) and are never used in training, validation, or metric calculation.
2. Terminology Mandate: Holdout predictions are strictly designated **"Unlabelled Holdout Test Inference"** and must **never** be labeled "OOF".
3. Descriptive Dispersion: Variability across the 25 fold models is recorded as **`between_model_prediction_dispersion`**, explicitly documented as descriptive model sample variation across splits, **not a validated epistemic uncertainty estimate**.
