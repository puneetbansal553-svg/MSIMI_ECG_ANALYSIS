### Section 15 & 16: Serialization Specifications & Publication Figures

Stage 10 defines three structured serialization targets and two publication diagnostic figures:
1. `reports/models/10_model_evaluation_metrics.csv`: Performance metrics per repeat and aggregate summary (Mean $\pm$ SD) for matched ($N=62$) and full ($N=78$) cohorts.
2. `reports/models/10_patient_predictions.csv`: 80 rows containing patient IDs, true labels, pooled OOF probabilities for eligible subjects, and separate holdout inference with `between_model_prediction_dispersion`.
3. `reports/models/10_modeling_governance_summary.json`: Hyperparameters ($\lambda = 1.0$), class weights, audit results, and environment metadata.
4. `outputs/figures/10_roc_prc_benchmark_curves.png`: High-resolution (300 DPI) ROC and PR curves.
5. `outputs/figures/10_patient_prediction_distribution_and_calibration.png`: Probability calibration and score distributions.
