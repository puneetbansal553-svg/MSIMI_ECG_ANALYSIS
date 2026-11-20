### Section 5: Fold-Local Preprocessing & Missing-Value Handling

**Mandatory Scientific Safeguards**:
1. Zero Data Leakage: Imputers and scalers must be **fitted strictly on the training partition (`X_train`)** of each fold.
2. Validation Transform: Validation partitions (`X_val`) are transformed using training parameters.
3. Scaler Robustness: `FoldLocalRobustScaler` utilizes median and interquartile range (IQR) to prevent outlier distortion from ECG sensor transients.
4. Structural Missingness Handling: `FoldLocalMedianImputer` fits training medians to handle the 16 skipped 100 Hz stress recordings.
