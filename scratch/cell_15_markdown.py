### Section 13 & 14: Comprehensive Forensic Audits & Deliverable Contracts

Before serializing artifacts, Stage 10 enforces five forensic verification audits:
1. **Audit 1 (Zero Leakage)**: Subject disjointness across all 25 training/validation folds.
2. **Audit 2 (Holdout Isolation)**: Verify `sub_005` and `sub_049` are strictly excluded from fitting, validation metrics, and threshold selection.
3. **Audit 3 (Multi-Rate Strata Co-Location)**: Verify 100 Hz and 500 Hz strata share identical fold assignments.
4. **Audit 4 (Confusion Matrix Accounting Invariance)**: Exact positive and negative accounting per repeat ($TP+FN=9, TN+FP=69$ on full cohort; $P=7, N=55$ on matched cohort).
5. **Audit 5 (Upstream Immutability)**: Cryptographic SHA-256 verification of Stage 08 and Stage 09 artifacts.
