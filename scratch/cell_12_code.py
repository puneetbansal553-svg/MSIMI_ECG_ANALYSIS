def run_repeated_patient_cross_validation(
    df_features: pd.DataFrame,
    df_splits_reg: pd.DataFrame,
    feature_cols: List[str],
    sampling_rate: int,
    model_type: str = "ridge",
    matched_only: bool = False
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Executes 5 Repeats x 5 Folds Cross-Validation using frozen Stage 09 splits.
    Strictly fit on training fold; transform on validation fold.
    Returns per-repeat metrics and pooled OOF predictions.
    """
    assert sampling_rate in (100, 500), f"Invalid sampling rate: {sampling_rate}"
    assert model_type in ("ridge", "lda", "rank_st"), f"Invalid model: {model_type}"

    # Filter stratum and merge with splits
    df_stratum = df_features[df_features["sampling_rate"] == sampling_rate].copy()
    m = pd.merge(df_stratum, df_splits_reg, on="subject_id")

    # Eligible subjects filter
    if matched_only:
        # Restrict strictly to the 62 complete dual-rate labelled subjects
        m_eval = m[(m["validation_eligible"]) & (m["subject_id"].isin(matched_ids))].copy()
        assert len(m_eval) == 62, f"Expected 62 matched subjects, found {len(m_eval)}"
    else:
        # Full 78 labelled subjects
        m_eval = m[m["validation_eligible"]].copy()
        assert len(m_eval) == 78, f"Expected 78 eligible subjects, found {len(m_eval)}"

    subjects = m_eval["subject_id"].values
    y_true_all = m_eval["target_label"].astype(int).values
    n_cohort = len(m_eval)

    oof_predictions_by_repeat = {}
    repeat_metrics = []

    for r_idx in range(5):
        fold_col = f"repeat_{r_idx}_fold"
        oof_scores = np.zeros(n_cohort, dtype=float)

        for f_idx in range(5):
            val_mask = m_eval[fold_col] == f_idx
            train_mask = m_eval[fold_col] != f_idx

            X_train = m_eval.loc[train_mask, feature_cols].values.astype(float)
            y_train = m_eval.loc[train_mask, "target_label"].astype(int).values
            X_val = m_eval.loc[val_mask, feature_cols].values.astype(float)

            # Fit fold-local imputer & scaler on X_train ONLY
            imputer = FoldLocalMedianImputer().fit(X_train, feature_cols)
            X_train_imp = imputer.transform(X_train)
            X_val_imp = imputer.transform(X_val)

            scaler = FoldLocalRobustScaler().fit(X_train_imp)
            X_train_scl = scaler.transform(X_train_imp)
            X_val_scl = scaler.transform(X_val_imp)

            # Model fitting on training data ONLY
            if model_type == "ridge":
                model = PureNumpyRidgeLogisticRegression(lambda_reg=1.0).fit(X_train_scl, y_train)
                scores_val = model.predict_score(X_val_scl)
            elif model_type == "lda":
                model = PureNumpyShrinkageLDA(gamma=0.20).fit(X_train_scl, y_train)
                scores_val = model.predict_score(X_val_scl)
            elif model_type == "rank_st":
                # Exploratory rank percentile on raw peak stress ST depression
                st_col_idx = feature_cols.index("stress_min_st60_global_adc")
                # More negative ST60 -> higher rank score
                neg_st_val = -X_val_imp[:, st_col_idx]
                min_val = np.min(neg_st_val)
                max_val = np.max(neg_st_val)
                scores_val = (neg_st_val - min_val) / (max_val - min_val + 1e-8)

            oof_scores[val_mask] = scores_val

        oof_predictions_by_repeat[f"repeat_{r_idx}"] = oof_scores

        # Compute Pooled OOF continuous metrics for repeat r
        roc_auc = compute_roc_auc(y_true_all, oof_scores)
        pr_auc = compute_pr_auc(y_true_all, oof_scores)
        brier = float(np.mean((oof_scores - y_true_all) ** 2))

        # Secondary reference score cutoffs
        ref_50 = compute_reference_cutoff_metrics(y_true_all, oof_scores, 0.50)
        prev_cutoff = 7/62 if matched_only else 9/78
        ref_prev = compute_reference_cutoff_metrics(y_true_all, oof_scores, prev_cutoff)

        repeat_metrics.append({
            "repeat": r_idx,
            "sampling_rate": sampling_rate,
            "cohort": "matched_62" if matched_only else "full_78",
            "model_type": model_type,
            "n_eval": n_cohort,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "brier_score": brier,
            "sens_at_50": ref_50["sensitivity"],
            "spec_at_50": ref_50["specificity"],
            "bacc_at_50": ref_50["balanced_accuracy"],
            "sens_at_prev": ref_prev["sensitivity"],
            "spec_at_prev": ref_prev["specificity"],
            "bacc_at_prev": ref_prev["balanced_accuracy"]
        })

    df_metrics = pd.DataFrame(repeat_metrics)
    return repeat_metrics, df_metrics

print("Cross-Validation Orchestration Engine compiled successfully.")
