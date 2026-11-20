def score_unlabelled_holdouts(
    df_features: pd.DataFrame,
    df_splits_reg: pd.DataFrame,
    feature_cols: List[str],
    sampling_rate: int,
    model_type: str = "ridge"
) -> pd.DataFrame:
    """
    Scores unlabelled holdout patients (sub_005, sub_049) across all 25 fold models.
    Computes ensemble mean score and between_model_prediction_dispersion.
    Zero metrics, losses, or gradients computed.
    """
    df_stratum = df_features[df_features["sampling_rate"] == sampling_rate].copy()
    m = pd.merge(df_stratum, df_splits_reg, on="subject_id")

    # Filter holdouts strictly
    m_holdouts = m[m["subject_id"].isin(["sub_005", "sub_049"])].copy()
    assert len(m_holdouts) == 2, f"Expected 2 holdouts, found {len(m_holdouts)}"
    m_train_pool = m[m["validation_eligible"]].copy()

    X_holdouts_raw = m_holdouts[feature_cols].values.astype(float)
    holdout_scores_all_models = {h_id: [] for h_id in m_holdouts["subject_id"]}

    for r_idx in range(5):
        fold_col = f"repeat_{r_idx}_fold"
        for f_idx in range(5):
            train_mask = m_train_pool[fold_col] != f_idx
            X_train = m_train_pool.loc[train_mask, feature_cols].values.astype(float)
            y_train = m_train_pool.loc[train_mask, "target_label"].astype(int).values

            # Fit fold-local preprocessors on training split ONLY
            imputer = FoldLocalMedianImputer().fit(X_train, feature_cols)
            X_train_imp = imputer.transform(X_train)
            X_hold_imp = imputer.transform(X_holdouts_raw)

            scaler = FoldLocalRobustScaler().fit(X_train_imp)
            X_train_scl = scaler.transform(X_train_imp)
            X_hold_scl = scaler.transform(X_hold_imp)

            # Fit model
            if model_type == "ridge":
                model = PureNumpyRidgeLogisticRegression(lambda_reg=1.0).fit(X_train_scl, y_train)
                scores = model.predict_score(X_hold_scl)
            elif model_type == "lda":
                model = PureNumpyShrinkageLDA(gamma=0.20).fit(X_train_scl, y_train)
                scores = model.predict_score(X_hold_scl)
            elif model_type == "rank_st":
                st_col_idx = feature_cols.index("stress_min_st60_global_adc")
                neg_st_train = -X_train_imp[:, st_col_idx]
                neg_st_hold = -X_hold_imp[:, st_col_idx]
                min_val = np.min(neg_st_train)
                max_val = np.max(neg_st_train)
                scores = (neg_st_hold - min_val) / (max_val - min_val + 1e-8)

            for i, h_id in enumerate(m_holdouts["subject_id"]):
                holdout_scores_all_models[h_id].append(float(scores[i]))

    holdout_summary_rows = []
    for h_id in sorted(list(holdout_scores_all_models.keys())):
        arr = np.array(holdout_scores_all_models[h_id])
        mean_score = float(np.mean(arr))
        # Renamed descriptive dispersion metric
        dispersion = float(np.std(arr, ddof=1))
        holdout_summary_rows.append({
            "subject_id": h_id,
            "sampling_rate": sampling_rate,
            "model_type": model_type,
            "actual_target": "UNLABELLED_HOLDOUT",
            "ensemble_mean_score": mean_score,
            "between_model_prediction_dispersion": dispersion,
            "n_models_evaluated": len(arr)
        })

    return pd.DataFrame(holdout_summary_rows)

print("Holdout Test Inference Engine compiled successfully.")
