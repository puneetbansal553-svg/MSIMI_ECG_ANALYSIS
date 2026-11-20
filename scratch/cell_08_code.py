class FoldLocalMedianImputer:
    """
    Pure-NumPy median imputer fitted strictly on training data.
    """
    def __init__(self):
        self.medians_: Optional[np.ndarray] = None
        self.feature_names_: Optional[List[str]] = None

    def fit(self, X: np.ndarray, feature_names: Optional[List[str]] = None) -> 'FoldLocalMedianImputer':
        assert X.ndim == 2, "X must be a 2D array."
        # Compute median along columns ignoring NaNs
        self.medians_ = np.nanmedian(X, axis=0)
        # If any column is entirely NaN, fall back to 0.0
        self.medians_ = np.where(np.isnan(self.medians_), 0.0, self.medians_)
        self.feature_names_ = feature_names
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        assert self.medians_ is not None, "Imputer is not fitted."
        X_out = X.copy()
        for j in range(X_out.shape[1]):
            nan_mask = np.isnan(X_out[:, j])
            if np.any(nan_mask):
                X_out[nan_mask, j] = self.medians_[j]
        return X_out

    def fit_transform(self, X: np.ndarray, feature_names: Optional[List[str]] = None) -> np.ndarray:
        return self.fit(X, feature_names).transform(X)


class FoldLocalRobustScaler:
    """
    Pure-NumPy Robust Scaler using Median and Interquartile Range (IQR = Q75 - Q25).
    Fitted strictly on training data.
    """
    def __init__(self, unit_variance: bool = True):
        self.unit_variance = unit_variance
        self.center_: Optional[np.ndarray] = None
        self.scale_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> 'FoldLocalRobustScaler':
        assert X.ndim == 2, "X must be a 2D array."
        self.center_ = np.median(X, axis=0)
        q25 = np.percentile(X, 25, axis=0)
        q75 = np.percentile(X, 75, axis=0)
        scale = q75 - q25
        # Prevent division by zero for constant features
        scale = np.where(scale < 1e-8, 1.0, scale)
        if self.unit_variance:
            # Scale factor for normal distribution unit variance
            scale = scale / 1.3489795003921634
        self.scale_ = scale
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        assert self.center_ is not None and self.scale_ is not None, "Scaler is not fitted."
        return (X - self.center_) / self.scale_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

print("Pure-NumPy Preprocessing Engines compiled successfully.")
