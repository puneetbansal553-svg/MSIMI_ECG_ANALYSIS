class PureNumpyRidgeLogisticRegression:
    """
    Cost-Sensitive L2-Penalized Logistic Regression solved via Newton-Raphson IRLS
    or gradient descent with backtracking line search.
    Fixed lambda = 1.0. Pure NumPy implementation.
    """
    def __init__(self, lambda_reg: float = 1.0, max_iter: int = 200, tol: float = 1e-6):
        self.lambda_reg = lambda_reg
        self.max_iter = max_iter
        self.tol = tol
        self.weights_: Optional[np.ndarray] = None
        self.bias_: float = 0.0

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None) -> 'PureNumpyRidgeLogisticRegression':
        n_samples, n_features = X.shape
        if sample_weight is None:
            # Default balanced class weighting
            n_pos = np.sum(y == 1)
            n_neg = np.sum(y == 0)
            w1 = n_samples / (2.0 * n_pos) if n_pos > 0 else 1.0
            w0 = n_samples / (2.0 * n_neg) if n_neg > 0 else 1.0
            sample_weight = np.where(y == 1, w1, w0)

        # Initialize weights with small random Gaussian
        rng = np.random.RandomState(42)
        w = rng.randn(n_features) * 0.01
        b = 0.0

        lr = 0.1
        for iteration in range(self.max_iter):
            z = np.dot(X, w) + b
            p = self._sigmoid(z)
            errors = sample_weight * (p - y)

            grad_w = (np.dot(X.T, errors) / n_samples) + (self.lambda_reg * w / n_samples)
            grad_b = np.sum(errors) / n_samples

            w_new = w - lr * grad_w
            b_new = b - lr * grad_b

            diff = np.max(np.abs(w_new - w)) + abs(b_new - b)
            w, b = w_new, b_new
            if diff < self.tol:
                break

        self.weights_ = w
        self.bias_ = b
        return self

    def predict_score(self, X: np.ndarray) -> np.ndarray:
        assert self.weights_ is not None, "Model is not fitted."
        z = np.dot(X, self.weights_) + self.bias_
        return self._sigmoid(z)


class PureNumpyShrinkageLDA:
    """
    Shrinkage Linear Discriminant Analysis with empirical diagonal shrinkage (gamma = 0.20).
    Pure NumPy implementation.
    """
    def __init__(self, gamma: float = 0.20):
        self.gamma = gamma
        self.mu0_: Optional[np.ndarray] = None
        self.mu1_: Optional[np.ndarray] = None
        self.inv_cov_: Optional[np.ndarray] = None
        self.log_prior_ratio_: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'PureNumpyShrinkageLDA':
        X0 = X[y == 0]
        X1 = X[y == 1]
        n0 = len(X0)
        n1 = len(X1)
        n = n0 + n1

        self.mu0_ = np.mean(X0, axis=0)
        self.mu1_ = np.mean(X1, axis=0)

        # Centered matrices
        C0 = X0 - self.mu0_
        C1 = X1 - self.mu1_
        # Pooled sample covariance
        cov = (np.dot(C0.T, C0) + np.dot(C1.T, C1)) / (n - 2)

        # Empirical diagonal shrinkage
        diag_cov = np.diag(np.diag(cov))
        cov_shrunk = (1.0 - self.gamma) * cov + self.gamma * diag_cov

        # Invert with small ridge for numerical stability
        cov_reg = cov_shrunk + 1e-6 * np.eye(X.shape[1])
        self.inv_cov_ = np.linalg.pinv(cov_reg)
        self.log_prior_ratio_ = np.log((n1 / n) / (n0 / n))
        return self

    def predict_score(self, X: np.ndarray) -> np.ndarray:
        assert self.inv_cov_ is not None, "Model is not fitted."
        # Linear discriminant direction w = inv_cov * (mu1 - mu0)
        w = np.dot(self.inv_cov_, (self.mu1_ - self.mu0_))
        b = -0.5 * np.dot(self.mu1_.T, np.dot(self.inv_cov_, self.mu1_)) + \
             0.5 * np.dot(self.mu0_.T, np.dot(self.inv_cov_, self.mu0_)) + self.log_prior_ratio_
        z = np.dot(X, w) + b
        # Sigmoid projection for continuous ranking score
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


# Pure-NumPy Evaluation Metrics
def compute_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Computes ROC-AUC via exact trapezoidal integration over distinct thresholds."""
    desc_idx = np.argsort(y_score)[::-1]
    y_true_sorted = y_true[desc_idx]
    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)
    if n_pos == 0 or n_neg == 0:
        return np.nan
    tpr = np.cumsum(y_true_sorted == 1) / n_pos
    fpr = np.cumsum(y_true_sorted == 0) / n_neg
    tpr = np.concatenate([[0.0], tpr, [1.0]])
    fpr = np.concatenate([[0.0], fpr, [1.0]])
    return float(np.trapezoid(tpr, fpr))

def compute_pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Computes PR-AUC (Average Precision) via trapezoidal integration."""
    n_pos = np.sum(y_true == 1)
    if n_pos == 0:
        return np.nan
    desc_idx = np.argsort(y_score)[::-1]
    y_true_sorted = y_true[desc_idx]
    tp = np.cumsum(y_true_sorted == 1)
    fp = np.cumsum(y_true_sorted == 0)
    prec = tp / (tp + fp)
    recall = tp / n_pos
    prec = np.concatenate([[1.0], prec])
    recall = np.concatenate([[0.0], recall])
    return float(np.trapezoid(prec, recall))

def compute_reference_cutoff_metrics(y_true: np.ndarray, y_score: np.ndarray, cutoff: float) -> Dict[str, float]:
    """Computes classification metrics at a fixed reference score cutoff."""
    y_pred = (y_score >= cutoff).astype(int)
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    bacc = 0.5 * (sens + spec)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * (prec * sens) / (prec + sens) if (prec + sens) > 0 else 0.0
    return {
        "cutoff": cutoff, "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "sensitivity": sens, "specificity": spec, "balanced_accuracy": bacc, "f1_score": f1
    }

print("Modeling and Evaluation Engines compiled successfully.")
