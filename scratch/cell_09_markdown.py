### Section 6, 7 & 8: Predeclared Modeling Portfolio & Evaluation Metrics

We specify three models implemented in pure NumPy:
1. **Model 0 (Exploratory ST-Deviation Rank Benchmark)**: Non-parametric reference ranking patients by raw peak stress ST depression (`-stress_min_st60_global_adc`). Strictly an exploratory benchmark, not a validated clinical rule.
2. **Model 1 (Cost-Sensitive Ridge Logistic Regression)**:
   - Objective:
     $$\min_{\mathbf{w}, b} \left\{ -\sum_{j \in \text{train}} w_{y_j} \left[ y_j \log \sigma(\mathbf{w}^T \mathbf{x}_j + b) + (1 - y_j) \log (1 - \sigma(\mathbf{w}^T \mathbf{x}_j + b)) \right] + \frac{\lambda}{2} \|\mathbf{w}\|_2^2 \right\}$$
   - Predeclared Regularization: Fixed $\lambda = 1.0$ (no validation tuning).
   - Training Class Weights: $w_c = \frac{N_{\text{total\_train}}}{2 \cdot N_c}$ ($w_1 \approx 4.43$ or $3.94$, $w_0 \approx 0.56$ or $0.57$).
   - Output Semantics: Uncalibrated discriminative ranking scores (not calibrated clinical probabilities).
3. **Model 2 (Shrinkage Linear Discriminant Analysis - LDA)**:
   - Empirical diagonal shrinkage $\boldsymbol{\Sigma}_{\text{shrunk}} = (1 - \gamma)\hat{\boldsymbol{\Sigma}} + \gamma \operatorname{diag}(\hat{\boldsymbol{\Sigma}})$ with $\gamma = 0.20$.
