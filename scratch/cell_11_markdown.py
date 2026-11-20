### Section 9 & 10: Repeated Patient-Level Cross-Validation Protocol

**Orchestration Invariants**:
1. Frozen Stage 09 Splits: Strictly iterates over the pre-generated 25 fold assignments (`repeat_0_fold` through `repeat_4_fold`).
2. Dual Population Evaluation:
   - **Primary Matched Comparison ($N=64$ Total / $N=62$ Labelled)**: Evaluates 100 Hz vs 500 Hz models on the exact same 62 subjects ($P=7, N=55$).
   - **Secondary Full-Cohort Analysis ($N=80$ Total / $N=78$ Labelled)**: Evaluates 500 Hz across all 78 labelled subjects ($P=9, N=69$).
3. Pooled Out-of-Fold (OOF) Assembly: Predictions across the 5 validation folds of each repeat are pooled to reconstruct the full cohort array before calculating continuous PR-AUC and ROC-AUC.
