### Section 2 & 3: Upstream Data Contracts, Feature Stores & Cohort Partitioning

We ingest the signed-off Stage 08 feature stores and the frozen Stage 09 patient validation splits.
- Invariant 1: Unconditional drop of 3 dead columns (`*_raw_detector_rr_mean_ms`) containing 160/160 nulls.
- Invariant 2: Explicit separation of the 2 unlabelled holdouts (`sub_005`, `sub_049`).
- Invariant 3: Explicit partitioning of the 78 CV-eligible subjects into the matched dual-rate complete sub-cohort ($N=62$) and the 100 Hz stress-skipped sub-cohort ($N=16$).
