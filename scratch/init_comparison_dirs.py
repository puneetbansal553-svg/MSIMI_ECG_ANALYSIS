from pathlib import Path

base_dir = Path("COMPARISON")
subdirs = [
    "01_DATA_INVENTORY",
    "02_COHORT_QC_COMPARISON",
    "03_PREPROCESSING_COMPARISON",
    "04_RPEAK_HRV_COMPARISON",
    "05_REST_STRESS_RECOVERY",
    "06_TRANSITION_COMPARISON",
    "07_FEATURE_COMPARISON",
    "08_VALIDATION_COMPARISON",
    "09_MODEL_PERFORMANCE",
    "10_LITERATURE_COMPARISON",
    "11_FINAL_REPORT",
    "figures",
    "tables",
    "code"
]

for sub in subdirs:
    p = base_dir / sub
    p.mkdir(parents=True, exist_ok=True)
    print(f"Created directory: {p}")

print("\nAll comparison directories initialized successfully.")

