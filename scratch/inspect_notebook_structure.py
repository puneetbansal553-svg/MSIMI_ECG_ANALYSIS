import json
from pathlib import Path

nb_names = [
    "01_dataset_exploration.ipynb",
    "02_cohort_and_label_verification.ipynb",
    "03_ecg_quality_control.ipynb",
    "04_ecg_preprocessing.ipynb",
    "05_rpeak_detection.ipynb",
    "06_rr_hrv_analysis.ipynb",
    "07_beat_window_segmentation.ipynb",
    "07a_inter_phase_transition_analysis.ipynb",
    "11_extended_model_benchmarking.ipynb"
]

for name in nb_names:
    p = Path("notebooks") / name
    with open(p, "r", encoding="utf-8") as f:
        nb = json.load(f)
    print(f"\n==================== {name} ====================")
    for i, c in enumerate(nb["cells"]):
        src = "".join(c.get("source", []))
        first_line = src.strip().split("\n")[0] if src.strip() else "[EMPTY]"
        if c["cell_type"] == "markdown":
            lines = [l for l in src.split("\n") if l.strip().startswith("#")]
            headers = " | ".join(lines) if lines else first_line[:60]
            print(f"Cell {i:02d} [MD]: {headers}")
        else:
            print(f"Cell {i:02d} [Code]: {first_line[:70]}")

