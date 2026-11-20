import json
from pathlib import Path

def print_nb_code_summary(nb_name):
    p = Path("notebooks") / nb_name
    with open(p, "r", encoding="utf-8") as f:
        nb = json.load(f)
    print(f"\n=======================================================")
    print(f"=== {nb_name} (Cells: {len(nb['cells'])}) ===")
    print(f"=======================================================")
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] == "code":
            src = "".join(c.get("source", []))
            lines = [l for l in src.split("\n") if l.strip() and not l.strip().startswith("#")]
            # find variable assignments
            assigns = []
            for l in lines:
                if "=" in l and not l.strip().startswith("if ") and not l.strip().startswith("assert "):
                    assigns.append(l.strip()[:60])
            print(f"Cell {i:02d} [Code]: {len(lines)} lines. Sample assigns: {assigns[:4]}")

for name in [
    "01_dataset_exploration.ipynb",
    "02_cohort_and_label_verification.ipynb",
    "03_ecg_quality_control.ipynb",
    "04_ecg_preprocessing.ipynb",
    "05_rpeak_detection.ipynb",
    "06_rr_hrv_analysis.ipynb",
    "07_beat_window_segmentation.ipynb",
    "07a_inter_phase_transition_analysis.ipynb",
    "11_extended_model_benchmarking.ipynb"
]:
    print_nb_code_summary(name)

