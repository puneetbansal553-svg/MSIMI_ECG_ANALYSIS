import sys
import os
import shutil
import json
import hashlib
import time
import math
import gc
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Pure NumPy deterministic implementation (zero external scikit-learn dependency)

# 1. Project Root & Working Directory Resolution
project_root = Path.cwd()
if not (project_root / "configs").exists() and (project_root.parent / "configs").exists():
    project_root = project_root.parent
os.chdir(project_root)
print(f"Working Directory set to: {project_root}")
assert (project_root / "configs").exists(), f"Configuration directory missing at {project_root / 'configs'}"

# Matplotlib cache isolation
mpl_dir = project_root / "scratch" / ".matplotlib"
mpl_dir.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(mpl_dir)

# 2. Overwrite Protection Guard (Default: False)
OVERWRITE_EXISTING_OUTPUTS = False

# 3. Path Definitions
data_dir = project_root / "data"
features_dir = data_dir / "features"
manifests_dir = data_dir / "manifests"
reports_features_dir = project_root / "reports" / "features"
reports_validation_dir = project_root / "reports" / "validation"
reports_models_dir = project_root / "reports" / "models"
figures_dir = project_root / "outputs" / "figures"

# Ensure output directories exist
reports_models_dir.mkdir(parents=True, exist_ok=True)
figures_dir.mkdir(parents=True, exist_ok=True)

# Stage 10 Target Outputs
metrics_csv_path = reports_models_dir / "10_model_evaluation_metrics.csv"
predictions_csv_path = reports_models_dir / "10_patient_predictions.csv"
governance_summary_json_path = reports_models_dir / "10_modeling_governance_summary.json"
figure_roc_prc_path = figures_dir / "10_roc_prc_benchmark_curves.png"
figure_calibration_path = figures_dir / "10_patient_prediction_distribution_and_calibration.png"

# 4. Check for Existing Outputs & Idempotent Guard
target_outputs = [
    metrics_csv_path,
    predictions_csv_path,
    governance_summary_json_path,
    figure_roc_prc_path,
    figure_calibration_path,
]
existing_outputs = [p for p in target_outputs if p.exists()]
missing_outputs = [p for p in target_outputs if not p.exists()]

if not OVERWRITE_EXISTING_OUTPUTS:
    if len(existing_outputs) == len(target_outputs):
        print("[REUSE] All 5 Stage 10 output artifacts already exist and are protected. Reusing verified outputs.")
    elif len(existing_outputs) > 0:
        msg_lines = [
            f"INCOMPLETE ARTIFACT SET: Found {len(existing_outputs)} of {len(target_outputs)} expected Stage 10 outputs.",
            "Existing outputs detected:"
        ]
        for ep in existing_outputs:
            msg_lines.append(f"  - {ep.relative_to(project_root)}")
        msg_lines.append("Missing outputs:")
        for mp in missing_outputs:
            msg_lines.append(f"  - {mp.relative_to(project_root)}")
        msg_lines.append("Halting safely to prevent inconsistent state.")
        raise RuntimeError("\n".join(msg_lines))
    else:
        print("[GUARD ACTIVE] OVERWRITE_EXISTING_OUTPUTS = False. No conflicting outputs found. Safe to proceed.")
else:
    print("[GUARD OVERRIDE] OVERWRITE_EXISTING_OUTPUTS = True. Existing outputs will be overwritten.")

# 5. Pre-Run Available Disk Space Check (Minimum 3.0 GB Required)
total_bytes, used_bytes, free_bytes = shutil.disk_usage(project_root)
free_gb = free_bytes / (1024 ** 3)
print(f"Pre-run Disk Storage: {free_gb:.3f} GB free on host.")
assert free_gb >= 3.0, f"HALT: Insufficient disk space ({free_gb:.3f} GB < 3.0 GB minimum requirement)."
print("  [PASS] Pre-run disk space check passed (>= 3.0 GB).")
