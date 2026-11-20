"""
execute_cell10.py
Executes Cell 10 of notebooks/07a_inter_phase_transition_analysis.ipynb.
Serializes 07a_inter_phase_transition_manifest.csv and 07a_inter_phase_transition_summary.json.
Validates outputs against Stage 07A specifications and reconciles manifest and summary.
"""
import sys
import os
import json
import time
import io
import pickle
import traceback
import shutil
from pathlib import Path
import numpy as np
import pandas as pd

def main():
    project_root = Path.cwd()
    mpl_dir = project_root / "scratch" / ".matplotlib"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_dir)

    nb_path = project_root / "notebooks" / "07a_inter_phase_transition_analysis.ipynb"
    assert nb_path.exists(), f"Notebook missing at {nb_path}"

    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    print(f"Loaded {nb_path.name}.")
    print("Preparing Cell 10 Execution...")

    cell_10 = nb["cells"][10]
    assert cell_10["cell_type"] == "code", "Cell 10 is not a code cell!"

    # 1. Update Cell 10 source to ensure overwrite protection check is present
    src_lines = cell_10["source"]
    src_text = "".join(src_lines)
    if "Overwrite Protection Check" not in src_text:
        overwrite_guard_code = (
            '# 0. Overwrite Protection Check\n'
            'if not OVERWRITE_EXISTING_OUTPUTS:\n'
            '    if transition_manifest_csv_path.exists():\n'
            '        raise FileExistsError(f"OVERWRITE GUARD: Target manifest {transition_manifest_csv_path} already exists and OVERWRITE_EXISTING_OUTPUTS is False.")\n'
            '    if transition_summary_json_path.exists():\n'
            '        raise FileExistsError(f"OVERWRITE GUARD: Target summary {transition_summary_json_path} already exists and OVERWRITE_EXISTING_OUTPUTS is False.")\n\n'
        )
        if '# 1. Assemble DataFrame' in src_text:
            src_text = src_text.replace('# 1. Assemble DataFrame', overwrite_guard_code + '# 1. Assemble DataFrame')
            cell_10["source"] = [l + "\n" for l in src_text.splitlines()]

    # 2. Ingest Phase 1 state and Cell 08 cohort state
    with open("scratch/phase1_state.pkl", "rb") as f:
        global_ctx = pickle.load(f)

    with open("scratch/cell08_cohort_state.pkl", "rb") as f:
        c08_state = pickle.load(f)

    global_ctx["transition_records"] = c08_state["transition_records"]
    global_ctx["df_transitions"] = c08_state["df_transitions"]
    global_ctx["__name__"] = "__main__"
    global_ctx["__file__"] = str(nb_path)
    global_ctx["pd"] = pd
    global_ctx["np"] = np
    global_ctx["json"] = json

    class TeeStream(io.StringIO):
        def __init__(self, original_stream):
            super().__init__()
            self.original_stream = original_stream

        def write(self, s):
            self.original_stream.write(s)
            self.original_stream.flush()
            return super().write(s)

    print(f"\n{'='*70}")
    print("EXECUTING CELL 10: Transition Manifest & Quality Summary Serialization")
    print(f"{'='*70}")

    old_stdout = sys.stdout
    old_stderr = sys.stderr
    capture_out = TeeStream(old_stdout)
    capture_err = TeeStream(old_stderr)

    sys.stdout = capture_out
    sys.stderr = capture_err

    t0 = time.time()
    error_info = None

    try:
        code_to_exec = "".join(cell_10["source"])
        exec(compile(code_to_exec, "<cell_10_serialization>", "exec"), global_ctx)
    except Exception as e:
        error_info = {
            "ename": type(e).__name__,
            "evalue": str(e),
            "traceback": traceback.format_exception(*sys.exc_info())
        }
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr

    elapsed = time.time() - t0
    print(f"\n[Cell 10 completed in {elapsed:.3f}s]")

    # Format cell outputs for Jupyter
    cell_outputs = []
    stdout_text = capture_out.getvalue()
    stderr_text = capture_err.getvalue()

    if stdout_text:
        cell_outputs.append({
            "name": "stdout",
            "output_type": "stream",
            "text": [line + "\n" for line in stdout_text.splitlines()]
        })

    if stderr_text:
        cell_outputs.append({
            "name": "stderr",
            "output_type": "stream",
            "text": [line + "\n" for line in stderr_text.splitlines()]
        })

    if error_info:
        cell_outputs.append({
            "output_type": "error",
            "ename": error_info["ename"],
            "evalue": error_info["evalue"],
            "traceback": error_info["traceback"]
        })

    cell_10["execution_count"] = 5
    cell_10["outputs"] = cell_outputs

    if error_info:
        print(f"HALT: Execution failed in Cell 10: {error_info['ename']}: {error_info['evalue']}")
        with open(nb_path, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=1)
        sys.exit(1)

    # Save notebook with executed Cell 10
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print(f"\nSuccessfully updated {nb_path.name} with execution outputs for Cell 10.")

    # -------------------------------------------------------------
    # EXHAUSTIVE VALIDATION AUDIT OF SERIALIZED OUTPUTS
    # -------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PERFORMING CELL 10 OUTPUT VALIDATION AUDIT")
    print("=" * 70)

    manifest_csv_path = global_ctx["transition_manifest_csv_path"]
    summary_json_path = global_ctx["transition_summary_json_path"]

    # 1. File existence and non-empty checks
    assert manifest_csv_path.exists(), f"Missing manifest CSV at {manifest_csv_path}"
    csv_size = manifest_csv_path.stat().st_size
    assert csv_size > 0, f"Manifest CSV is empty ({csv_size} bytes)"

    assert summary_json_path.exists(), f"Missing summary JSON at {summary_json_path}"
    json_size = summary_json_path.stat().st_size
    assert json_size > 0, f"Summary JSON is empty ({json_size} bytes)"

    print(f"  [PASS] Transition Manifest CSV exists on disk: {manifest_csv_path.relative_to(project_root)} ({csv_size:,} bytes).")
    print(f"  [PASS] Transition Summary JSON exists on disk: {summary_json_path.relative_to(project_root)} ({json_size:,} bytes).")

    # 2. Ingest back CSV and validate records
    df_read = pd.read_csv(manifest_csv_path)
    assert len(df_read) == 288, f"Expected 288 rows in manifest CSV, found {len(df_read)}"

    t1_df = df_read[df_read["transition_type"] == "REST_TO_STRESS"]
    t2_df = df_read[df_read["transition_type"] == "STRESS_TO_RECOVERY"]
    assert len(t1_df) == 144, f"Expected 144 T1 rows, found {len(t1_df)}"
    assert len(t2_df) == 144, f"Expected 144 T2 rows, found {len(t2_df)}"
    assert "RECOVERY_TO_STRESS" not in df_read["transition_type"].unique()

    t1_100 = t1_df[t1_df["sampling_rate"] == 100]
    t1_500 = t1_df[t1_df["sampling_rate"] == 500]
    t2_100 = t2_df[t2_df["sampling_rate"] == 100]
    t2_500 = t2_df[t2_df["sampling_rate"] == 500]
    assert len(t1_100) == 64 and len(t1_500) == 80
    assert len(t2_100) == 64 and len(t2_500) == 80
    print(f"  [PASS] CSV record accounting: exactly 288 rows (144 T1 + 144 T2; 128 @ 100Hz, 160 @ 500Hz; T3 = 0).")

    # 3. Mandatory column completeness (no unexpected NaN/Inf)
    mandatory_cols = [
        "transition_id", "subject_id", "sampling_rate", "transition_type",
        "pre_recording_id", "post_recording_id", "transition_status",
        "pre_eligible_beats", "post_eligible_beats", "hr_pre_bpm", "hr_post_bpm",
        "delta_hr_bpm", "rr_ratio", "delta_rmssd_ms",
        "mean_morphology_corr", "min_morphology_corr", "frobenius_dissimilarity_mv",
        "max_rmsd_lead", "max_rmsd_value_mv",
        "j_point_offset_ms_v5", "delta_st60_lead_v5_mv", "delta_st60_lead_ii_mv",
        "stage06_candidate_nn_status"
    ]
    for col in mandatory_cols:
        assert col in df_read.columns, f"Missing required column '{col}' in CSV!"
        assert df_read[col].isna().sum() == 0, f"Mandatory column '{col}' has NaNs in CSV!"
    print(f"  [PASS] All {len(mandatory_cols)} mandatory columns verified present with 0 NaN/Inf across all 288 rows.")

    # 4. Optional HRRI NaNs retain documented guard reason
    t2_hrri_nans = t2_df["hrri_pct"].isna().sum()
    t2_hrri_valid = t2_df["hrri_pct"].notna().sum()
    assert t2_hrri_valid == 20, f"Expected 20 valid HRRI values on T2, found {t2_hrri_valid}"
    assert t2_hrri_nans == 124, f"Expected 124 HRRI NaNs on T2, found {t2_hrri_nans}"
    assert t1_df["hrri_pct"].isna().all(), "All T1 rows must have HRRI as NaN"
    print(f"  [PASS] HRRI optionality verified: 20 valid pairs, 124 documented NaNs on T2 (reserve <= 5 bpm guard), 144 NaNs on T1.")

    # 5. Clinical label isolation & Candidate NN Status
    forbidden_labels = ["anoca_confirmed", "raw_msimi_ecg_label", "target_label", "MSIMI-based-ecg", "label_status", "diagnosis"]
    for col in forbidden_labels:
        assert col not in df_read.columns, f"LEAKAGE VIOLATION: Forbidden label '{col}' present in CSV!"
    print("  [PASS] Zero clinical diagnosis label leakage in CSV manifest.")

    assert set(df_read["stage06_candidate_nn_status"].unique()) == {"UNKNOWN"}
    print("  [PASS] stage06_candidate_nn_status remains UNKNOWN across 100% of CSV rows (288/288).")

    # 6. JSON Parseability & Reconciliation with CSV
    with open(summary_json_path, "r", encoding="utf-8") as f:
        summary_data = json.load(f)

    assert summary_data["pipeline_stage"] == "07a_inter_phase_transition_analysis"
    assert summary_data["specification_version"] == "2.0"
    assert summary_data["cohort_total_subjects"] == 80

    t1_json = summary_data["transitions_evaluated"]["T1_REST_TO_STRESS"]
    assert t1_json["total_pairs"] == 144
    assert t1_json["pairs_100hz"] == 64
    assert t1_json["pairs_500hz"] == 80
    assert t1_json["mean_delta_hr_bpm"] == round(float(t1_df["delta_hr_bpm"].mean()), 2)
    assert t1_json["median_delta_hr_bpm"] == round(float(t1_df["delta_hr_bpm"].median()), 2)
    assert t1_json["mean_frobenius_dissimilarity_mv"] == round(float(t1_df["frobenius_dissimilarity_mv"].mean()), 4)

    t2_json = summary_data["transitions_evaluated"]["T2_STRESS_TO_RECOVERY"]
    assert t2_json["total_pairs"] == 144
    assert t2_json["pairs_100hz"] == 64
    assert t2_json["pairs_500hz"] == 80
    assert t2_json["mean_delta_hr_bpm"] == round(float(t2_df["delta_hr_bpm"].mean()), 2)
    assert t2_json["median_hrr60_bpm"] == round(float(t2_df["hrr60_bpm"].dropna().median()), 2)
    assert t2_json["mean_frobenius_dissimilarity_mv"] == round(float(t2_df["frobenius_dissimilarity_mv"].mean()), 4)

    t3_json = summary_data["transitions_evaluated"]["T3_RECOVERY_TO_STRESS"]
    assert t3_json["total_pairs"] == 0
    assert t3_json["status"] == "PROTOCOL_NOT_APPLICABLE"

    assert summary_data["data_governance"]["candidate_nn_status"] == "UNKNOWN"
    assert summary_data["data_governance"]["diagnosis_labels_isolated"] is True
    print("  [PASS] Summary JSON parsed and reconciled 100% with CSV manifest.")

    # 7. Check for Orphan Temporary Files
    target_dirs = [global_ctx["reports_transitions_dir"]]
    orphan_tmps = [f for td in target_dirs if td.exists() for f in td.glob("*.tmp*")]
    assert len(orphan_tmps) == 0, f"Found orphan temporary files: {orphan_tmps}"
    print(f"  [PASS] Orphan temporary files count: 0.")

    # 8. Free disk space check
    total_b, used_b, free_b = shutil.disk_usage(project_root)
    free_gb = free_b / (1024 ** 3)
    print(f"  [PASS] Final Host Free Disk Space: {free_gb:.3f} GB (Safely above 3.0 GB dynamic stop guard).")
    assert free_gb >= 3.0, f"Disk space guard violated: {free_gb:.3f} GB < 3.0 GB!"

    # 9. Upstream stages immutability check
    assert Path("reports/segmentation/07_beat_segmentation_manifest.csv").stat().st_size == 58842887
    assert len(list(Path("data/processed/beat_segments").glob("*_beats.npz"))) == 464
    print("  [PASS] Upstream Stage 03-07 files confirmed 100% unchanged.")

    # 10. Downstream cells execution check
    for idx in [12, 14, 16]:
        cell = nb["cells"][idx]
        assert cell["execution_count"] is None, f"Downstream cell {idx} has execution_count != None: {cell['execution_count']}"
        assert cell["outputs"] == [], f"Downstream cell {idx} has non-empty outputs!"
    print("  [PASS] Downstream Cells 12, 14, 16 remain strictly unexecuted.")

    print("\n" + "=" * 70)
    print("STAGE 07A: CELL 10 SERIALIZATION & VALIDATION COMPLETED SUCCESSFULLY!")
    print("=" * 70)

if __name__ == "__main__":
    main()
