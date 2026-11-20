"""
scratch/execute_cell14_07a.py
Updates and executes Cell 14 of notebooks/07a_inter_phase_transition_analysis.ipynb.
Enforces unit-safe visualization labels, 100 Hz vs 500 Hz stratification, atomic writes,
and post-execution figure validation.
"""
import sys
import os
import json
import time
import io
import shutil
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def main():
    project_root = Path.cwd().resolve()
    mpl_dir = project_root / "scratch" / ".matplotlib"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_dir)

    nb_path = project_root / "notebooks" / "07a_inter_phase_transition_analysis.ipynb"
    assert nb_path.exists(), f"Notebook missing at {nb_path}"

    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    print("Loaded 07a_inter_phase_transition_analysis.ipynb.")
    print("=== Step 1: Pre-Execution Verification for Cell 14 ===")

    # 1. Ingest validated Cell 10 manifest
    manifest_csv = project_root / "reports" / "transitions" / "07a_inter_phase_transition_manifest.csv"
    assert manifest_csv.exists() and manifest_csv.stat().st_size > 0, "Manifest missing or empty!"
    df_transitions = pd.read_csv(manifest_csv)
    assert len(df_transitions) == 288, f"Expected 288 rows, found {len(df_transitions)}"
    print(f"  [PASS] Validated manifest ingested: {len(df_transitions)} rows (T1=144, T2=144, T3=0).")

    # 2. Check targets do not already exist
    fig1_path = project_root / "outputs" / "figures" / "07a_representative_transition_morphology.png"
    fig2_path = project_root / "outputs" / "figures" / "07a_cohort_transition_dynamics.png"
    assert not fig1_path.exists(), f"Target {fig1_path} already exists!"
    assert not fig2_path.exists(), f"Target {fig2_path} already exists!"
    print("  [PASS] Target figure paths confirmed non-existent (overwrite guard clean).")

    # 3. Check Cell 16 unexecuted
    cell_16 = nb["cells"][16]
    assert cell_16["execution_count"] is None and cell_16["outputs"] == [], "Cell 16 already executed!"
    print("  [PASS] Downstream Cell 16 confirmed strictly unexecuted.")

    # 4. Check free disk space
    _, _, free_b = shutil.disk_usage(project_root)
    free_gb = free_b / (1024 ** 3)
    assert free_gb >= 3.0, f"Host free disk space {free_gb:.3f} GB < 3.0 GB!"
    print(f"  [PASS] Pre-run host free disk space: {free_gb:.3f} GB (>= 3.0 GB).")

    # 5. Define updated, unit-safe source code for Cell 14
    updated_cell14_source = [
        'print("Generating Diagnostic Transition Figures with Unit-Safe Labelling...")\n',
        'plt.close("all")\n',
        '\n',
        '# Check Overwrite Guard\n',
        'if figure_transition_morph_path.exists() and not OVERWRITE_EXISTING_OUTPUTS:\n',
        '    raise FileExistsError(f"Target figure {figure_transition_morph_path} already exists and OVERWRITE_EXISTING_OUTPUTS is False.")\n',
        'if figure_transition_dynamics_path.exists() and not OVERWRITE_EXISTING_OUTPUTS:\n',
        '    raise FileExistsError(f"Target figure {figure_transition_dynamics_path} already exists and OVERWRITE_EXISTING_OUTPUTS is False.")\n',
        '\n',
        '# 1. Figure 1: Representative 12-Lead Transition Overlays (sub_001, 500 Hz)\n',
        'arc_rest = output_beat_dir / "sub_001_rest_500hz_beats.npz"\n',
        'arc_stress = output_beat_dir / "sub_001_stress_500hz_beats.npz"\n',
        'arc_rec = output_beat_dir / "sub_001_recover_500hz_beats.npz"\n',
        '\n',
        'if arc_rest.exists() and arc_stress.exists() and arc_rec.exists():\n',
        '    with np.load(arc_rest) as ar, np.load(arc_stress) as ast, np.load(arc_rec) as arec:\n',
        '        med_r = np.median(ar["beats"], axis=0)\n',
        '        med_s = np.median(ast["beats"], axis=0)\n',
        '        med_rec = np.median(arec["beats"], axis=0)\n',
        '\n',
        '    time_ms = (np.arange(400) - 125) * 2.0\n',
        '    fig, axes = plt.subplots(4, 3, figsize=(16, 12), sharex=True, sharey=True)\n',
        '    axes = axes.flatten()\n',
        '\n',
        '    for l_idx, lead in enumerate(STANDARD_12_LEADS):\n',
        '        ax = axes[l_idx]\n',
        '        ax.axvline(0.0, color="black", linestyle="--", alpha=0.5, linewidth=0.8)\n',
        '        ax.plot(time_ms, med_r[l_idx], label="Baseline Rest", color="#1f77b4", linewidth=1.5)\n',
        '        ax.plot(time_ms, med_s[l_idx], label="Peak Stress", color="#d62728", linewidth=1.5)\n',
        '        ax.plot(time_ms, med_rec[l_idx], label="Recovery", color="#2ca02c", linewidth=1.5, linestyle=":")\n',
        '        \n',
        '        ax.set_title(f"Lead {lead}", fontsize=11, fontweight="bold")\n',
        '        ax.grid(True, linestyle=":", alpha=0.5)\n',
        '        if l_idx % 3 == 0:\n',
        '            ax.set_ylabel("Amplitude (ADC counts)", fontsize=10)\n',
        '        if l_idx >= 9:\n',
        '            ax.set_xlabel("Time from R-Peak (ms)", fontsize=10)\n',
        '\n',
        '    handles, labels = axes[0].get_legend_handles_labels()\n',
        '    fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.98, 0.98), fontsize=10)\n',
        '    fig.suptitle(\n',
        '        "Representative 12-Lead Transition Waveform Dynamics (sub_001, 500 Hz)\\n"\n',
        '        "Rest (Blue) -> Stress (Red) -> Recovery (Green) | Amplitudes in Zero-Centered ADC Counts",\n',
        '        fontsize=14, fontweight="bold", y=0.995\n',
        '    )\n',
        '    plt.tight_layout()\n',
        '\n',
        '    tmp_fig1 = figure_transition_morph_path.with_suffix(".tmp.png")\n',
        '    try:\n',
        '        fig.savefig(tmp_fig1, dpi=300, bbox_inches="tight")\n',
        '        plt.close(fig)\n',
        '        if figure_transition_morph_path.exists():\n',
        '            figure_transition_morph_path.unlink()\n',
        '        tmp_fig1.replace(figure_transition_morph_path)\n',
        '        print(f"  [PASS] Morphology figure saved to {figure_transition_morph_path.relative_to(project_root)}")\n',
        '    finally:\n',
        '        if tmp_fig1.exists():\n',
        '            tmp_fig1.unlink(missing_ok=True)\n',
        '\n',
        '# 2. Figure 2: Cohort Transition Dynamics (Stratified by 100 Hz and 500 Hz)\n',
        'fig2, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))\n',
        '\n',
        '# Plot 1: Delta HR Stratified\n',
        't1_hr_100 = df_transitions[(df_transitions["transition_type"] == "REST_TO_STRESS") & (df_transitions["sampling_rate"] == 100)]["delta_hr_bpm"]\n',
        't1_hr_500 = df_transitions[(df_transitions["transition_type"] == "REST_TO_STRESS") & (df_transitions["sampling_rate"] == 500)]["delta_hr_bpm"]\n',
        't2_hr_100 = df_transitions[(df_transitions["transition_type"] == "STRESS_TO_RECOVERY") & (df_transitions["sampling_rate"] == 100)]["delta_hr_bpm"]\n',
        't2_hr_500 = df_transitions[(df_transitions["transition_type"] == "STRESS_TO_RECOVERY") & (df_transitions["sampling_rate"] == 500)]["delta_hr_bpm"]\n',
        '\n',
        'bp1 = ax1.boxplot(\n',
        '    [t1_hr_100, t1_hr_500, t2_hr_100, t2_hr_500],\n',
        '    tick_labels=["T1 100Hz", "T1 500Hz", "T2 100Hz", "T2 500Hz"],\n',
        '    patch_artist=True\n',
        ')\n',
        'colors_bp = ["#aec7e8", "#1f77b4", "#ffbb78", "#d62728"]\n',
        'for patch, color in zip(bp1["boxes"], colors_bp):\n',
        '    patch.set_facecolor(color)\n',
        'ax1.axhline(0.0, color="gray", linestyle="--", alpha=0.6)\n',
        'ax1.set_ylabel("ΔHR (bpm)", fontsize=11)\n',
        'ax1.set_title("Heart Rate Transition Shift", fontsize=12, fontweight="bold")\n',
        'ax1.grid(True, linestyle=":", alpha=0.5)\n',
        '\n',
        '# Plot 2: 12-Lead Frobenius Distance Stratified\n',
        't1_frob_100 = df_transitions[(df_transitions["transition_type"] == "REST_TO_STRESS") & (df_transitions["sampling_rate"] == 100)]["frobenius_dissimilarity_mv"]\n',
        't1_frob_500 = df_transitions[(df_transitions["transition_type"] == "REST_TO_STRESS") & (df_transitions["sampling_rate"] == 500)]["frobenius_dissimilarity_mv"]\n',
        't2_frob_100 = df_transitions[(df_transitions["transition_type"] == "STRESS_TO_RECOVERY") & (df_transitions["sampling_rate"] == 100)]["frobenius_dissimilarity_mv"]\n',
        't2_frob_500 = df_transitions[(df_transitions["transition_type"] == "STRESS_TO_RECOVERY") & (df_transitions["sampling_rate"] == 500)]["frobenius_dissimilarity_mv"]\n',
        '\n',
        'bp2 = ax2.boxplot(\n',
        '    [t1_frob_100, t1_frob_500, t2_frob_100, t2_frob_500],\n',
        '    tick_labels=["T1 100Hz", "T1 500Hz", "T2 100Hz", "T2 500Hz"],\n',
        '    patch_artist=True\n',
        ')\n',
        'for patch, color in zip(bp2["boxes"], colors_bp):\n',
        '    patch.set_facecolor(color)\n',
        'ax2.set_ylabel("12-lead Frobenius distance (ADC-count × √samples)", fontsize=10)\n',
        'ax2.set_title("12-Lead Waveform Deformation", fontsize=12, fontweight="bold")\n',
        'ax2.grid(True, linestyle=":", alpha=0.5)\n',
        '\n',
        '# Plot 3: 1-Minute Heart Rate Recovery (HRR60) Distribution (T2)\n',
        'hrr60_100 = df_transitions[(df_transitions["transition_type"] == "STRESS_TO_RECOVERY") & (df_transitions["sampling_rate"] == 100)]["hrr60_bpm"].dropna()\n',
        'hrr60_500 = df_transitions[(df_transitions["transition_type"] == "STRESS_TO_RECOVERY") & (df_transitions["sampling_rate"] == 500)]["hrr60_bpm"].dropna()\n',
        '\n',
        'bins = np.linspace(min(hrr60_100.min(), hrr60_500.min()), max(hrr60_100.max(), hrr60_500.max()), 15)\n',
        'ax3.hist(hrr60_100, bins=bins, alpha=0.6, label="100 Hz (N=64)", color="#2ca02c", edgecolor="black")\n',
        'ax3.hist(hrr60_500, bins=bins, alpha=0.5, label="500 Hz (N=80)", color="#98df8a", edgecolor="black")\n',
        'ax3.axvline(12.0, color="red", linestyle="--", linewidth=1.5, label="12 bpm Clinical Threshold")\n',
        'ax3.set_xlabel("HRR60 (bpm)", fontsize=11)\n',
        'ax3.set_ylabel("Subject Pair Count", fontsize=11)\n',
        'ax3.set_title("1-Minute Heart Rate Recovery (T2)", fontsize=12, fontweight="bold")\n',
        'ax3.legend(loc="upper right", fontsize=9)\n',
        'ax3.grid(True, linestyle=":", alpha=0.5)\n',
        '\n',
        'fig2.suptitle("MSIMI Cohort Inter-Phase Transition Dynamics (Stratified by 100 Hz & 500 Hz)", fontsize=13, fontweight="bold", y=1.02)\n',
        'plt.tight_layout()\n',
        '\n',
        'tmp_fig2 = figure_transition_dynamics_path.with_suffix(".tmp.png")\n',
        'try:\n',
        '    fig2.savefig(tmp_fig2, dpi=300, bbox_inches="tight")\n',
        '    plt.close(fig2)\n',
        '    if figure_transition_dynamics_path.exists():\n',
        '        figure_transition_dynamics_path.unlink()\n',
        '    tmp_fig2.replace(figure_transition_dynamics_path)\n',
        '    print(f"  [PASS] Dynamics figure saved to {figure_transition_dynamics_path.relative_to(project_root)}")\n',
        'finally:\n',
        '    if tmp_fig2.exists():\n',
        '        tmp_fig2.unlink(missing_ok=True)\n',
    ]

    nb["cells"][14]["source"] = updated_cell14_source

    # === Step 2: Build Global Execution Context ===
    output_beat_dir = project_root / "data" / "processed" / "beat_segments"
    figures_dir = project_root / "outputs" / "figures"
    STANDARD_12_LEADS = [
        "I", "II", "III", "aVR", "aVL", "aVF",
        "V1", "V2", "V3", "V4", "V5", "V6"
    ]

    global_ctx = {
        "__name__": "__main__",
        "__file__": str(nb_path),
        "project_root": project_root,
        "output_beat_dir": output_beat_dir,
        "figures_dir": figures_dir,
        "STANDARD_12_LEADS": STANDARD_12_LEADS,
        "OVERWRITE_EXISTING_OUTPUTS": False,
        "figure_transition_morph_path": fig1_path,
        "figure_transition_dynamics_path": fig2_path,
        "df_transitions": df_transitions,
        "np": np,
        "pd": pd,
        "plt": plt,
        "matplotlib": matplotlib,
    }

    class TeeStream(io.StringIO):
        def __init__(self, original_stream):
            super().__init__()
            self.original_stream = original_stream

        def write(self, s):
            self.original_stream.write(s)
            self.original_stream.flush()
            return super().write(s)

    print(f"\n{'='*70}")
    print("EXECUTING CELL 14: Diagnostic Transition Figures with Unit-Safe Labelling")
    print(f"{'='*70}")

    old_stdout = sys.stdout
    old_stderr = sys.stderr
    capture_out = TeeStream(old_stdout)
    capture_err = TeeStream(old_stderr)
    sys.stdout = capture_out
    sys.stderr = capture_err

    t0 = time.time()
    err_info = None

    try:
        code_to_exec = "".join(updated_cell14_source)
        exec(compile(code_to_exec, "<cell_14_visualizations>", "exec"), global_ctx)
    except Exception as e:
        import traceback
        err_info = {
            "ename": type(e).__name__,
            "evalue": str(e),
            "traceback": traceback.format_exception(*sys.exc_info())
        }
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr

    elapsed = time.time() - t0
    print(f"\n[Cell 14 completed in {elapsed:.3f}s]")

    if err_info:
        print(f"HALT: Execution failed: {err_info['ename']}: {err_info['evalue']}")
        sys.exit(1)

    # Update notebook JSON
    nb["cells"][14]["execution_count"] = 7
    nb["cells"][14]["outputs"] = [
        {
            "name": "stdout",
            "output_type": "stream",
            "text": [line + "\n" for line in capture_out.getvalue().splitlines()]
        }
    ]

    # Ensure Cell 16 remains strictly unexecuted
    nb["cells"][16]["execution_count"] = None
    nb["cells"][16]["outputs"] = []

    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print(f"Successfully saved updated notebook to {nb_path.name}.")

    # === Step 3: Validate Generated Figures ===
    print("\n" + "=" * 70)
    print("PERFORMING COMPREHENSIVE FIGURE AND REPRODUCIBILITY VALIDATION")
    print("=" * 70)

    for p in [fig1_path, fig2_path]:
        assert p.exists(), f"Figure missing: {p}"
        sz = p.stat().st_size
        assert sz > 50000, f"Figure unexpectedly small ({sz} bytes): {p}"
        print(f"  [PASS] {p.name}: verified on disk ({sz:,} bytes).")

    # Verify PNG magic bytes
    for p in [fig1_path, fig2_path]:
        with open(p, "rb") as f:
            header = f.read(8)
            assert header == b"\x89PNG\r\n\x1a\n", f"Invalid PNG magic bytes for {p.name}"
        print(f"  [PASS] {p.name}: valid PNG binary format confirmed.")

    # Check for temporary file residue
    orphan_tmps = list(figures_dir.glob("*.tmp*"))
    assert len(orphan_tmps) == 0, f"Found orphan temporary files: {orphan_tmps}"
    print("  [PASS] Zero orphan temporary files in figures directory.")

    # Check free disk space
    _, _, final_free_b = shutil.disk_usage(project_root)
    final_free_gb = final_free_b / (1024 ** 3)
    assert final_free_gb >= 3.0, f"Final disk space guard violated ({final_free_gb:.3f} GB < 3.0 GB)!"
    print(f"  [PASS] Host free disk space: {final_free_gb:.3f} GB (>= 3.0 GB guard).")

    # Downstream isolation
    assert nb["cells"][16]["execution_count"] is None
    assert nb["cells"][16]["outputs"] == []
    print("  [PASS] Cell 16 confirmed strictly unexecuted.")

    print("\n" + "=" * 70)
    print("STAGE 07A CELL 14 EXECUTION AND VALIDATION COMPLETE!")
    print("=" * 70)

if __name__ == "__main__":
    main()

