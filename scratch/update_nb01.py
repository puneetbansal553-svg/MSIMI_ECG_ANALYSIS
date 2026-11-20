import json
import ast
from pathlib import Path

nb_path = Path("notebooks/01_dataset_exploration.ipynb")
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

md_intro = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "### 5.2 Recording Completeness & Sampling Rate Stratification Profile\n",
        "\n",
        "The following multi-panel publication-grade visualization provides a comprehensive breakdown of the acquired ECG recording inventory across the study cohort:\n",
        "- **Panel A (Protocol Completeness)**: Quantifies verified recording completeness across all three protocol phases (Rest, Stress, Recovery) stratified by native sampling rate (100 Hz vs 500 Hz).\n",
        "- **Panel B (Duration Fidelity)**: Compares the empirical recording durations against the nominal protocol specifications (Rest: 360 s, Stress: 720 s, Recovery: 360 s).\n",
        "- **Panel C (Duration Variance)**: Evaluates the exact delta between acquired and nominal duration ($\\Delta t = t_{\\text{acquired}} - t_{\\text{nominal}}$) across phases.\n",
        "- **Panel D (Lead Architecture Profile)**: Details cohort-wide compliance with the standard 12-lead schema, highlighting the prevalence of synchronous 12-lead acquisitions and the specific incidence of shortened Lead V6."
    ]
}

code_fig = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# =========================================================================\n",
        "# 5.2 Publication Figure: Recording Completeness & Sampling Distribution\n",
        "# =========================================================================\n",
        "fig_out_dir = project_root / 'outputs' / 'figures'\n",
        "fig_out_dir.mkdir(parents=True, exist_ok=True)\n",
        "target_fig_path = fig_out_dir / '01_recording_completeness_and_sampling_distribution.png'\n",
        "\n",
        "# Overwrite Guard\n",
        "OVERWRITE_FIGURES = False\n",
        "if target_fig_path.exists() and not OVERWRITE_FIGURES:\n",
        "    print(f\"[NOTICE] Figure already exists and OVERWRITE_FIGURES=False: {target_fig_path.name}\")\n",
        "else:\n",
        "    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')\n",
        "    fig, axes = plt.subplots(2, 2, figsize=(15, 11), dpi=300)\n",
        "\n",
        "    palette_phase = {'rest': '#1f77b4', 'stress': '#d62728', 'recover': '#2ca02c'}\n",
        "    phase_labels = {'rest': 'Rest (360s)', 'stress': 'Stress (720s)', 'recover': 'Recovery (360s)'}\n",
        "\n",
        "    # Panel A: Recording Count per Phase stratified by Sampling Frequency\n",
        "    ax_a = axes[0, 0]\n",
        "    counts_100 = [len(df_records[(df_records['rate'] == 100) & (df_records['phase'] == p)]) for p in ['rest', 'stress', 'recover']]\n",
        "    counts_500 = [len(df_records[(df_records['rate'] == 500) & (df_records['phase'] == p)]) for p in ['rest', 'stress', 'recover']]\n",
        "    x = np.arange(3)\n",
        "    width = 0.35\n",
        "    rects1 = ax_a.bar(x - width/2, counts_100, width, label='100 Hz Reference', color='#4575b4', edgecolor='black', linewidth=0.8)\n",
        "    rects2 = ax_a.bar(x + width/2, counts_500, width, label='500 Hz High-Res', color='#f46d43', edgecolor='black', linewidth=0.8)\n",
        "    ax_a.set_ylabel('Number of Verified Recordings', fontsize=11, fontweight='bold')\n",
        "    ax_a.set_title('A: Protocol Recording Completeness (N=480 Total)', fontsize=12, fontweight='bold', pad=10)\n",
        "    ax_a.set_xticks(x)\n",
        "    ax_a.set_xticklabels(['Rest', 'Stress', 'Recover'], fontsize=11)\n",
        "    ax_a.set_ylim(0, 100)\n",
        "    ax_a.axhline(80, color='gray', linestyle='--', linewidth=1, alpha=0.7, label='Target Cohort (N=80)')\n",
        "    for bar in rects1 + rects2:\n",
        "        h = bar.get_height()\n",
        "        ax_a.annotate(f'{h}', xy=(bar.get_x() + bar.get_width()/2, h), xytext=(0, 3),\n",
        "                      textcoords=\"offset points\", ha='center', va='bottom', fontsize=9, fontweight='bold')\n",
        "    ax_a.legend(loc='lower right', frameon=True)\n",
        "\n",
        "    # Panel B: Duration Distribution Across Protocol Phases\n",
        "    ax_b = axes[0, 1]\n",
        "    box_data = [df_records[df_records['phase'] == p]['duration_sec'].dropna() for p in ['rest', 'stress', 'recover']]\n",
        "    bplot = ax_b.boxplot(box_data, tick_labels=['Rest', 'Stress', 'Recover'], patch_artist=True,\n",
        "                         medianprops=dict(color='black', linewidth=1.5))\n",
        "    colors = ['#abd9e9', '#fdae61', '#a6d96a']\n",
        "    for patch, color in zip(bplot['boxes'], colors):\n",
        "        patch.set_facecolor(color)\n",
        "        patch.set_edgecolor('black')\n",
        "    ax_b.scatter([1, 2, 3], [360, 720, 360], color='red', marker='D', s=50, zorder=5, label='Nominal Target')\n",
        "    ax_b.set_ylabel('Acquired Duration (seconds)', fontsize=11, fontweight='bold')\n",
        "    ax_b.set_title('B: Protocol Duration Fidelity by Phase', fontsize=12, fontweight='bold', pad=10)\n",
        "    ax_b.legend(loc='upper right', frameon=True)\n",
        "\n",
        "    # Panel C: Duration Delta Distribution (Acquired - Nominal)\n",
        "    ax_c = axes[1, 0]\n",
        "    dur_deltas = df_records['duration_sec'] - df_records['nominal_sec']\n",
        "    for p, color in palette_phase.items():\n",
        "        sub_deltas = dur_deltas[df_records['phase'] == p]\n",
        "        ax_c.hist(sub_deltas, bins=25, alpha=0.6, color=color,\n",
        "                  label=f\"{phase_labels[p]} (mean: {sub_deltas.mean():+.1f}s)\", edgecolor='black', linewidth=0.5)\n",
        "    ax_c.axvline(0, color='black', linestyle='--', linewidth=1.2, label='Nominal Zero Delta')\n",
        "    ax_c.set_xlabel('Duration Delta: Acquired - Nominal (seconds)', fontsize=11, fontweight='bold')\n",
        "    ax_c.set_ylabel('Recording Frequency', fontsize=11, fontweight='bold')\n",
        "    ax_c.set_title('C: Duration Variance from Protocol Specification', fontsize=12, fontweight='bold', pad=10)\n",
        "    ax_c.legend(loc='upper left', frameon=True, fontsize=9)\n",
        "\n",
        "    # Panel D: 12-Lead Acquisition Compliance & Length Uniformity\n",
        "    ax_d = axes[1, 1]\n",
        "    compliance_categories = [\n",
        "        '12-Lead Complete\\n(100% of 480)',\n",
        "        'Standard Duration\\n(98.3% of 480)',\n",
        "        'Equal Lead Lengths\\n(82.3% of 480)',\n",
        "        'Shortened Lead V6\\n(17.7% of 480)'\n",
        "    ]\n",
        "    n_total = len(df_records)\n",
        "    n_12lead = (df_records['lead_count'] == 12).sum()\n",
        "    n_std_dur = (df_records['is_dur_irregular'] != True).sum()\n",
        "    n_unequal = (df_records['has_unequal_leads'] == True).sum()\n",
        "    n_equal = n_total - n_unequal\n",
        "\n",
        "    percentages = [\n",
        "        n_12lead / n_total * 100,\n",
        "        n_std_dur / n_total * 100,\n",
        "        n_equal / n_total * 100,\n",
        "        n_unequal / n_total * 100\n",
        "    ]\n",
        "    bar_colors = ['#2ca02c', '#1f77b4', '#17becf', '#ff7f0e']\n",
        "    bars = ax_d.barh(compliance_categories[::-1], percentages[::-1], color=bar_colors[::-1], edgecolor='black', linewidth=0.8)\n",
        "    ax_d.set_xlabel('Cohort Compliance (%)', fontsize=11, fontweight='bold')\n",
        "    ax_d.set_title('D: 12-Lead Signal Architecture & Integrity Profile', fontsize=12, fontweight='bold', pad=10)\n",
        "    ax_d.set_xlim(0, 115)\n",
        "    for bar in bars:\n",
        "        w_val = bar.get_width()\n",
        "        ax_d.annotate(f'{w_val:.1f}% ({int(round(w_val * n_total / 100))}/{n_total})',\n",
        "                      xy=(w_val, bar.get_y() + bar.get_height()/2), xytext=(5, 0),\n",
        "                      textcoords=\"offset points\", ha='left', va='center', fontsize=9, fontweight='bold')\n",
        "\n",
        "    plt.tight_layout()\n",
        "    plt.savefig(target_fig_path, dpi=300, bbox_inches='tight')\n",
        "    plt.show()\n",
        "    print(f\"[SAVED] {target_fig_path.name} (300 DPI)\")\n"
    ]
}

md_guide = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "### Interpretation Guide: Protocol Integrity & Signal Architecture\n",
        "\n",
        "**Clinical & Methodological Significance:**\n",
        "1. **Acquisition Completeness (Panel A)**: All 80 ANOCA subjects have verified recording files across all 3 protocol phases for both 100 Hz and 500 Hz modalities ($N = 480$ total files). This establishes complete paired baseline coverage across the cohort.\n",
        "2. **Protocol Duration Compliance (Panels B & C)**: Over 98% of all recordings conform strictly to standard protocol lengths (Rest: 360 s, Stress: 720 s, Recovery: 360 s). Minor duration deviations are clustered within $\\pm 2$ seconds, reflecting clinical technician start/stop latencies rather than protocol non-compliance.\n",
        "3. **Shortened Lead V6 Accounting (Panel D)**: 85 recordings (17.7%) exhibit shortened sample lengths in Lead V6 (typically truncated by 1–5 seconds). Downstream segmentation and feature extraction pipelines (Stages 04, 05, 07) must implement synchronous boundary inequalities using `min_lead_samples` to prevent index out-of-bounds exceptions while preserving full 12-lead diagnostic validity."
    ]
}

# Insert before Section 6 (which is currently Cell 9)
nb["cells"].insert(9, md_guide)
nb["cells"].insert(9, code_fig)
nb["cells"].insert(9, md_intro)

# Validate AST of all code cells
for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "code":
        src = "".join(cell.get("source", []))
        try:
            ast.parse(src)
        except Exception as e:
            raise RuntimeError(f"AST parse failed in Cell {i}: {e}")

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print(f"Successfully updated {nb_path.name}: now has {len(nb['cells'])} cells (was 10).")

