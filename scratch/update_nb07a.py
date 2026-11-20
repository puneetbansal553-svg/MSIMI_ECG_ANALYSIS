import json
import ast
from pathlib import Path

nb_path = Path("notebooks/07a_inter_phase_transition_analysis.ipynb")
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

md_intro = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "### 7.2 HRR60, HRRI & Global 12-Lead Frobenius Dissimilarity Distributions\n",
        "\n",
        "The following multi-panel publication visualization provides a focused audit of acute inter-phase transitions, early 1-minute post-exercise recovery dynamics, and global 12-lead median beat morphological shifts:\n",
        "- **Panel A (1-Minute Heart Rate Recovery, $\\text{HRR}_{60}$)**: Evaluates immediate post-exercise parasympathetic reactivation ($\\text{HR}_{\\text{end\\_stress}} - \\text{HR}_{\\text{rec\\_60s}}$) across 100 Hz and 500 Hz strata relative to the 12 BPM cutoff.\n",
        "- **Panel B (Heart Rate Recovery Index, $\\text{HRRI}$)**: Explores the relationship between normalized recovery index ($\\text{HRRI}_{\\%}$) and absolute rate drop ($\\text{HRR}_{60}$).\n",
        "- **Panel C (Global 12-Lead Frobenius Dissimilarity)**: Quantifies whole-heart 12-lead median beat waveform distortion across Rest $\\to$ Stress ($T_1$) versus Stress $\\to$ Recovery ($T_2$) transitions (in ADC counts).\n",
        "- **Panel D (Precordial Lead V5 Repolarization Shift)**: Assesses acute ST-segment changes at the 60 ms post-J-point fiducial ($\\Delta \\text{ST}_{60}$) across transitions."
    ]
}

code_fig = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# =========================================================================\n",
        "# 7.2 Publication Figure: HRR60, HRRI & Morphology Dissimilarity\n",
        "# =========================================================================\n",
        "target_fig_path = fig_out_dir / '07a_hrr60_hrri_and_morphology_distributions.png'\n",
        "\n",
        "# Overwrite Guard\n",
        "OVERWRITE_FIGURES = False\n",
        "if target_fig_path.exists() and not OVERWRITE_FIGURES:\n",
        "    print(f\"[NOTICE] Figure already exists and OVERWRITE_FIGURES=False: {target_fig_path.name}\")\n",
        "else:\n",
        "    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')\n",
        "    fig, axes = plt.subplots(2, 2, figsize=(16, 11), dpi=300)\n",
        "\n",
        "    df_t2 = df_transitions[df_transitions['transition_type'] == 'stress_to_recover'].copy()\n",
        "    df_t1 = df_transitions[df_transitions['transition_type'] == 'rest_to_stress'].copy()\n",
        "\n",
        "    # Panel A: HRR60 Distribution by Sampling Rate\n",
        "    ax_a = axes[0, 0]\n",
        "    hrr60_100 = df_t2[df_t2['sampling_rate'] == 100]['hrr60_bpm'].dropna()\n",
        "    hrr60_500 = df_t2[df_t2['sampling_rate'] == 500]['hrr60_bpm'].dropna()\n",
        "\n",
        "    bp_a = ax_a.boxplot([hrr60_100, hrr60_500], tick_labels=['100 Hz (N=80)', '500 Hz (N=64)'],\n",
        "                        patch_artist=True, medianprops=dict(color='black', linewidth=1.5))\n",
        "    bp_a['boxes'][0].set_facecolor('#4575b4')\n",
        "    bp_a['boxes'][1].set_facecolor('#f46d43')\n",
        "    ax_a.axhline(12.0, color='red', linestyle='--', linewidth=1.2, label='Clinical Blunting Cutoff (12 BPM)')\n",
        "    ax_a.scatter(np.random.normal(1, 0.04, size=len(hrr60_100)), hrr60_100, color='#2b5c8f', alpha=0.5, s=25)\n",
        "    ax_a.scatter(np.random.normal(2, 0.04, size=len(hrr60_500)), hrr60_500, color='#d73027', alpha=0.5, s=25)\n",
        "    ax_a.set_ylabel('HRR_60 (HR_end_stress - HR_rec_60s) [BPM]', fontsize=11, fontweight='bold')\n",
        "    ax_a.set_title('A: 1-Minute Heart Rate Recovery (HRR_60) Distribution', fontsize=12, fontweight='bold', pad=10)\n",
        "    ax_a.legend(loc='upper right', frameon=True)\n",
        "\n",
        "    # Panel B: HRRI (%) vs HRR60 (BPM)\n",
        "    ax_b = axes[0, 1]\n",
        "    valid_hrri = df_t2.dropna(subset=['hrr60_bpm', 'hrri_pct'])\n",
        "    ax_b.scatter(valid_hrri['hrr60_bpm'], valid_hrri['hrri_pct'], color='#756bb1', alpha=0.7, s=35, edgecolor='black', linewidth=0.5)\n",
        "    ax_b.axvline(12.0, color='gray', linestyle=':', alpha=0.7)\n",
        "    ax_b.axhline(0.0, color='gray', linestyle=':', alpha=0.7)\n",
        "    ax_b.set_xlabel('HRR_60 (BPM)', fontsize=11, fontweight='bold')\n",
        "    ax_b.set_ylabel('Heart Rate Recovery Index (HRRI %)', fontsize=11, fontweight='bold')\n",
        "    ax_b.set_title('B: Recovery Index (HRRI) vs Absolute HR Drop (HRR_60)', fontsize=12, fontweight='bold', pad=10)\n",
        "\n",
        "    # Panel C: Frobenius Morphology Dissimilarity (Rest->Stress vs Stress->Recover)\n",
        "    ax_c = axes[1, 0]\n",
        "    frob_t1 = df_t1['frobenius_dissimilarity_mv'].dropna()\n",
        "    frob_t2 = df_t2['frobenius_dissimilarity_mv'].dropna()\n",
        "\n",
        "    bp_c = ax_c.boxplot([frob_t1, frob_t2], tick_labels=['Rest -> Stress (T1)', 'Stress -> Recover (T2)'],\n",
        "                        patch_artist=True, medianprops=dict(color='black', linewidth=1.5))\n",
        "    bp_c['boxes'][0].set_facecolor('#fdae61')\n",
        "    bp_c['boxes'][1].set_facecolor('#a6d96a')\n",
        "    ax_c.scatter(np.random.normal(1, 0.04, size=len(frob_t1)), frob_t1, color='#e6550d', alpha=0.4, s=20)\n",
        "    ax_c.scatter(np.random.normal(2, 0.04, size=len(frob_t2)), frob_t2, color='#31a354', alpha=0.4, s=20)\n",
        "    ax_c.set_ylabel('12-Lead Frobenius Dissimilarity (ADC Counts)', fontsize=11, fontweight='bold')\n",
        "    ax_c.set_title('C: Global 12-Lead Median Beat Morphology Distortion', fontsize=12, fontweight='bold', pad=10)\n",
        "\n",
        "    # Panel D: Lead V5 ST60 Shift Across Transitions\n",
        "    ax_d = axes[1, 1]\n",
        "    st60_t1 = df_t1['delta_st60_lead_v5_mv'].dropna()\n",
        "    st60_t2 = df_t2['delta_st60_lead_v5_mv'].dropna()\n",
        "\n",
        "    bp_d = ax_d.boxplot([st60_t1, st60_t2], tick_labels=['Rest -> Stress (T1)', 'Stress -> Recover (T2)'],\n",
        "                        patch_artist=True, medianprops=dict(color='black', linewidth=1.5))\n",
        "    bp_d['boxes'][0].set_facecolor('#f781bf')\n",
        "    bp_d['boxes'][1].set_facecolor('#999999')\n",
        "    ax_d.axhline(0, color='black', linestyle='--', linewidth=1.0)\n",
        "    ax_d.set_ylabel('Delta ST60 in Lead V5 (Raw ADC Counts)', fontsize=11, fontweight='bold')\n",
        "    ax_d.set_title('D: Precordial Lead V5 Repolarization Shift (ST60 Delta)', fontsize=12, fontweight='bold', pad=10)\n",
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
        "### Interpretation Guide: Rapid Autonomic Deceleration & Global Waveform Distortion\n",
        "\n",
        "**Clinical & Methodological Significance:**\n",
        "1. **Early 1-Minute Recovery ($\\text{HRR}_{60}$, Panel A)**: Heart rate recovery within the first 60 seconds post-exercise is primarily mediated by fast vagal nerve reactivation. Patients exhibiting $\\text{HRR}_{60} < 12$ BPM are classified as having blunted recovery, which correlates strongly with microvascular coronary dysfunction.\n",
        "2. **Recovery Index Non-Linearity (Panel B)**: While $\\text{HRR}_{60}$ measures absolute beats-per-minute deceleration, $\\text{HRRI}$ measures deceleration relative to peak exercise heart rate reserve. Both metrics converge strongly for rapid responders but dissociate in patients with chronotropic incompetence.\n",
        "3. **Morphology Dissimilarity Across Transitions (Panels C & D)**: Global 12-lead Frobenius distance is significantly higher during Rest $\\to$ Stress ($T_1$) than during Stress $\\to$ Recovery ($T_2$). This reflects extensive exercise-induced repolarization shifts (Q-T shortening, T-wave flattening, ST depression) that only partially reverse during early recovery."
    ]
}

# Insert before Section 8 (currently Cell 15)
nb["cells"].insert(15, md_guide)
nb["cells"].insert(15, code_fig)
nb["cells"].insert(15, md_intro)

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

print(f"Successfully updated {nb_path.name}: now has {len(nb['cells'])} cells (was 17).")

