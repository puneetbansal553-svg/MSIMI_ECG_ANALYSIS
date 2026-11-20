import json
import ast
from pathlib import Path

nb_path = Path("notebooks/02_cohort_and_label_verification.ipynb")
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

md_intro = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "### 7.2 Cohort Eligibility & Multi-Level Completeness Profile\n",
        "\n",
        "The following multi-panel publication visualization provides a rigorous audit of the cohort eligibility pipeline and clinical label architecture:\n",
        "- **Panel A (Verification Funnel)**: Depicts the step-by-step verification funnel from all 80 enrolled subjects through confirmed ANOCA status, complete 14-file data presence, and definitive ground-truth label audit.\n",
        "- **Panel B (Class Balance Architecture)**: Illustrates the exact clinical target distribution (MSIMI-Negative: $n=62, 79.5\\%$, MSIMI-Positive: $n=16, 20.5\\%$, Unspecified Holdouts: $n=2, 2.5\\%$), highlighting the severe $\\sim 1:3.88$ class imbalance.\n",
        "- **Panel C (Stratified Modality Availability)**: Compares recording modality completeness across clinical subgroups, detailing 100 Hz vs 500 Hz matched subject availability."
    ]
}

code_fig = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# =========================================================================\n",
        "# 7.2 Publication Figure: Cohort Eligibility & Completeness Waterfall\n",
        "# =========================================================================\n",
        "fig_out_dir = project_root / 'outputs' / 'figures'\n",
        "fig_out_dir.mkdir(parents=True, exist_ok=True)\n",
        "target_fig_path = fig_out_dir / '02_cohort_eligibility_and_completeness_waterfall.png'\n",
        "\n",
        "# Overwrite Guard\n",
        "OVERWRITE_FIGURES = False\n",
        "if target_fig_path.exists() and not OVERWRITE_FIGURES:\n",
        "    print(f\"[NOTICE] Figure already exists and OVERWRITE_FIGURES=False: {target_fig_path.name}\")\n",
        "else:\n",
        "    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')\n",
        "    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=300)\n",
        "\n",
        "    # Panel A: Cohort Verification & Eligibility Flow\n",
        "    ax_a = axes[0]\n",
        "    stages = [\n",
        "        'Enrolled Cohort\\n(sub_001 - sub_080)',\n",
        "        'Confirmed ANOCA\\n(100% Match)',\n",
        "        'Complete Metadata\\n(14/14 Files Present)',\n",
        "        'Definitive Target Label\\n(78 Labeled, 2 Unspec)'\n",
        "    ]\n",
        "    values = [80, 80, 80, 78]\n",
        "    bar_colors = ['#2b5c8f', '#41b6c4', '#7fcdbb', '#2ca02c']\n",
        "    bars_a = ax_a.bar(stages, values, color=bar_colors, edgecolor='black', width=0.55)\n",
        "    ax_a.set_ylabel('Number of Patients (N)', fontsize=11, fontweight='bold')\n",
        "    ax_a.set_title('A: Cohort Eligibility & Verification Funnel', fontsize=12, fontweight='bold', pad=10)\n",
        "    ax_a.set_ylim(0, 95)\n",
        "    for bar in bars_a:\n",
        "        h = bar.get_height()\n",
        "        ax_a.annotate(f'N = {h}\\n(100%)' if h == 80 else f'N = {h}\\n(97.5%)',\n",
        "                      xy=(bar.get_x() + bar.get_width()/2, h), xytext=(0, 4),\n",
        "                      textcoords=\"offset points\", ha='center', va='bottom', fontsize=9, fontweight='bold')\n",
        "\n",
        "    # Panel B: Ground Truth Target Class Balance (MSIMI vs Non-MSIMI)\n",
        "    ax_b = axes[1]\n",
        "    n_total = len(df_candidate_manifest)\n",
        "    n_pos = int((df_candidate_manifest['label_status'] == 'POSITIVE').sum())\n",
        "    n_neg = int((df_candidate_manifest['label_status'] == 'NEGATIVE').sum())\n",
        "    n_unspec = int((df_candidate_manifest['label_status'] == 'UNSPECIFIED').sum())\n",
        "\n",
        "    labels_pie = [\n",
        "        f'MSIMI-Negative\\n(N={n_neg}, {n_neg/n_total*100:.1f}%)',\n",
        "        f'MSIMI-Positive\\n(N={n_pos}, {n_pos/n_total*100:.1f}%)',\n",
        "        f'Unspecified Holdout\\n(N={n_unspec}, {n_unspec/n_total*100:.1f}%)'\n",
        "    ]\n",
        "    sizes = [n_neg, n_pos, n_unspec]\n",
        "    colors_pie = ['#3182bd', '#de2d26', '#bdbdbd']\n",
        "    explode = (0.03, 0.08, 0.05)\n",
        "\n",
        "    wedges, texts, autotexts = ax_b.pie(\n",
        "        sizes, explode=explode, labels=labels_pie, colors=colors_pie,\n",
        "        autopct='%1.1f%%', startangle=140, pctdistance=0.6,\n",
        "        wedgeprops=dict(edgecolor='black', linewidth=1.2),\n",
        "        textprops=dict(fontsize=10, fontweight='bold')\n",
        "    )\n",
        "    ax_b.set_title('B: Ground-Truth Target Label Distribution\\n(Prevalence: 20.5% in Labeled Cohort)', fontsize=12, fontweight='bold', pad=10)\n",
        "\n",
        "    # Panel C: Stratified Modality Completeness across Subgroups\n",
        "    ax_c = axes[2]\n",
        "    categories = ['MSIMI-\\n(N=62)', 'MSIMI+\\n(N=16)', 'Holdout\\n(N=2)']\n",
        "    x_c = np.arange(3)\n",
        "    w_c = 0.35\n",
        "\n",
        "    def count_available_subjects(rate, status):\n",
        "        subs = df_candidate_manifest[df_candidate_manifest['label_status'] == status]['subject_id']\n",
        "        cnt = 0\n",
        "        for s in subs:\n",
        "            if (raw_dir / f\"{s}_baseline-data_rest-ecg-{rate}hz.csv\").exists():\n",
        "                cnt += 1\n",
        "        return cnt\n",
        "\n",
        "    counts_100hz = [count_available_subjects(100, 'NEGATIVE'), count_available_subjects(100, 'POSITIVE'), count_available_subjects(100, 'UNSPECIFIED')]\n",
        "    counts_500hz = [count_available_subjects(500, 'NEGATIVE'), count_available_subjects(500, 'POSITIVE'), count_available_subjects(500, 'UNSPECIFIED')]\n",
        "\n",
        "    b1 = ax_c.bar(x_c - w_c/2, counts_100hz, w_c, label='100 Hz Reference', color='#4575b4', edgecolor='black')\n",
        "    b2 = ax_c.bar(x_c + w_c/2, counts_500hz, w_c, label='500 Hz High-Res', color='#f46d43', edgecolor='black')\n",
        "    ax_c.set_ylabel('Number of Available Patients', fontsize=11, fontweight='bold')\n",
        "    ax_c.set_title('C: Modality Availability by Clinical Subgroup', fontsize=12, fontweight='bold', pad=10)\n",
        "    ax_c.set_xticks(x_c)\n",
        "    ax_c.set_xticklabels(categories, fontsize=10, fontweight='bold')\n",
        "    ax_c.set_ylim(0, 75)\n",
        "    ax_c.legend(loc='upper right', frameon=True)\n",
        "    for bar in b1 + b2:\n",
        "        h = bar.get_height()\n",
        "        ax_c.annotate(f'{h}', xy=(bar.get_x() + bar.get_width()/2, h), xytext=(0, 3),\n",
        "                      textcoords=\"offset points\", ha='center', va='bottom', fontsize=9, fontweight='bold')\n",
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
        "### Interpretation Guide: Cohort Verification Funnel & Stratified Balance\n",
        "\n",
        "**Clinical & Methodological Significance:**\n",
        "1. **Cohort Eligibility (Panel A)**: 100% of the 80 study subjects are confirmed ANOCA patients with complete file collections. 78 subjects possess definitive binary ground truth labels (`MSIMI-based-ecg` = 0 or 1), while 2 subjects (`sub_005` and `sub_049`) lack definitive binary labels and are quarantined as unlabelled holdouts.\n",
        "2. **Class Imbalance Realities (Panel B)**: The target prevalence of MSIMI is 20.5% (16 positive vs 62 negative), reflecting an imbalance ratio of 1:3.88. Standard unweighted accuracy is clinically misleading; all evaluation must emphasize PR-AUC (Average Precision), sensitivity at fixed specificity, and balanced accuracy.\n",
        "3. **Cross-Modality Cohort Partitioning (Panel C)**: All 78 labeled subjects have complete 100 Hz recordings. A matched sub-cohort of 62 subjects (48 negative, 14 positive) also possess 500 Hz high-resolution recordings. To ensure unbiased comparative modeling, downstream benchmarks must be evaluated across both the complete 100 Hz cohort and the paired 500 Hz matched strata."
    ]
}

# Insert before Section 8 (currently Cell 13)
nb["cells"].insert(13, md_guide)
nb["cells"].insert(13, code_fig)
nb["cells"].insert(13, md_intro)

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

print(f"Successfully updated {nb_path.name}: now has {len(nb['cells'])} cells (was 14).")

