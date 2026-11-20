import json
import re
from pathlib import Path

nb_dir = Path("notebooks")
nbs = sorted([f for f in nb_dir.glob("*.ipynb")])

print(f"Found {len(nbs)} notebooks:")
for nb_path in nbs:
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)
    cells = nb.get("cells", [])
    code_cells = [c for c in cells if c.get("cell_type") == "code"]
    md_cells = [c for c in cells if c.get("cell_type") == "markdown"]
    
    # search for savefig or outputs/figures
    fig_refs = []
    for i, c in enumerate(cells):
        src = "".join(c.get("source", []))
        matches = re.findall(r"savefig\((.*?)\)", src)
        for m in matches:
            fig_refs.append((i, c.get("cell_type"), m.strip()))
        matches_png = re.findall(r"[\"']([^\"']*\.png)[\"']", src)
        for m in matches_png:
            if "figure" in m or "0" in m or "1" in m:
                fig_refs.append((i, c.get("cell_type"), m.strip()))
                
    fig_refs = list(dict.fromkeys(fig_refs))
    print(f"\n--- {nb_path.name} ---")
    print(f"Total cells: {len(cells)} (Code: {len(code_cells)}, Markdown: {len(md_cells)})")
    if fig_refs:
        print("Figure references:")
        for idx, ctype, r in fig_refs:
            print(f"  Cell {idx} ({ctype}): {r}")

