"""
inspect_cells_07a.py
Print outline of all 17 cells in Notebook 07A.
"""
import json
from pathlib import Path

nb_path = Path("notebooks/07a_inter_phase_transition_analysis.ipynb")
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

print(f"{'Cell':<6} {'Type':<10} {'Lines':<8} {'Title / First line'}")
print("-" * 75)
for i, cell in enumerate(nb["cells"]):
    c_type = cell["cell_type"]
    src = cell["source"]
    first_line = src[0].strip() if src else ""
    first_line = (first_line[:55] + "...") if len(first_line) > 55 else first_line
    print(f"Cell {i:<2} {c_type:<10} {len(src):<8} {first_line}")

