import json
import os
from pathlib import Path

nb_path = Path(r"C:\Users\HP\Desktop\MSIMI_Project\notebooks\08_ecg_feature_extraction.ipynb")
print(f"Inspecting exact path: {nb_path}")
print(f"File exists: {nb_path.exists()}")
print(f"File size: {nb_path.stat().st_size:,} bytes")
print(f"File mtime: {nb_path.stat().st_mtime}")

with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

print(f"Total cells: {len(nb['cells'])}")
print("\n--- Searching for target phrases in all cells ---")
phrases = ["OVERWRITE GUARD ENFORCED", "raise FileExistsError", "OVERWRITE_EXISTING_OUTPUTS", "[REUSE]"]

matches = []
for idx, cell in enumerate(nb['cells']):
    src = "".join(cell.get("source", []))
    for p in phrases:
        if p in src:
            matches.append((idx, cell['cell_type'], p))
            print(f"Found '{p}' in Cell {idx:02d} ({cell['cell_type']})")

print("\n--- Inspecting Cell 02 source lines around overwrite guard ---")
c02_lines = nb['cells'][2].get("source", [])
for i, line in enumerate(c02_lines):
    if any(k in line for k in ["OVERWRITE", "existing_outputs", "FileExistsError", "REUSE", "RuntimeError"]):
        print(f"Line {i+1:02d}: {line.rstrip()}")

print("\n--- Checking for other .ipynb files in workspace ---")
project_root = Path(r"C:\Users\HP\Desktop\MSIMI_Project")
for ipynb in project_root.rglob("*.ipynb"):
    print(f"Found notebook: {ipynb.relative_to(project_root)} | {ipynb.stat().st_size:,} B | mtime: {ipynb.stat().st_mtime}")

