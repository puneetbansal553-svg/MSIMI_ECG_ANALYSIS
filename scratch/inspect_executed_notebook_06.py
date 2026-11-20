import json
from pathlib import Path

nb_path = Path("notebooks/06_rr_hrv_analysis.ipynb")
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

cells = nb.get("cells", [])
print(f"Total cells: {len(cells)}")

for i, cell in enumerate(cells):
    cell_type = cell.get("cell_type")
    exec_count = cell.get("execution_count")
    has_outputs = len(cell.get("outputs", [])) > 0
    print(f"Cell {i} ({cell_type}): exec_count={exec_count}, num_outputs={len(cell.get('outputs', []))}")
    if has_outputs:
        for out_idx, out in enumerate(cell.get("outputs", [])):
            out_type = out.get("output_type")
            text = out.get("text", "")
            if isinstance(text, list):
                text = "".join(text)
            print(f"   Output {out_idx} ({out_type}): {text[:200]}...")

print("\n" + "="*50)
# Inspect Cell 8
print("--- CELL 8 SOURCE ---")
print("".join(cells[8].get("source", [])))
print("--- CELL 8 OUTPUTS ---")
print(json.dumps(cells[8].get("outputs", []), indent=2))

print("\n" + "="*50)
# Inspect Cell 12
print("--- CELL 12 SOURCE ---")
print("".join(cells[12].get("source", [])))
print("--- CELL 12 OUTPUTS ---")
print(json.dumps(cells[12].get("outputs", []), indent=2))

print("\n" + "="*50)
# Inspect Cell 16
print("--- CELL 16 SOURCE ---")
print("".join(cells[16].get("source", [])))
print("--- CELL 16 OUTPUTS ---")
print(json.dumps(cells[16].get("outputs", []), indent=2))

print("\n" + "="*50)
# Inspect Cell 20
print("--- CELL 20 SOURCE ---")
print("".join(cells[20].get("source", [])))
print("--- CELL 20 OUTPUTS ---")
print(json.dumps(cells[20].get("outputs", []), indent=2))

