import json
from pathlib import Path

nb_p = Path("notebooks/06_rr_hrv_analysis.ipynb")
with open(nb_p, "r", encoding="utf-8") as f:
    nb = json.load(f)

code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
print(f"Total code cells: {len(code_cells)}")
for idx, c in enumerate(code_cells):
    ec = c.get("execution_count")
    num_out = len(c.get("outputs", []))
    print(f"  Code cell {idx}: execution_count={ec}, outputs={num_out}")
    assert ec is not None, f"Cell {idx} has None execution_count"
    assert num_out > 0, f"Cell {idx} has empty outputs"

print("[PASS] All 10 code cells have non-null execution counts and valid outputs!")

