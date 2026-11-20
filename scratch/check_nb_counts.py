import json

nb = json.load(open("notebooks/07a_inter_phase_transition_analysis.ipynb", "r", encoding="utf-8"))
for i, c in enumerate(nb["cells"]):
    if c["cell_type"] == "code":
        count = c.get("execution_count")
        n_out = len(c.get("outputs", []))
        print(f"Cell {i:02d}: count={count} | outputs={n_out}")

