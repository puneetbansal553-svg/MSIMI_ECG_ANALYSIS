import json

with open("notebooks/07a_inter_phase_transition_analysis.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

print("=== CELL 10 ===")
print("".join(nb["cells"][10]["source"])[:600])

print("=== CELL 12 ===")
print("".join(nb["cells"][12]["source"])[:600])

print("=== CELL 14 ===")
print("".join(nb["cells"][14]["source"])[:600])

