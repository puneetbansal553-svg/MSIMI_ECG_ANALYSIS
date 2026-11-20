import json

with open("notebooks/06_rr_hrv_analysis.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

print("=== CELL 14 ===")
print("".join(nb["cells"][14]["source"])[:600])

print("=== CELL 16 ===")
print("".join(nb["cells"][16]["source"])[:600])

print("=== CELL 18 ===")
print("".join(nb["cells"][18]["source"])[:600])

