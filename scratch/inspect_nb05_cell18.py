import json

with open("notebooks/05_rpeak_detection.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

print("=== CELL 18 ===")
print("".join(nb["cells"][18]["source"])[:800])

