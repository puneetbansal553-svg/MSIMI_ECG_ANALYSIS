import json

with open("notebooks/04_ecg_preprocessing.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

print("=== CELL 08 ===")
print("".join(nb["cells"][8]["source"])[:1200])

print("=== CELL 14 ===")
print("".join(nb["cells"][14]["source"])[:1200])

