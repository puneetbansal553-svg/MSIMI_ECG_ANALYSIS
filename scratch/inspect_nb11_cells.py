import json

with open("notebooks/11_extended_model_benchmarking.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

print("=== CELL 19 ===")
print("".join(nb["cells"][19]["source"])[:800])

print("=== CELL 21 ===")
print("".join(nb["cells"][21]["source"])[:800])

