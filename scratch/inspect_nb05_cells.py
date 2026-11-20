import json

with open("notebooks/05_rpeak_detection.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

print("=== CELL 14 ===")
print("".join(nb["cells"][14]["source"])[:800])

print("=== CELL 16 ===")
print("".join(nb["cells"][16]["source"])[:1200])

