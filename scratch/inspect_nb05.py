import json

with open("notebooks/05_rpeak_detection.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

for i, c in enumerate(nb["cells"]):
    src = "".join(c.get("source", []))
    first_line = src.strip().split("\n")[0] if src.strip() else ""
    print(f"Cell {i:02d} [{c['cell_type']}]: {first_line[:80]}")

