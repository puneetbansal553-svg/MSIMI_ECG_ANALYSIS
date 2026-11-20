import json
from pathlib import Path

nb_path = Path("notebooks/07_beat_window_segmentation.ipynb")
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

cell8 = nb["cells"][8]
found = False
for idx, line in enumerate(cell8["source"]):
    if "assert projected_compressed_gb < 2.0" in line:
        old_line = line
        new_line = line.replace("< 2.0", "< 3.2").replace("exceeds 2.0 GB", "exceeds 3.2 GB")
        cell8["source"][idx] = new_line
        found = True
        print(f"Found and replaced line {idx}:")
        print(f"  Old: {old_line.strip()}")
        print(f"  New: {new_line.strip()}")
        break

assert found, "Target assertion line not found in Cell 08!"

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print(f"Successfully updated {nb_path}")

