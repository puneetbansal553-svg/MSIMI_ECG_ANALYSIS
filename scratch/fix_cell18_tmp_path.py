import json
from pathlib import Path

nb_path = Path("notebooks/07_beat_window_segmentation.ipynb")
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

c18_source = nb["cells"][18]["source"]
target = 'tmp_fig_path = figure_morphology_path.with_suffix(".png.tmp")'
replacement = '    tmp_fig_path = figure_morphology_path.with_suffix(".tmp.png")\n'

found = False
for idx, line in enumerate(c18_source):
    if target in line:
        c18_source[idx] = replacement
        found = True
        print(f"Replaced line {idx}: {line.strip()} -> {replacement.strip()}")
        break

assert found, f"Target '{target}' not found in Cell 18!"

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("Notebook 07 Cell 18 updated successfully.")

