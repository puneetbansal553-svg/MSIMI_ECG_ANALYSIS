import json
from pathlib import Path

nb_path = Path("notebooks/07_beat_window_segmentation.ipynb")
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

c10_source = nb["cells"][10]["source"]
target_str = 'tmp_archive_path = output_beat_dir / f"{rec_id}_beats.npz.tmp"\n'
replacement_str = '    tmp_archive_path = output_beat_dir / f"{rec_id}_beats.tmp.npz"\n'

found = False
for idx, line in enumerate(c10_source):
    if 'tmp_archive_path = output_beat_dir / f"{rec_id}_beats.npz.tmp"' in line:
        c10_source[idx] = replacement_str
        found = True
        print(f"Replaced line {idx}: {line.strip()} -> {replacement_str.strip()}")
        break

assert found, "Target string was not found in Cell 10 source!"

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("Notebook 07 Cell 10 updated successfully.")

