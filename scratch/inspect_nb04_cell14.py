import json

with open("notebooks/04_ecg_preprocessing.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

src = "".join(nb["cells"][14]["source"])
print(f"Total length: {len(src)}")
# Find all plt.savefig
import re
for m in re.finditer(r"plt\.savefig\((.*?)\)", src):
    print(m.group(0))

