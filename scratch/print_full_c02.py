import json
from pathlib import Path

nb_p = Path(r"C:\Users\HP\Desktop\MSIMI_Project\notebooks\08_ecg_feature_extraction.ipynb")
with open(nb_p, "r", encoding="utf-8") as f:
    nb = json.load(f)

c02 = nb["cells"][2]
print(f"Cell index: 2")
print(f"Cell type: {c02['cell_type']}")
print(f"Execution count: {c02.get('execution_count')}")
print("--- Full Cell 02 Source ---")
for idx, line in enumerate(c02["source"], 1):
    print(f"{idx:02d}: {line.rstrip()}")

