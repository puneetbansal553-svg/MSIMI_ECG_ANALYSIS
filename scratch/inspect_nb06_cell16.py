import json

with open("notebooks/06_rr_hrv_analysis.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

src = "".join(nb["cells"][16]["source"])
print(src[:1500])

