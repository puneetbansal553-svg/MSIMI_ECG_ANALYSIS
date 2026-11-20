import os
for k, v in os.environ.items():
    if any(term in k.upper() for term in ["PATH", "CONDA", "PYTHON", "VIRTUAL"]):
        print(f"{k} = {v}\n")

