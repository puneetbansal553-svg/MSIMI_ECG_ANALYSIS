"""
scratch/notebook_builder_utils.py

Provides utility functions to assemble valid Jupyter Notebooks (.ipynb)
and validate their JSON structure and Python AST syntax.
"""

import json
import ast
from pathlib import Path

def make_md_cell(text):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.strip().split("\n")]
    }

def make_code_cell(code):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in code.strip().split("\n")]
    }

def save_and_validate_notebook(cells, filepath):
    nb = {
        "cells": cells,
        "metadata": {
            "language_info": {
                "name": "python",
                "version": "3.12"
            },
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }
    
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
        
    # Validate JSON reload
    with open(filepath, "r", encoding="utf-8") as f:
        reloaded = json.load(f)
        
    # Validate Python AST of each code cell
    code_cell_count = 0
    for idx, cell in enumerate(reloaded["cells"]):
        if cell["cell_type"] == "code":
            code_cell_count += 1
            code_str = "".join(cell["source"])
            try:
                ast.parse(code_str)
            except SyntaxError as e:
                print(f"SYNTAX ERROR in {filepath.name} cell {idx}: {e}")
                raise e
                
    print(f"Validated {filepath.name}: {len(cells)} cells ({code_cell_count} code cells) syntactically valid!")
    return True

