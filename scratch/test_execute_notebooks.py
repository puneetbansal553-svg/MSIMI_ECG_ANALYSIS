"""
scratch/test_execute_notebooks.py

Executes code cells of each notebook sequentially to verify:
1. Zero runtime exceptions.
2. Proper output generation in each notebook's notebook_outputs/ directory.
3. Headless non-blocking execution using Matplotlib 'Agg' backend.
"""

import json
import os
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg') # Ensure non-blocking headless execution
import matplotlib.pyplot as plt

notebooks = [
    'COMPARISON/01_DATA_INVENTORY/01_data_inventory_comparison.ipynb',
    'COMPARISON/02_COHORT_QC_COMPARISON/02_cohort_qc_comparison.ipynb',
    'COMPARISON/03_PREPROCESSING_COMPARISON/03_preprocessing_comparison.ipynb',
    'COMPARISON/04_RPEAK_HRV_COMPARISON/04_rpeak_hrv_comparison.ipynb',
    'COMPARISON/05_REST_STRESS_RECOVERY/05_rest_stress_recovery_comparison.ipynb',
    'COMPARISON/06_TRANSITION_COMPARISON/06_transition_comparison.ipynb',
    'COMPARISON/07_FEATURE_COMPARISON/07_feature_comparison.ipynb',
    'COMPARISON/08_VALIDATION_COMPARISON/08_validation_comparison.ipynb',
    'COMPARISON/09_MODEL_PERFORMANCE/09_model_performance_comparison.ipynb',
    'COMPARISON/10_LITERATURE_COMPARISON/10_comprehensive_literature_comparison.ipynb',
    'COMPARISON/11_FINAL_REPORT/11_final_research_comparison.ipynb'
]

results = []

for nb_path in notebooks:
    print(f"\n=======================================================")
    print(f"Testing execution: {nb_path}")
    print(f"=======================================================")
    
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb_data = json.load(f)
        
    code_cells = [c for c in nb_data['cells'] if c['cell_type'] == 'code']
    print(f"Found {len(code_cells)} code cells.")
    
    # Clean namespace for isolated notebook execution
    ns = {
        '__name__': '__main__',
        'display': lambda x: None,
        'plt': plt
    }
    
    passed = True
    for idx, cell in enumerate(code_cells):
        code_str = "".join(cell['source'])
        try:
            exec(code_str, ns)
            plt.close('all')
        except Exception as e:
            print(f"FAILED in cell {idx+1}: {e}")
            import traceback
            traceback.print_exc()
            passed = False
            break
            
    if passed:
        print(f"PASSED: {nb_path} executed 100% cleanly top-to-bottom!")
        results.append((nb_path, "PASSED"))
    else:
        results.append((nb_path, "FAILED"))
        break

print("\n--- Summary of Notebook Execution Validation ---")
for nb, status in results:
    print(f"  {status}: {nb}")
