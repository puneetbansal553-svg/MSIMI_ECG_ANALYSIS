import os
import sys
import json
import numpy as np
import pandas as pd

sys.path.insert(0, os.getcwd())

def inspect_notebook_05():
    print("=== NOTEBOOK 05 CELLS ===")
    with open('notebooks/05_rpeak_detection.ipynb', 'r', encoding='utf-8') as f:
        nb = json.load(f)
    for i, c in enumerate(nb['cells']):
        first_line = c['source'][0].strip() if c['source'] else ''
        print(f"Cell {i:2d} ({c['cell_type']:8s}): {first_line[:80]}")

def inspect_stage05_manifest():
    print("\n=== STAGE 05 MANIFEST SAMPLE ===")
    df = pd.read_csv('reports/rpeaks/05_rpeak_detection_manifest.csv')
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Status value counts:\n{df['rpeak_detection_status'].value_counts()}")
    print(f"Selected rhythm leads:\n{df['selected_rhythm_lead'].value_counts()}")
    print("\nFirst 3 rows:")
    print(df[['recording_id', 'phase', 'sampling_rate', 'rpeak_detection_status', 'selected_rhythm_lead', 'rhythm_peak_count', 'rhythm_mean_hr_bpm', 'rhythm_median_rr_ms']].head(3))
    
    # Check subjects
    subjects = df['subject_id'].unique()
    print(f"\nUnique subjects count: {len(subjects)} (min: {subjects[0]}, max: {subjects[-1]})")

def inspect_sample_npz():
    print("\n=== SAMPLE NPZ ARCHIVES INSPECTION ===")
    sample_files = [
        'sub_001_rest_100hz_rpeaks.npz',
        'sub_001_rest_500hz_rpeaks.npz',
        'sub_002_stress_500hz_rpeaks.npz',  # known T-wave oversensing
        'sub_022_rest_100hz_rpeaks.npz',   # fallback to V5
        'sub_045_stress_500hz_rpeaks.npz'  # fallback to V5
    ]
    for fn in sample_files:
        p = os.path.join('data', 'processed', 'rpeaks', fn)
        if not os.path.exists(p):
            print(f"File {fn} does not exist!")
            continue
        data = np.load(p, allow_pickle=True)
        meta = json.loads(str(data['metadata_json']))
        print(f"\nFile: {fn}")
        print(f"  Rhythm lead: {meta['selected_rhythm_lead']}, fallback used: {meta['rhythm_fallback_used']}")
        rhythm_lead = meta['selected_rhythm_lead']
        rr = data[f"{rhythm_lead}_rr_intervals_ms"]
        peaks = data[f"{rhythm_lead}_r_peaks"]
        print(f"  Peaks count: {len(peaks)}, RR count: {len(rr)}")
        if len(rr) > 0:
            print(f"  RR range: [{np.min(rr):.1f}, {np.max(rr):.1f}] ms, mean: {np.mean(rr):.1f} ms, std: {np.std(rr):.1f} ms")
            # calculate sample diffs
            successive_diffs = np.diff(rr)
            print(f"  Successive diffs count: {len(successive_diffs)}, RMSSD est: {np.sqrt(np.mean(successive_diffs**2)):.2f} ms")
            pnn50 = np.sum(np.abs(successive_diffs) > 50.0) / len(successive_diffs) * 100.0
            print(f"  pNN50 est: {pnn50:.2f}%")

if __name__ == '__main__':
    inspect_notebook_05()
    inspect_stage05_manifest()
    inspect_sample_npz()

