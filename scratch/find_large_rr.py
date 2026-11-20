import os
import json
import numpy as np
import pandas as pd

manifest = pd.read_csv('reports/rpeaks/05_rpeak_detection_manifest.csv')
success_rows = manifest[manifest['rpeak_detection_status'] == 'SUCCESS']

sample_ids = success_rows['recording_id'].sample(30, random_state=42)

for rec_id in sample_ids:
    p = os.path.join('data', 'processed', 'rpeaks', f"{rec_id}_rpeaks.npz")
    d = np.load(p, allow_pickle=True)
    meta = json.loads(str(d['metadata_json']))
    rhythm_lead = meta['selected_rhythm_lead']
    rr = d[f"{rhythm_lead}_rr_intervals_ms"]
    max_rr = np.max(rr)
    if max_rr > 5000:
        print(f"Recording {rec_id} ({rhythm_lead}): max RR = {max_rr:.1f} ms, short_count = {np.sum(rr < 300)}, long_count = {np.sum(rr > 2000)}, total_rr = {len(rr)}")

