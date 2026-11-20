import os
import json
import numpy as np
import pandas as pd

manifest = pd.read_csv('reports/rpeaks/05_rpeak_detection_manifest.csv')
success_rows = manifest[manifest['rpeak_detection_status'] == 'SUCCESS']

print(f"Total SUCCESS rows: {len(success_rows)}")

# Check min and max RR interval across a sample of 20 recordings
min_rrs = []
max_rrs = []
sample_ids = success_rows['recording_id'].sample(30, random_state=42)

for rec_id in sample_ids:
    p = os.path.join('data', 'processed', 'rpeaks', f"{rec_id}_rpeaks.npz")
    d = np.load(p, allow_pickle=True)
    meta = json.loads(str(d['metadata_json']))
    rhythm_lead = meta['selected_rhythm_lead']
    rr = d[f"{rhythm_lead}_rr_intervals_ms"]
    min_rrs.append(np.min(rr))
    max_rrs.append(np.max(rr))

print(f"Sample 30 recordings - Minimum RR observed: {np.min(min_rrs):.1f} ms, Maximum RR observed: {np.max(max_rrs):.1f} ms")

