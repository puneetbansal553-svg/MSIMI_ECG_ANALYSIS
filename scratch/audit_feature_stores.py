import pandas as pd
import numpy as np

df_rec = pd.read_csv('data/features/08_recording_level_features.csv')
print(f"08_recording_level_features.csv shape: {df_rec.shape}")
print("Sample columns:", df_rec.columns[:30].tolist())
print("Phases present in recording-level features:", df_rec['phase'].value_counts().to_dict() if 'phase' in df_rec.columns else "No phase col")
print("Sampling rates present:", df_rec['sampling_rate'].value_counts().to_dict() if 'sampling_rate' in df_rec.columns else "No sampling_rate col")

df_pat = pd.read_csv('data/features/08_patient_phase_features.csv')
print(f"\n08_patient_phase_features.csv shape: {df_pat.shape}")
print("Sample columns:", df_pat.columns[:30].tolist())
if 'sampling_rate' in df_pat.columns:
    print("Sampling rates in patient-phase features:", df_pat['sampling_rate'].value_counts().to_dict())

# Check transitions
df_trans = pd.read_csv('reports/transitions/07a_inter_phase_transition_manifest.csv')
print(f"\n07a_inter_phase_transition_manifest.csv shape: {df_trans.shape}")
print("Transition types:", df_trans['transition_type'].value_counts().to_dict())
print("Sample columns:", df_trans.columns[:20].tolist())

