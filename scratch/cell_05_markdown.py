### Section 4: Verified Feature Groups & Clinical Domain Spaces

To avoid the curse of dimensionality ($p = 259 \gg P_{	ext{train}} = 7$), feature selection is constrained strictly to three small, clinically grounded, verified feature subsets from [`08_patient_phase_features.csv`](file:///C:/Users/HP/Desktop/MSIMI_Project/data/features/08_patient_phase_features.csv):
- **Subset A (Primary Ischemic ST Dynamics, $k=4$)**: Focuses on minimum ST60 amplitude across lead territories at peak stress and stress-induced ST shift.
- **Subset B (Autonomic Reactivity & Recovery, $k=3$)**: Focuses on heart rate recovery at 60s post-stress, heart rate reactivity, and baseline parasympathetic tone.
- **Subset C (Multimodal Composite, $k=5$)**: Combines peak ST depression, ST reactivity, heart rate recovery, heart rate reactivity, and global beat tensor deformation.
