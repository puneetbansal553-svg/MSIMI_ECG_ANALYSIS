import os
import sys
import json
import hashlib
import shutil
import numpy as np
import pandas as pd

sys.path.insert(0, os.getcwd())

def compute_sha256(filepath):
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def main():
    print("=== POST-BATCH 4 & FULL COHORT VERIFICATION SCRIPT ===")
    
    # 1. Check disk space
    usage = shutil.disk_usage('C:')
    free_gb = usage.free / (1024 ** 3)
    print(f"Free disk space on C: {free_gb:.2f} GB")
    assert free_gb >= 6.0, f"Insufficient disk space: {free_gb:.2f} GB < 6.0 GB"
    
    # 2. Check temporary files
    rpeaks_dir = os.path.join('data', 'processed', 'rpeaks')
    tmp_files = [f for f in os.listdir(rpeaks_dir) if '_tmp_' in f or f.endswith('.tmp')]
    print(f"Temporary files found in {rpeaks_dir}: {len(tmp_files)}")
    assert len(tmp_files) == 0, f"Uncertain or orphan temporary files detected: {tmp_files}"
    
    # 3. Check manifest rows and unique IDs
    manifest_path = os.path.join('reports', 'rpeaks', '05_rpeak_detection_manifest.csv')
    assert os.path.exists(manifest_path), f"Manifest file missing: {manifest_path}"
    df_manifest = pd.read_csv(manifest_path)
    total_rows = len(df_manifest)
    unique_ids = df_manifest['recording_id'].nunique()
    print(f"Total manifest rows: {total_rows}")
    print(f"Unique recording IDs in manifest: {unique_ids}")
    assert total_rows == 480, f"Expected 480 rows, found {total_rows}"
    assert unique_ids == 480, f"Expected 480 unique IDs, found {unique_ids}"
    
    # Check status counts
    status_counts = df_manifest['rpeak_detection_status'].value_counts().to_dict()
    print(f"rpeak_detection_status breakdown: {status_counts}")
    success_count = status_counts.get('SUCCESS', 0)
    audit_error_count = status_counts.get('SKIPPED_AUDIT_ERROR', 0)
    no_rhythm_lead_count = status_counts.get('NO_ACCEPTABLE_RHYTHM_LEAD', 0)
    print(f"SUCCESS: {success_count}")
    print(f"SKIPPED_AUDIT_ERROR: {audit_error_count}")
    print(f"NO_ACCEPTABLE_RHYTHM_LEAD: {no_rhythm_lead_count}")
    
    # Reconcile against preprocessing manifest
    prep_manifest_path = os.path.join('reports', 'preprocessing', '04_ecg_preprocessing_manifest.csv')
    df_prep = pd.read_csv(prep_manifest_path)
    prep_errors = set(df_prep[df_prep['preprocessing_status'] == 'SKIPPED_AUDIT_ERROR']['recording_id'])
    stage5_errors = set(df_manifest[df_manifest['rpeak_detection_status'] == 'SKIPPED_AUDIT_ERROR']['recording_id'])
    assert prep_errors == stage5_errors, f"Mismatch in audit error IDs between stage 4 and stage 5! Diff: {prep_errors ^ stage5_errors}"
    print(f"[OK] Audit error IDs match Stage 04 manifest exactly ({len(stage5_errors)} recordings).")
    
    # 4. Count archives on disk
    archive_files = [f for f in os.listdir(rpeaks_dir) if f.endswith('_rpeaks.npz') and '_tmp_' not in f]
    print(f"Total archives on disk: {len(archive_files)}")
    assert len(archive_files) == success_count, f"Mismatch: {len(archive_files)} archives on disk vs {success_count} SUCCESS in manifest"
    
    # 5. Verify 40 Pilot Hashes
    pilot_hash_path = os.path.join('reports', 'rpeaks', 'pilot_40_sha256_manifest.json')
    with open(pilot_hash_path, 'r') as f:
        pilot_baseline = json.load(f)
    print(f"Verifying {len(pilot_baseline)} pilot archives against baseline...")
    pilot_mismatches = []
    for rec_id, val in pilot_baseline.items():
        expected_hash = val['sha256'] if isinstance(val, dict) else val
        arc_path = os.path.join(rpeaks_dir, f"{rec_id}_rpeaks.npz")
        assert os.path.exists(arc_path), f"Missing pilot archive: {arc_path}"
        actual_hash = compute_sha256(arc_path)
        if actual_hash != expected_hash:
            pilot_mismatches.append((rec_id, expected_hash, actual_hash))
    print(f"Pilot hash mismatches: {len(pilot_mismatches)}")
    assert len(pilot_mismatches) == 0, f"Pilot hash mismatches detected: {pilot_mismatches}"
    print("[OK] All 40 pilot archives are 100% byte-for-byte identical to baseline.")
    
    # 6. Verify 331 Pre-Existing Hashes (Batches 1-3)
    pre_existing_hash_path = os.path.join('reports', 'rpeaks', 'pre_existing_331_hashes.json')
    with open(pre_existing_hash_path, 'r') as f:
        pre_existing_baseline = json.load(f)
    print(f"Verifying {len(pre_existing_baseline)} pre-existing archives (Batches 1-3)...")
    pre_existing_mismatches = []
    for rec_id, expected_hash in pre_existing_baseline.items():
        arc_path = os.path.join(rpeaks_dir, f"{rec_id}_rpeaks.npz")
        assert os.path.exists(arc_path), f"Missing pre-existing archive: {arc_path}"
        actual_hash = compute_sha256(arc_path)
        if actual_hash != expected_hash:
            pre_existing_mismatches.append((rec_id, expected_hash, actual_hash))
    print(f"Pre-existing hash mismatches: {len(pre_existing_mismatches)}")
    assert len(pre_existing_mismatches) == 0, f"Pre-existing hash mismatches detected: {pre_existing_mismatches}"
    print("[OK] All 331 pre-existing archives (Batches 1-3) are 100% byte-for-byte identical to baseline.")
    
    # Check new archives in Batch 4
    new_archives = [f for f in archive_files if f.replace('_rpeaks.npz', '') not in pre_existing_baseline]
    print(f"New archives added in Batch 4: {len(new_archives)}")
    assert len(new_archives) == 133, f"Expected 133 new archives in Batch 4, got {len(new_archives)}"
    
    # 7. Deep Structural & Metadata Validation across ALL 464 archives
    print("Performing deep structural and numerical integrity validation on all 464 archives...")
    leads_order = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    
    structural_errors = []
    hr_exceptions = []
    rhythm_lead_counts = {}
    fallback_details = []
    
    for arc_file in archive_files:
        rec_id = arc_file.replace('_rpeaks.npz', '')
        arc_path = os.path.join(rpeaks_dir, arc_file)
        
        # Non-zero size
        size_bytes = os.path.getsize(arc_path)
        if size_bytes == 0:
            structural_errors.append(f"{rec_id}: zero size archive")
            continue
            
        data = np.load(arc_path, allow_pickle=True)
        files_in_npz = set(data.files)
        
        # Check metadata
        if 'metadata_json' not in files_in_npz:
            structural_errors.append(f"{rec_id}: missing metadata_json in npz")
            continue
            
        meta = json.loads(str(data['metadata_json']))
        if meta.get('recording_id') != rec_id:
            structural_errors.append(f"{rec_id}: embedded recording_id {meta.get('recording_id')} != filename")
            
        rhythm_lead = meta.get('selected_rhythm_lead')
        rhythm_lead_counts[rhythm_lead] = rhythm_lead_counts.get(rhythm_lead, 0) + 1
        
        if rhythm_lead != 'II':
            fallback_details.append({
                'recording_id': rec_id,
                'selected_lead': rhythm_lead,
                'rhythm_fallback_reason': meta.get('rhythm_fallback_reason', 'N/A'),
                'rhythm_fallback_used': meta.get('rhythm_fallback_used', False)
            })
            
        # Check 12 leads present
        for lead in leads_order:
            peaks_key = f"{lead}_r_peaks"
            rr_key = f"{lead}_rr_intervals_ms"
            if peaks_key not in files_in_npz or rr_key not in files_in_npz:
                structural_errors.append(f"{rec_id}: missing {peaks_key} or {rr_key}")
                continue
            
            peaks = data[peaks_key]
            rr_ms = data[rr_key]
            
            # Check length relation len(rr) == max(0, len(peaks) - 1)
            expected_rr_len = max(0, len(peaks) - 1)
            if len(rr_ms) != expected_rr_len:
                structural_errors.append(f"{rec_id}_{lead}: len(rr)={len(rr_ms)} != expected {expected_rr_len}")
                
            # Check finite
            if len(peaks) > 0 and not np.all(np.isfinite(peaks)):
                structural_errors.append(f"{rec_id}_{lead}: non-finite values in peaks")
            if len(rr_ms) > 0 and not np.all(np.isfinite(rr_ms)):
                structural_errors.append(f"{rec_id}_{lead}: non-finite values in rr_ms")
                
            # Check sorted peaks
            if len(peaks) > 1 and np.any(np.diff(peaks) <= 0):
                structural_errors.append(f"{rec_id}_{lead}: peaks not strictly monotonically increasing")
                
        # Check HR consistency on rhythm lead
        # |60000 / mean(RR) - mean_hr| <= 0.10 bpm
        rhythm_rr = data[f"{rhythm_lead}_rr_intervals_ms"]
        lead_summary = meta.get('per_lead_summary', {}).get(rhythm_lead, {})
        reported_mean_hr = lead_summary.get('mean_hr_bpm')
        if len(rhythm_rr) > 0 and reported_mean_hr is not None:
            calc_mean_rr = float(np.mean(rhythm_rr))
            expected_mean_hr = 60000.0 / calc_mean_rr
            diff = abs(expected_mean_hr - reported_mean_hr)
            if diff > 0.10:
                hr_exceptions.append({
                    'recording_id': rec_id,
                    'lead': rhythm_lead,
                    'calc_mean_hr': expected_mean_hr,
                    'reported_mean_hr': reported_mean_hr,
                    'diff': diff
                })
        elif len(rhythm_rr) == 0:
            if reported_mean_hr is not None and not np.isnan(reported_mean_hr):
                hr_exceptions.append({
                    'recording_id': rec_id,
                    'lead': rhythm_lead,
                    'calc_mean_hr': None,
                    'reported_mean_hr': reported_mean_hr,
                    'diff': 'No RR intervals but reported HR is not None/NaN'
                })
                
    print(f"Structural validation errors: {len(structural_errors)}")
    assert len(structural_errors) == 0, f"Structural errors found: {structural_errors[:10]}"
    print("[OK] All 464 archives pass structural and numerical validation.")
    
    print(f"HR consistency exceptions: {len(hr_exceptions)}")
    if len(hr_exceptions) > 0:
        print(f"HR exceptions found ({len(hr_exceptions)}):")
        for exc in hr_exceptions[:10]:
            print("  ", exc)
    else:
        print("[OK] Internal HR consistency (|60000/mean(RR) - mean_hr| <= 0.10 bpm) verified across all 464 archives.")
        
    print(f"Rhythm lead distribution across cohort: {rhythm_lead_counts}")
    print(f"Total fallback occurrences (rhythm lead != II): {len(fallback_details)}")
    if len(fallback_details) > 0:
        print("Fallback recordings:")
        for fb in fallback_details:
            print(f"  {fb['recording_id']}: {fb['selected_lead']} (Reason: {fb['rhythm_fallback_reason']})")

    # Save verification metrics to json
    audit_results = {
        'total_manifest_rows': total_rows,
        'unique_recording_ids': unique_ids,
        'status_counts': status_counts,
        'total_archives_on_disk': len(archive_files),
        'pilot_mismatches': len(pilot_mismatches),
        'pre_existing_mismatches': len(pre_existing_mismatches),
        'batch4_new_archives': len(new_archives),
        'structural_errors': len(structural_errors),
        'hr_exceptions': hr_exceptions,
        'rhythm_lead_distribution': rhythm_lead_counts,
        'fallback_count': len(fallback_details),
        'fallback_details': fallback_details,
        'free_disk_space_gb': free_gb
    }
    with open(os.path.join('reports', 'rpeaks', 'post_batch4_verification_summary.json'), 'w') as f:
        json.dump(audit_results, f, indent=2)
    print("Audit results written to reports/rpeaks/post_batch4_verification_summary.json")

if __name__ == '__main__':
    main()
