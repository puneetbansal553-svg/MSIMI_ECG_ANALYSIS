import shutil
import hashlib
from datetime import datetime
from pathlib import Path

project_root = Path(r"C:\Users\HP\Desktop\MSIMI_Project")
backup_dir = project_root / "backups" / "stage06_pre_corrected_rerun_20261031"

target_relative_files = [
    Path("reports/hrv/06_rr_hrv_manifest.csv"),
    Path("reports/hrv/06_rr_hrv_summary.json"),
    Path("reports/hrv/06_patient_hrv_summary.csv"),
    Path("outputs/figures/06_hrv_phase_distributions.png"),
    Path("outputs/figures/06_hrv_100hz_vs_500hz_concordance.png"),
    Path("outputs/figures/06_hrv_representative_rr_tachograms.png")
]

def get_file_info(file_path: Path):
    assert file_path.exists(), f"File does not exist: {file_path}"
    content = file_path.read_bytes()
    size = len(content)
    sha256 = hashlib.sha256(content).hexdigest()
    mtime = datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
    return size, sha256, mtime

print(f"=== Stage 06 Output Backup Operation ===")
print(f"Project root: {project_root}")
print(f"Backup target directory: {backup_dir}")

# Record pre-copy info for source files
pre_source_info = {}
for rel_p in target_relative_files:
    src_p = project_root / rel_p
    size, sha256, mtime = get_file_info(src_p)
    pre_source_info[rel_p] = {"size": size, "sha256": sha256, "mtime": mtime}
    print(f"Source verified: {rel_p} | {size} bytes | SHA256: {sha256}")

# Create backup directories and copy preserving relative structure
backup_dir.mkdir(parents=True, exist_ok=True)
for rel_p in target_relative_files:
    src_p = project_root / rel_p
    dst_p = backup_dir / rel_p
    dst_p.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_p, dst_p)

print("\n=== Post-Copy Verification ===")
verification_results = []
all_matched = True

for rel_p in target_relative_files:
    src_p = project_root / rel_p
    dst_p = backup_dir / rel_p
    
    assert dst_p.exists(), f"Backup file missing: {dst_p}"
    
    # Check backup file
    dst_size, dst_sha256, dst_mtime = get_file_info(dst_p)
    
    # Check original file again to ensure unmutated
    post_src_size, post_src_sha256, post_src_mtime = get_file_info(src_p)
    
    pre = pre_source_info[rel_p]
    src_unmutated = (post_src_size == pre["size"]) and (post_src_sha256 == pre["sha256"])
    hashes_match = (post_src_sha256 == dst_sha256)
    sizes_match = (post_src_size == dst_size)
    
    if not (src_unmutated and hashes_match and sizes_match):
        all_matched = False
        
    verification_results.append({
        "relative_path": str(rel_p),
        "source_size": post_src_size,
        "backup_size": dst_size,
        "source_sha256": post_src_sha256,
        "backup_sha256": dst_sha256,
        "src_unmutated": src_unmutated,
        "hashes_match": hashes_match,
        "sizes_match": sizes_match
    })
    
    status_str = "MATCH" if (hashes_match and sizes_match and src_unmutated) else "MISMATCH"
    print(f"[{status_str}] {rel_p}")
    print(f"  Source: {post_src_size} bytes | {post_src_sha256}")
    print(f"  Backup: {dst_size} bytes | {dst_sha256}")

print("\nOverall Status:")
if all_matched:
    print("[PASS] All 6 backup files exist, sizes match, and SHA-256 hashes match perfectly.")
    print("[PASS] Original files remain strictly unmutated.")
else:
    print("[FAIL] Mismatch detected during backup verification.")

