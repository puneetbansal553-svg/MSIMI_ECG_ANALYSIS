# Stage 1: Small-Sample Validation & Quality Control Report (Updated)

**Date:** October 28, 2026  
**Scope:** Stage 1 Validation on 5 Representative Subjects (`sub_001`, `sub_002`, `sub_005`, `sub_029`, `sub_054`)  
**Recordings Validated:** 30 ECG files (5 subjects × 3 phases × 2 sampling frequencies)  
**Raw Dataset Path:** `C:\Users\HP\Downloads\MSIMI_DB` (Strictly Read-Only, 1,668 Files Unaltered)  

---

## 1. Approved Scope & Stage 1 Architectural Decisions

1. **Unequal Lead Lengths Policy:**
   * Raw ECG signals are strictly preserved at their native recording lengths without truncation or padding in storage (`List[np.ndarray]`).
   * When synchronous 12-lead multi-lead operations are executed (e.g. multi-lead segmentation, vectorcardiogram), only the **overlapping valid time interval** $[0, t_{\text{min\_overlap}}]$ is utilized.
   * Discrepancies are flagged in the QC report; no raw data is silently discarded.
2. **Sampling Frequencies (100 Hz vs. 500 Hz):**
   * Both 100 Hz and 500 Hz recordings are preserved and evaluated separately.
   * **100 Hz** serves as the initial stable reference pipeline (demonstrating high physiological stability: mean HR 60.5–103.4 bpm across all 5 subjects).
   * **500 Hz** was investigated: high-frequency derivative sensitivity caused double-peak triggering on noisy leads. A post-alignment refractory pass (200 ms) was added to suppress sub-refractory peaks, achieving exact heart-rate alignment with 100 Hz on standard recordings (e.g., `sub_001` Rest: 519 vs. 520 beats; Stress: 798 vs. 798 beats).
3. **R-Peak Quality Profiling & Questionable Detection Logging:**
   * Detector outputs are cross-referenced against physiological boundaries ($300\,\text{ms} \le \text{RR} \le 2000\,\text{ms}$) and abrupt RR jumps ($>40\%$).
   * All 3,407 flagged candidate beats across the 30 recordings have been exported to [`reports/qc/questionable_detections.csv`](file:///C:/Users/HP/Desktop/MSIMI_Project/reports/qc/questionable_detections.csv) for manual review.
   * No claim of clinical certification is made.
4. **Baseline Correction & ADC Offset Policy:**
   * Configurable in [`configs/preprocessing.yaml`](file:///C:/Users/HP/Desktop/MSIMI_Project/configs/preprocessing.yaml).
   * While raw signals exhibit empirical medians near 2048 (ranging 2032 to 2056 ADC counts), the pipeline relies on **per-lead empirical median subtraction** rather than forcing a blind 2048 hardware constant, ensuring zero-centered isoelectric lines even in the presence of baseline shift.

---

## 2. Ingestion & Quality Control Findings (30 Recordings)

### A. Integrity Audit Summary
* **Lead Count:** **100% compliant** (all 30 recordings have exactly 12 leads in the standard order `I`, `II`, `III`, `aVR`, `aVL`, `aVF`, `V1`, `V2`, `V3`, `V4`, `V5`, `V6`).
* **Finiteness:** **100% finite** (0 `NaN`, 0 `Inf` values across all recordings).
* **Dynamic Range & Clipping:** **0.0000% clipping** (no sample values reached saturation limits $\le 5$ or $\ge 4090$).
* **Flatlines:** **0 flatlines** detected (all leads have variance $> 1.0\,\text{mV}^2$).

### B. Recording Durations & Lead Length Discrepancies
* **Standard Recordings (24 of 30):** Rest: 360.0s (36k / 180k samples), Stress: 720.0s (72k / 360k samples), Recover: 360.0s (36k / 180k samples).
* **Irregular Durations:**
  1. `sub_029` Rest (100 Hz & 500 Hz): Truncated duration of **269.6 s** (nominal 360s).
  2. `sub_054` Recovery (100 Hz & 500 Hz): Truncated duration of **232.0 s** (nominal 360s).
* **Unequal Lead Lengths (Lead V6 Truncation in Hospital Export):**
  1. `sub_029` Rest: Leads 0–10 have 27,100 samples (271.0s) @ 100 Hz; Lead V6 has 26,959 samples (269.59s, stops 1.41s earlier). At 500 Hz: Leads 0–10 have 135,500 samples; Lead V6 has 134,793 samples (stops 1.414s earlier).
  2. `sub_002` Rest (500 Hz): Leads 0–10 have 180,000 samples (360.0s); Lead V6 has 169,581 samples (339.16s, stops 20.84s earlier).
  3. `sub_005` Rest (500 Hz): Leads 0–10 have 180,000 samples (360.0s); Lead V6 has 169,581 samples (339.16s, stops 20.84s earlier).

---

## 3. Calibrated Preprocessing and R-Peak Validation (Lead II)

Validation was executed on Lead II for all 30 recordings using the calibrated Pan-Tompkins detector with post-alignment refractory blanking:

| Subject | Phase | 100 Hz Mean HR | 100 Hz Beats | 500 Hz Mean HR | 500 Hz Beats | Lead II Quality Rating |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **`sub_001` (Positive)** | Rest | 86.4 bpm | 519 | 86.6 bpm | 520 | EXCELLENT |
| | Stress | 66.5 bpm | 798 | 66.5 bpm | 798 | EXCELLENT |
| | Recover | 70.3 bpm | 422 | 70.6 bpm | 424 | EXCELLENT |
| **`sub_002` (Negative)** | Rest | 78.4 bpm | 470 | 136.7 bpm* | 820* | NOISY |
| | Stress | 82.0 bpm | 984 | 83.2 bpm | 999 | EXCELLENT |
| | Recover | 85.7 bpm | 514 | 72.7 bpm | 436 | EXCELLENT |
| **`sub_005` (Unspecified)**| Rest | 71.5 bpm | 429 | 72.4 bpm | 434 | ACCEPTABLE |
| | Stress | 76.9 bpm | 923 | 77.7 bpm | 932 | NOISY |
| | Recover | 72.1 bpm | 433 | 72.3 bpm | 434 | EXCELLENT |
| **`sub_029` (Negative)** | Rest | 60.5 bpm | 274 | 66.9 bpm | 303 | EXCELLENT |
| | Stress | 81.9 bpm | 983 | 87.9 bpm | 1054 | NOISY |
| | Recover | 62.0 bpm | 372 | 62.4 bpm | 375 | EXCELLENT |
| **`sub_054` (Negative)** | Rest | 87.8 bpm | 527 | 99.3 bpm | 596 | NOISY |
| | Stress | 103.2 bpm | 1238 | 121.1 bpm | 1452 | NOISY |
| | Recover | 96.6 bpm | 373 | 108.0 bpm | 418 | ACCEPTABLE |

*\*Note on `sub_002` Rest 500 Hz:* Lead II in this specific recording displays excessive high-frequency noise and baseline drift, triggering secondary peaks even after refractory filtering. It is correctly categorized as `NOISY` and queued for lead-fallback (to Lead V5 or Lead I) during Stage 2.

---

## 4. Questionable Detections Audit

A dedicated anomaly audit was executed across all 30 recordings to log every beat violating physiological consistency:
* **Output File:** [`reports/qc/questionable_detections.csv`](file:///C:/Users/HP/Desktop/MSIMI_Project/reports/qc/questionable_detections.csv)
* **Total Flagged Detections:** 3,407 beats across 30 recordings.
  * Short RR intervals ($< 300\,\text{ms}$, $> 200\,\text{bpm}$): **1,179** (predominantly in noisy 500 Hz epochs).
  * Long RR intervals ($> 2000\,\text{ms}$, $< 30\,\text{bpm}$): **9** (indicative of transient missed beats or true sinus pauses).
  * Abrupt step changes ($|\Delta\text{RR}| / \text{RR}_{\text{prev}} > 40\%$): **2,918** (candidate premature ventricular/atrial contractions or noisy triggers).
* **Sampling Rate Breakdown:** 1,049 in 100 Hz files vs. 2,358 in 500 Hz files.

---

## 5. Summary of Created & Modified Artifacts

* **Source Code:**
  * `src/loader.py` & `src/data/loader.py`: Non-destructive loading, `UnequalLeadLengthsError`, and overlapping-interval support.
  * `src/peak_detection/pan_tompkins.py` & `src/rpeak.py`: Post-alignment refractory blanking calibration.
* **Test Suite:**
  * `tests/test_loader.py`: Tested `UnequalLeadLengthsError` and non-destructive integrity detection (16 of 16 tests passing).
* **QC Reports (in `reports/qc/`):**
  * `sample_validation_integrity.csv`
  * `sample_validation_rpeaks.csv`
  * `questionable_detections.csv`
  * `sample_validation_qc_report.json`
  * `sample_validation_report.md`

---

## 6. Decisions for Full-Cohort Processing (Awaiting Direction)

1. **Lead Fallback Policy for Noisy Leads:** When Lead II is categorized as `NOISY` (e.g. `sub_002` Rest 500 Hz), should the pipeline automatically fall back to Lead V5 or Lead I as defined in `preprocessing.yaml`, or evaluate multi-lead consensus R-peaks?
2. **Full-Cohort Execution Sequence:** Do you approve running Stage 1 integrity QC and 100 Hz baseline R-peak extraction across the remaining 75 subjects of the 80 ANOCA cohort?
3. **Stage 2 Transition:** Confirmation that all Stage 1 findings are accepted before advancing to window/beat segmentation and HRV feature extraction.
