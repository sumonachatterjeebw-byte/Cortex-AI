# Cortex AI: Reliability-Aware Brain MRI Analysis Prototype (v2.0)

> **MANDATORY SCIENTIFIC DISCLAIMER**  
> **Research/educational prototype only. Not for diagnosis, screening, triage, or clinical decisions.**  
> Not a diagnostic medical device, not a clinically validated tool, and not a claim that a new segmentation model has been invented.

---

## 1. Executive Summary & Purpose

Automated brain MRI segmentation can be biased or perturbed by scan quality variations, acquisition differences, field strength, motion artifacts, and model domain shifts. 

**Cortex AI** is an educational and research software prototype designed to answer the core research question:
> *How can a brain MRI analysis interface help users identify potentially unreliable automated segmentation results and understand the limits of derived anatomical measurements, rather than presenting every prediction as equally trustworthy?*

---

## 2. System Architecture

```mermaid
graph TD
    A[Browser UI (React + TypeScript + Vite)] -->|REST / Multipart| B[FastAPI Backend Server]
    B --> C[Validators & Decompression Guard]
    C --> D[NiBabel Spatial Geometry Engine]
    D --> E[Quality Control IQM: SNR, CNR, Entropy]
    D --> F{Model Adapter Layer}
    F -->|Option A| G[SynthSeg Real CLI Adapter]
    F -->|Option B| H[FastSurfer Container / CLI Adapter]
    F -->|Option C| I[Synthetic Demo Generator (STAMPED DEMO DATA)]
    G & H & I --> J[Overlay Geometry Gate (Tolerance Check)]
    J --> K[Physical Volume Engine: count x |det(affine)|]
    J --> L[Reference Evaluator: Dice & HD95 in mm]
    K & L --> M[Transparent Multi-Aspect QC Report]
    M --> N[Slice Renderer with Anatomical Markers]
    M --> O[Traceable Exports: HTML Report & CSV Data]
```

---

## 3. Section 9 Register: Challenges, Mitigations, and Verification

| ID | Scientific & Engineering Challenge | Why It Happens | Cortex AI Implementation & Remedy | Verification Status |
|---|---|---|---|---|
| **A1** | Orientation & Affine Confusion | Scanners use different coordinate codes (RAS vs LPS). sform/qform can disagree. | Read affine via NiBabel. Audit sform/qform consistency. Display canonical orientation on every plane. Original affine preserved for volumetric calculations. | **Verified** via `test_geometry.py` with known non-orthogonal matrices. |
| **A2** | Anisotropic or Oblique Voxel Spacing | Clinical scans have thick slices or shear transforms. | Compute exact physical voxel volume from absolute determinant of 3x3 spatial affine: `\|det(affine[:3,:3])\|`. Flag high aspect ratio (>2.0). | **Verified** via unit test `test_voxel_volume_anisotropic`. |
| **A3** | Non-3D or Unsupported Data (2D / 4D) | Diffusion/fMRI scans are 4D; some files have singleton dimensions. | Strictly accept 3D (or squeeze single-frame 4D). Plain-language rejection for 2D or multi-volume 4D. | **Verified** via `test_reject_2d_file` and `test_reject_4d_multivolume`. |
| **A4** | NaN/Inf Values & Uniform/Empty Volumes | Corrupted export or skull-strip mask error. | Reject uniform or all-zero arrays. Zero-out finite NaNs if <0.1%, reject if excessive (>50%). | **Verified** via `test_reject_all_zero_or_constant`. |
| **A5** | Intensity Scaling & Data Types | Header slope/intercept alters stored vs real values. | Access scaled floating-point values via `get_fdata(dtype=np.float32)`. Record original dtype. | **Verified** in image validation pipeline. |
| **A6** | Non-Brain Scan / Field of View Anomaly | User uploads non-cranial MRI or CT. | Surface heuristic field-of-view bounding checks (flags if span <40mm or >500mm). | **Verified** in QC report compiler. |
| **A7** | Corrupted or Fake `.nii` / `.nii.gz` Files | Interrupted downloads or renamed files. | Parse header with NiBabel inside `try/except`. Return sanitized error messages. | **Verified** via `test_reject_corrupted_file`. |
| **B1/B5** | Viewer WebGL Complexity & Fallbacks | Browser compatibility variations. | Robust 2D multi-planar slice renderer with contrast/windowing, zoom, and fallback navigation. | **Verified** in browser UI walkthrough. |
| **B2** | Overlay Geometry Misalignment Gate | Segmentation output on different grid. | Strict geometry validation gate verifies shape and affine alignment before overlay. | **Verified** via `test_validate_overlay_geometry_mismatched_*`. |
| **B4** | Label Legend & Color Readability | Overlapping colormaps confuse users. | Documented FreeSurfer/SynthSeg LUT with explicit anatomical names, hex swatches, and ID cues. | **Verified** via `lut.py` lookup engine. |
| **B6** | Radiological vs Neurological Convention | Inconsistent left/right display misleads interpretation. | Visible directional markers (`L/R`, `A/P`, `S/I`) on every slice viewport. Note in About view. | **Verified** on all 3 viewing planes. |
| **C1/C2** | Model Installation on Windows / Licensing | PyTorch/CUDA/FreeSurfer have Linux or container dependencies. | Pluggable adapter architecture. App operates in viewer-only mode when models are unconfigured. | **Verified**: honest "Model not configured" state. |
| **C5** | Unknown or Mismatched Label IDs | Model outputs unmapped numerical structures. | Unknown IDs rendered as `"Unmapped label N"`, never arbitrarily guessed. | **Verified** via `get_label_info`. |
| **C8** | Fake AI Results Creeping into UI | Reusing demo data without disclosure. | Permanent `[DEMO DATA - NOT REAL AI INFERENCE]` stamping on every demo table, chart, and export. | **Verified** via `test_demo_marker.py`. |
| **D1** | Volume Formula Inaccuracies | Using naive voxel count without affine determinant. | Tested formula: `Volume (mm³) = Count × \|det(affine[:3,:3])\|`, `Volume (mL) = mm³ / 1000`. | **Verified** via `test_volume_calculation_known_counts`. |
| **D2** | Partial Volume & Resolution Effects | Small structures suffer voxel discretization errors. | Warning flag attached to structures <50 voxels; limitations recorded in report. | **Verified** via `test_small_structure_warning`. |
| **D3** | Empty Structure Absences | Model misses a structure entirely. | Reported as 0 voxels with an explicit `"Empty label"` review flag rather than hiding the row. | **Verified** via `test_empty_label_handling`. |
| **D4** | Undefined Dice for Empty Masks | Both masks empty yields 0/0. | Defined mathematical policy: returns `undefined` (None) and excludes from mean. Returns 0 if one is empty. | **Verified** via `test_dice_both_empty` & `test_dice_one_empty`. |
| **D5** | Hausdorff Distance in Physical Units | HD95 computed in voxels instead of millimeters. | Boundary voxels scaled by physical spacing in mm. Undefined for empty masks. | **Verified** via `test_hd95_shifted_box`. |
| **D6** | Averaging Obfuscation | High mean Dice conceals local failure. | Complete per-structure metric distribution table displayed, never only a single mean. | **Verified** via `test_evaluation_per_label_no_hiding`. |
| **E1** | Pseudo-Confidence Score Temptation | Collapsing unrelated checks into an artificial 0-100% score. | **Prohibited.** Displays discrete input checks, geometry flags, IQM indicators, and sanity flags. | **Verified**: No combined confidence score exists in codebase. |
| **E2** | Image Quality Metrics (IQMs) Confused with Accuracy | No-reference SNR/CNR describe the scan, not the model. | IQMs placed in dedicated scan quality panel with explicit explanatory wording. | **Verified** in Quality Control Panel. |
| **G1** | PHI Leakage via Filenames / Headers | Scans contain patient identifiers. | Client filenames discarded for storage; unique server UUIDs (`scan_<uuid>`) generated. | **Verified** in `storage.py`. |
| **G2** | Orphaned Files on Disk | Crashes leave scans in temp storage. | Dedicated temp folder cleared on session delete, startup, and shutdown sweeps. | **Verified** via `test_temporary_session_creation_and_cleanup`. |
| **G3** | Decompression Bombs | Small `.nii.gz` expanding to gigabytes. | Maximum compressed size limit (120MB) and declared uncompressed dimension check (<512M voxels). | **Verified** in `validate_nifti_file`. |
| **G5** | Accidental Cloud/Third-Party Transfer | Analytics or cloud APIs receiving scans. | Local-only backend binding (`127.0.0.1`); strict CORS restricted to localhost. No telemetry. | **Verified** in FastAPI config. |

---

## 4. Getting Started on Windows (PowerShell)

### Prerequisites
- **Python**: Version 3.10+ (Tested on Python 3.14.7)
- **Node.js**: Version 18+ (Tested on Node.js v24)
- **Git** (optional)

### Step 1: Virtual Environment Execution Policy (Windows Fix)
If PowerShell blocks virtual environment activation script execution (`PSSecurityException`), run this command in your PowerShell terminal:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
```

### Step 2: Set Up Backend
Open a PowerShell terminal and navigate to the backend folder:
```powershell
cd c:\Users\somnath\sih2026\cortex-ai\backend

# Install required Python dependencies
pip install -r requirements.txt

# Run automated pytest verification suite (all 25 tests should pass)
python -m pytest tests/

# Start FastAPI backend server
python run_server.py
```
Backend will start at: `http://127.0.0.1:8000` (API documentation available at `http://127.0.0.1:8000/docs`).

### Step 3: Set Up and Start Frontend
Open a **second** PowerShell terminal and navigate to the frontend folder:
```powershell
cd c:\Users\somnath\sih2026\cortex-ai\frontend

# Install dependencies (only needed once)
npm install

# Start Vite dev server
npm run dev
```
Open your browser and navigate to: `http://localhost:5173`

---

## 5. Testing the Prototype (Quick Walkthrough)

1. **Open Dashboard**: Go to `http://localhost:5173`.
2. **Instant Test Scan**: Click **"Launch With Built-in Sample Scan"** on the Dashboard or Upload panel. The server generates a realistic 3D synthetic brain MRI volume with gray matter, white matter, ventricles, and realistic T1w contrast.
3. **Inspect Slices**: Navigate the **MRI Viewer**. Toggle between **Axial**, **Coronal**, and **Sagittal** views. Observe the anatomical orientation markers (`L/R`, `A/P`, `S/I`).
4. **Execute Segmentation**: In the Pipeline panel, select **"Synthetic Demonstration Pipeline"** and click **Execute Pipeline**. (Or observe the honest *Model Not Configured* message if SynthSeg is selected without external binaries).
5. **Overlay Inspection**: View the anatomical label map overlaid on the MRI slices with opacity slider controls and interactive LUT legend.
6. **Volumetric Measurements**: Switch to the **Volumetric Measures** tab. Review structure volumes in mm³ and mL, calculated with `count × |det(spatial affine)|`.
7. **Quality Control**: Open **Quality Control** to review independent input checks, geometry determinant, and image quality metrics (SNR, CNR, entropy).
8. **Ground Truth Evaluation**: Click **"Evaluate Against Sample Reference"** in the **Ground Truth Eval** tab to calculate per-structure Dice similarity and HD95 surface distances in millimeters.
9. **Export Report**: Open **Reports & Export** to preview and download traceable HTML and CSV reports. Notice the mandatory **DEMO DATA** stamps.
10. **Clean Session**: Click the trash icon in the header to trigger guaranteed server-side temporary file deletion.

---

## 6. Integrating Genuine Models (SynthSeg / FastSurfer)

Cortex AI provides pluggable adapter interfaces for external deep learning pipelines:
- **SynthSeg (Billot et al. 2023)**:
  - Repository: `https://github.com/BBillot/SynthSeg`
  - Setup: Install SynthSeg CLI in a dedicated environment, ensure `synthseg` is in system PATH.
- **FastSurfer (Henschel et al. 2020)**:
  - Repository: `https://github.com/Deep-MI/FastSurfer`
  - Recommended setup on Windows: Run via Docker or WSL2 container (`deepmi/fastsurfer:latest`).

---

## 7. Automated Test Summary

The backend includes 25 automated unit tests covering:
- `test_geometry.py`: Isotropic, anisotropic, and sheared/oblique affine determinant volume calculations, overlay geometry alignment gates.
- `test_volumes.py`: Physical volume conversions, small structure partial volume warnings (<50 voxels), empty label handling.
- `test_metrics.py`: Dice similarity (both-empty undefined policy, disjoint, identical), HD95 in physical millimeters, per-label distribution checks.
- `test_validators.py`: Rejection of 2D files, 4D multi-frame series, uniform/empty volumes, corrupted/renamed files.
- `test_demo_marker.py`: Permanent `DEMO DATA` stamps in metadata, CSV exports, and HTML reports.
- `test_cleanup.py`: Guaranteed deletion of temporary upload directories and session files.
