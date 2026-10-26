# AFIDs-Pred

Code, coordinate data, trained model, figures and manuscript files for:

> Taha A., Thrower J., Bansal D., Thurairajah A., Abbass M., Snyder M., Zhao J., Coskun E., Liu V., Gilmore G., Khan A. R., Lau J. C.
> *A coordinate-to-coordinate machine learning framework for contrast-agnostic brain target localization.* (manuscript submitted)

Citation metadata is in [`CITATION.cff`](CITATION.cff).

## Summary

AFIDs-Pred predicts the location of a deep brain target from the coordinates of 16 anatomical fiducials (AFIDs) that a rater places on any MRI. The target is never segmented on the image, so the prediction does not depend on image contrast. The AFIDs are aligned to AC-PC space and centred on the mid-commissural point (MCP). The pipeline is StandardScaler → PCA (99% variance) → one ridge regression per axis.

The model was trained on 7 T data from the SNSX dataset. On the held-out test set it localized the subthalamic nucleus (STN) with a median Euclidean error of 1.20 mm [IQR 0.84–1.53] (80 STNs). The test set covers 1.5 T gadolinium-enhanced clinical scans, 3 T and 7 T. On the clinical 1.5 T scans the model outperformed deformable registration. The same framework was extended to the caudal zona incerta (cZI), the fasciculus cerebellothalamicus (fct) and the red nucleus (RN). The STN model is deployed at [stereotaxy.afids.io](https://stereotaxy.afids.io).

## Repository map

| Path | Contents |
|---|---|
| `data/coordinate/AFIDs/<dataset>/` | AFID ground truth: `afids_groundtruth/` (native space) and `afids_mcp/` (AC-PC/MCP space, with the ITK transform `.txt`) |
| `data/coordinate/STN/<dataset>/` | STN ground truth (`stn_mcp/`, plus native `stn_groundtruth/` where available) and per-rater STN centroids/volumes |
| `data/coordinate/{train,test}_data.csv` | Participant split used for training and held-out testing |
| `data/secondary_targets_thrower19/` | cZI / fct / RN ground-truth centroids and the original training notebooks (vendored, see below) |
| `data/*.csv` | Analysis outputs: cZI/fct/RN screens (`*_logocv_results_unified.csv`, `table_*_final_harmonized.csv`), STN model comparison, landmark subsets |
| `data/supplementary/` | Supplementary STN analyses (model screen, baselines, bootstrap CI, sensitivity, landmark analyses) |
| `notebooks/` | Analysis notebooks 2.1–2.6, the scripts that regenerate the reported numbers, and the `afidutils` / `afids_pipeline` modules |
| `resources/models/afids_stn_model_all_data.pkl` | Deployed STN model |
| `resources/MNI152NLin2009bAsym/` | AFID/STN annotations on the MNI152NLin2009bAsym template (ground truth and repeated rater placements) |
| `resources/LeadDBS/` | Registration-based STN coordinates (Lead-DBS) used for the benchmark |
| `resources/segmentations_binary_MNIspace/` | MNI-space binary masks of the targets (used for figures) |
| `figures/manuscript/`, `figures/source/`, `figures/source_data/` | Figures, the code that renders them, and the exact data they plot |
| `data/QC/` | Supplementary Attachment A: interactive HTML quality-control reports for the AFID and STN annotations |
| `figures/all_coordinates_interactive.html` | Supplementary Attachment B: interactive 3D visualization of all coordinates |
| `reproduce.sh`, `docs/REPRODUCIBILITY.md` | Pipeline runner and documentation of what reproduces from this repository |

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Predict both STNs from an AFID file that is already in AC-PC/MCP space:

```python
import sys
sys.path.insert(0, "notebooks")
import afidutils

stn = afidutils.predict_stn_coordinates(
    "data/coordinate/AFIDs/3T7T_7T/afids_mcp/sub-R01/anat/sub-R01_space-MCPPMJ_desc-groundtruth_afids.fcsv",
    "resources/models/afids_stn_model_all_data.pkl",
)
right = stn[0]
left = stn[1] * [-1, 1, 1]   # the model predicts in a mirrored hemisphere frame; flip x for the left STN
```

The input must be a 3D Slicer `.fcsv` with the 32 AFIDs in protocol order. Only the 16 AFIDs of the modified protocol (Supplementary Figure S1) are used. Coordinates are in millimetres, RAS orientation, AC-PC aligned and centred on the MCP.

To start from native-space AFIDs:
1. Run `afidutils.transform_afids(native_fcsv, output_dir=..., save_matrix=True)`. It writes the MCP-space `.fcsv` and the transform, and it expects BIDS-style file names, as in `data/coordinate`.
2. Map the predictions back with the inverse of `afidutils.acpcmatrix(native_fcsv, "PMJ")`.

To regenerate the reported numbers, run `./reproduce.sh`. See [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) for the run order, runtimes, and which outputs map to which table and figure.

## Data

Folder names under `data/coordinate/` map to the datasets in the manuscript (Table 1) as follows:

| Manuscript dataset | Folders | Notes |
|---|---|---|
| SNSX (7 T; 32 HC, 30 PD) | `SNSXCN`, `SNSXPD` | Training and held-out testing. AFIDs and STN. Western University, REB #109045 |
| LHSCPD (1.5 T + gadolinium; 10 PD, paired with SNSX) | `SNSXPD1.5T` | Clinical validation. AFIDs only; STN reference from the paired 7 T scan. LHSC, REB #R-17-156 |
| 3T-7T (10 HC, Chen et al., 2023) | `3T7T_3T`, `3T7T_7T` | External validation. STN annotated on 7 T only and used as the reference for both scans |
| afids-data subset (Taha et al., 2023; n = 110) | `AFIDs-HCP`, `AFIDs-OASIS`, `ADNI`, `LHSCPD` | AFIDs only. Together with two SNSX PD participants that are in neither split (sub-P010, sub-P088), these subjects form the AFID-only PCA pool (112 participants). The `LHSCPD` folder is the afids-data LHSC PD cohort and is not the 1.5 T LHSCPD test set above |
| MNI152NLin2009bAsym template | `resources/MNI152NLin2009bAsym/` | Ground truth plus repeated rater placements (annotation-sensitivity analysis) |

**Third-party data.**
- `data/secondary_targets_thrower19/` is a copy of the cZI, fct and RN ground-truth centroids and training notebooks from [github.com/thrower19/all_target_model_data](https://github.com/thrower19/all_target_model_data) by J. Thrower (co-author). It is redistributed here for reproducibility. The upstream repository states no licence.
- The 3T-7T coordinates are derived from the openly released paired 3 T/7 T dataset of Chen X., Qu L., Xie Y., Ahmad S., Yap P.-T. (2023), *A paired dataset of T1- and T2-weighted MRI at 3 Tesla and 7 Tesla*, Scientific Data 10. Cite that paper when using these coordinates.
- The afids-data coordinates come from Taha A. et al. (2023), *Magnetic resonance imaging datasets with anatomical fiducials for quality control and registration*, Scientific Data 10:449. The underlying HCP, OASIS and ADNI images remain under their original data use terms.

## Data availability

This repository contains every coordinate file, analysis output, trained model and figure source file used in the study.
- **Patient MRI is not shared.** This covers the SNSX and LHSCPD images and the Lead-DBS derivatives computed from them, under the terms of the ethics approvals.
- **Supplementary Attachments A and B** are in `data/QC/` and `figures/all_coordinates_interactive.html`.
- **Large template volumes** used only as figure backgrounds are not redistributed: the 1.1 GB 200-µm FLASH template and the MNI ICBM152 2009b high-resolution template.

[`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) lists exactly which steps need these files.

## Licence

Code: MIT (see `LICENSE`). Third-party data keep the terms of their sources (see above).
