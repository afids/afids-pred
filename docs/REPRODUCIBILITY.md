# Reproducing the analyses


Every number in the manuscript and supplement is read from a CSV in `data/` or `figures/source_data/`. This page explains:
- how to regenerate those CSVs from the coordinate data in this repository;
- which output feeds which table or figure;
- what cannot be regenerated from the repository alone.

## Environment

The pipeline was tested with Python 3.11 in a clean virtual environment on macOS (arm64).

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

The deployed model (`resources/models/afids_stn_model_all_data.pkl`) was pickled with scikit-learn 1.6.1. Loading it with 1.8.0 raises an `InconsistentVersionWarning` that can be ignored: the StandardScaler, PCA and Ridge objects load unchanged, and their test-set predictions reproduce the reported errors exactly.

## Run order

`./reproduce.sh` runs the stages below in order. `./reproduce.sh <stage> ...` runs selected stages. Runtimes were measured on an Apple-silicon laptop that was shared with other jobs, so expect them to be shorter on an idle machine.

| # | Stage | Command | Writes | Runtime |
|---|---|---|---|---|
| 1 | `analyses` | `python notebooks/2.6_Generalized_Target_Prediction_unified.py` | `data/{cZI,fct,RN}_logocv_results_unified.csv`, `data/tableS4_*_model_comparison_unified.csv`, `data/table_*_final_harmonized.csv` | 1 h 42 min (cZI 31, fct 39, RN 32 min) |
| | | `python notebooks/reproduce_test_set.py` | prints the held-out test metrics | < 1 min |
| | | `python notebooks/verify_acpc_degeneracy.py` | prints only (why the AC and PC ablations are identical) | < 1 min |
| 2 | `figdata` | `python figures/source/prepare_data.py` | `figures/source_data/*.csv`, `verified_summary.json` | < 1 min |
| 3 | `supplementary` | `cd notebooks && python supplementary_analyses.py` | `data/supplementary/*.csv` | 1 h 27 min |
| 4 | `figures` | `python figures/source/render_figures.py`, then `python figures/source/verify.py` | `figures/manuscript/Figure_*.png`, `figures/source_data/Verification.json` | about 1 min when the external inputs are present |
| – | `notebooks` (not run by default) | `jupyter nbconvert --execute` on 2.1–2.6 and 2.x | executed copies in `executed_notebooks/` | about 2 h in total, mostly 2.4 and 2.5 |

**Ordering constraints.**
- `supplementary` reads `figures/source_data/predictions.csv`, so it runs after `figdata`.
- `figdata` reads the per-fold cZI/fct/RN results from stage 1.

**Regression guards.**
- `prepare_data.py` refuses to overwrite `predictions.csv` if the regenerated errors differ from the committed ones.
- `verify.py` re-checks several values against the committed results: the benchmark Wilcoxon p-values, sample sizes, fold counts and residual norms.

## Outputs → manuscript items

| Manuscript item | Source file(s) | Produced by |
|---|---|---|
| Table 2, §3.3 test-set ED (1.20 mm [0.84–1.53], 80 STNs) | `figures/source_data/predictions.csv`, `verified_summary.json` | `prepare_data.py` (also printed by `reproduce_test_set.py`) |
| Test-set bootstrap CI (1.01–1.38 mm), % < 2 mm | `data/supplementary/test_bootstrap_ci.csv` | `supplementary_analyses.py bootstrap` |
| Figure 1 | `figures/source_data/secondary_target_coords.csv`, `development_stn.csv` + external volumes | `prepare_data.py`, `render_figures.py` |
| Figure 2 (correlations, PCA) | `figures/source_data/fig2_*.csv` (from `data/coordinate/*/df_*_mcp_raw.csv`, notebook 2.3) | `prepare_data.py` |
| Figure 3a–c (acquisition pairs), 3d–e (Lead-DBS benchmark) | `figures/source_data/predictions.csv` (methods `AFIDs` / `Registration`) | `prepare_data.py` |
| Figure 4a–b (annotation sensitivity) | `figures/source_data/annotation_predictions.csv`, `annotation_coordinates.csv` | `prepare_data.py` |
| Figure 4c–d, §3.5 (cZI/fct/RN) | `figures/source_data/secondary_targets.csv` ← `data/*_logocv_results_unified.csv` | 2.6 unified script, `prepare_data.py` |
| Figure S1 | `figures/source_data/mni_landmarks.csv` + MNI template | `render_figures.py` |
| Table S2, Figures S2–S3 (STN 10-model screen, Ridge vs XGBoost) | `data/supplementary/tableS_stn_model_screen.csv`, `ridge_vs_xgb.csv`, `stn_model_screen_per_*.csv`; S3 also uses `data/*_logocv_results_unified.csv` | `supplementary_analyses.py screen` |
| Table S3 (coordinate baselines) | `data/supplementary/test_baselines.csv`, `test_baselines_per_stn.csv` | `supplementary_analyses.py baselines` |
| Table S4 / Figure S4 (PCA threshold, learning curve, augmentation) | `data/supplementary/pca_threshold_sensitivity.csv`, `learning_curve.csv`, `augmentation_ablation.csv` | `supplementary_analyses.py pca curve ablation` |
| Figure S5 (leave-one-AFID-out) | `data/supplementary/leave_one_afid_out*.csv` | `supplementary_analyses.py loafo` |
| Figure S6 (landmark importance) | `data/supplementary/landmark_ablation.csv`, `landmark_subset_size_nestedCV.csv`, `landmark_selection_order.csv` | `supplementary_analyses.py landmarks` |
| HC vs PD CV error (§3.3) | `data/supplementary/hc_vs_pd_cv_error.csv` | `supplementary_analyses.py screen` |
| Tables S5–S7 (cZI / fct / RN screens) | `data/table_{cZI,fct,RN}_final_harmonized.csv` | 2.6 unified script |

**Legacy files.** These files in `data/` come from earlier analysis versions and no script in this repository regenerates them:
- `ml_comparison_results.csv`
- `tableS4_model_comparison_corrected.csv`
- `tableS3_subset_size_nestedCV.csv`
- `landmark_importance_summary.csv`
- `landmark_loo_results.csv`
- `{cZI,fct,RN}_logocv_results.csv` and `*_logocv_results.csv` (cZI/fct/RN with the original per-model methodology)

They are kept for provenance. The current manuscript does not read them.

Two scripts are kept but superseded: `notebooks/reproduce_subset_selection.py` (writes `data/tableS3_subset_selection_corrected.csv`) and `notebooks/reproduce_landmark_importance.py`.
- Both still group CV folds with `afidutils.create_list`, which leaks mirrored hemisphere pairs.
- `supplementary_analyses.py landmarks` replaces them with true-participant groups (Figure S6). `reproduce.sh` does not run them.
- When re-run, `reproduce_subset_selection.py` regenerates its CSV byte for byte (22 min). `reproduce_landmark_importance.py` takes 1 min.

## Verified in a clean environment

The following were checked on a fresh venv built from `requirements.txt`.
- `reproduce_test_set.py` gives the reported values with the deployed model:
  - median test ED 1.20 mm [0.84–1.53], 98.8% < 2 mm, bootstrap CI 1.01–1.38 mm;
  - per cohort: SNSX 7 T 1.19, LHSCPD 1.5 T 1.32, 3T-7T 3 T 1.20, 3T-7T 7 T 1.08 mm.
- `prepare_data.py` regenerates every file in `figures/source_data/` byte for byte. The one exception is `fig2_stn_correlations.csv`, which differs by at most 2e-16.
- `verify.py` passes.
- `table_{cZI,fct,RN}_final_harmonized.csv` are regenerated byte for byte from the per-fold results.
- Re-running the 2.6 script regenerates the per-fold `{cZI,fct,RN}_logocv_results_unified.csv` and `tableS4_*_unified.csv` to within 2e-15. The rounded tables are byte-identical.
- Notebook 2.3 regenerates `df_afids_mcp_raw.csv` and `df_stn_mcp_raw.csv` byte for byte.
- `supplementary_analyses.py` (all analyses) regenerates every CSV in `data/supplementary/` byte for byte.
- With the external inputs present, `render_figures.py` reproduces Figures 3, S2, S3, S4, S5 and S6 pixel for pixel (S1 differs by at most 1 grey level). Figures 1, 2 and 4 differ in 0.01–3.8% of pixels from the committed PNGs; check that the committed PNGs were rendered from the current `render_figures.py`.
- `./reproduce.sh figdata supplementary figures docs` ran end to end in 88 min. `analyses` adds 1 h 42 min, so the full default run takes about 3.5 h on a busy machine.

## Not reproducible from this repository alone

| What | Why | Effect |
|---|---|---|
| Subject MRI (SNSX, LHSCPD; 3T-7T is public from Chen et al., 2023) | Patient data; not shareable under the ethics approvals | Notebook 2.2 cannot regenerate the QC HTML reports (Supplementary Attachment A). Its MRI cells are skipped unless `AFIDS_IMAGING_DIR` points to a local copy. The native → MCP transform itself needs only the `.fcsv` files (`afidutils.transform_afids`). |
| Lead-DBS derivatives (per-subject coregistration/normalization transforms) | Computed from patient MRI | Notebook 2.x is skipped unless `AFIDS_LEADDBS_DIR` is set. Its outputs, the registration-based STN coordinates, are included in `resources/LeadDBS/`, so the benchmark (Figure 3d–e) does reproduce. |
| Native-space STN segmentations and Dice overlap (§3.1 descriptives) | Segmentation volumes are not in the repository | The §3.1 AFID error, Dice and STN-volume statistics cannot be recomputed. Notebook 2.1 reports volumes from `STN_all.csv` only. |
| Figure background volumes: the 1.1 GB 200-µm Synthesized FLASH25 template, the MNI ICBM152 2009b high-resolution T1, the DISTAL STN/RN meshes (`atlas_sources/`), `reference_review/geometry.npz` | Too large, or third-party licences | `render_figures.py` reads them from `AFIDS_FIG_EXTERNAL` (default: the parent directory of the repository). Without them `reproduce.sh` skips rendering and keeps the committed figures. All plotted numbers are still regenerated in `figures/source_data/` and checked by `verify.py`. |
| Exact refit of the deployed model | The deployed `.pkl` comes from an earlier run of notebook 2.4 with scikit-learn 1.6.1. Its committed outputs list the alphas stored in the `.pkl` (11.59 / 4.60 / 9.95). Those alphas match the pre-fix fold grouping (`afidutils.create_list`), which split each participant's mirrored hemisphere pair across CV folds. Re-running that grouping today gives 10.88 / 4.47 / 9.21: close, but not identical. That grouping affected only how alpha was chosen; the held-out test participants were never used. | Re-executing 2.4 with the corrected grouping selects different alphas (x/y/z 26.0/5.3/25.5 instead of 11.6/4.6/9.9). The re-trained model gets a median test ED of 1.20 mm [0.90–1.55]; the deployed model gets 1.20 mm [0.84–1.53]. (The refit in `reproduce_test_set.py` uses the `afids_pipeline` alpha grid and gives 1.22 mm.) To keep the deployed file, notebook 2.4 writes `afids_stn_model_retrained.pkl`. All reported numbers use the deployed model. |
| Committed notebook outputs | Notebooks 2.4 and 2.5 were last executed before the grouping fix (see above) | Re-executed outputs of 2.4 (fold errors, alphas, Ridge vs XGBoost p-values) differ from the committed ones. The manuscript takes these comparisons from `data/supplementary/` (corrected grouping), not from the notebook outputs. |

## Notebooks, executed headless

I executed each notebook with `jupyter nbconvert --to notebook --execute` in a clean venv, writing the executed copies outside the repository. The runtimes below are from a heavily loaded machine.

| Notebook | Before path fixes | After fixes | Runtime | Notes |
|---|---|---|---|---|
| 2.1 Rater curated data | 2 errors: the STN glob started with `/../data`, which matched nothing | 0 errors | 25 s | Text outputs identical to the committed ones |
| 2.2 Annotation QC | 3 errors: MRI read from `/Users/.../afid_pred_imaging` | 0 errors; MRI cells skipped with a message unless `AFIDS_IMAGING_DIR` is set | 12 min (all of it in the final 3D plot/GIF cell) | Writes `figures/all_coordinates*.{html,gif}` (git-ignored) |
| 2.3 Coordinate processing | 0 errors | 0 errors | 1.5 min | Regenerates `df_afids_mcp_raw.csv` and `df_stn_mcp_raw.csv` byte for byte. Writes exploratory PNGs to `figures/` and CSVs to `notebooks/` (git-ignored) |
| 2.4 Data split and ML | 0 errors, but it overwrote the deployed model | 0 errors; writes `afids_stn_model_retrained.pkl` | 16–24 min | Fold errors and alphas differ from the committed outputs because of the grouping fix (see above) |
| 2.5 Robustness / generalizability | One cell ran past a 1 h per-cell timeout | 0 errors with no timeout | 42 min | Test-set metrics, annotation sensitivity and the Lead-DBS benchmark match the committed outputs to 4 decimals (median test ED 1.195 mm) |
| 2.6 Generalized target prediction (demo) | 1 error: `Ridge.alpha_` should be `Ridge.alpha` | 0 errors | 6 min | Writes `notebooks/afids_stn_generalized.pkl` (git-ignored) |
| 2.x Lead-DBS transforms | 2 errors: Lead-DBS derivatives read from `/Users/...` | 0 errors; skipped with a message unless `AFIDS_LEADDBS_DIR` is set | 12 s | Outputs are already in `resources/LeadDBS/` |
