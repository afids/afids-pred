# Figure captions

Generated from `paper/submission/AFIDsPred_Manuscript_v5.docx` and `paper/submission/AFIDsPred_Supplementary_Material_v5.docx` by `paper/build/export_captions.py`. Edit the captions in the build scripts, not here.

## Main figures

### Figure 1. Coordinate-to-coordinate framework for brain target localization.

File: `Figure_1.png`

(a) anatomical fiducial (AFID) coordinates after AC-PC alignment, centering on the mid-commissural point (MCP), and left-to-right reflection, shown as an oblique three-dimensional point cloud (11 landmark types; 320 hemisphere examples from 160 development scans). Large dark points mark landmark means and dashed lines project each mean onto the floor grid. Colored points are ground-truth centroids of the four targets: STN (coral; 94 hemispheres from 47 participants), cZI (purple; 64 from 32), fct (gold; 64 from 32), and RN (teal; 62 from 31). (b) the circled target cluster magnified on an axial slice of the 100-µm 7-T ex vivo human brain (Edlow et al., 2019; synthesized FLASH25 image resampled to 200 µm in MNI152NLin2009bAsym space), with outlines of this cohort's target segmentations (probability ≥ 0.5); target z coordinates are collapsed in this projection, circles denote right and squares left hemispheres, and the overlay is anatomical context only. (c) schematic of the ground-truth definition, in which each target is segmented by at least two raters on 7-T T2-weighted MRI (three shown), combined by majority vote, and reduced to its center of mass. (d) learning pipeline. The 33 AFID coordinates are standardized and projected by PCA (23 components in the exploratory analysis shown; 24 in the deployed STN model), and ten candidate regression architectures are screened under nested cross-validation (shown for STN: 4 outer × 3 inner folds; cZI, fct, and RN use leave-one-participant-out outer folds). The table lists the architecture selected for each target and its median Euclidean error.

### Figure 2. Correlation of landmarks and principal component analysis (PCA).

File: `Figure_2.png`

(a) Pearson correlations among 28 non-degenerate AFID coordinate features, grouped by landmark, and their correlations with right-STN x, y, and z (right columns). AFID–AFID correlations use 320 hemisphere examples; AFID–STN correlations use 47 STN-labelled training participants retained after outlier processing. AC x/z, PC x/z, and PMJ x are fixed by alignment and excluded. (b) Development examples projected onto the first three standardized principal components (color encodes PC1 score); below, cumulative explained variance reaches 99% at 23 components. This exploratory PCA differs from the PCA fitted with the additional AFID-only pool in the deployed model, which retains 24 components. (c) Landmark configurations reconstructed at −2 SD (open) and +2 SD (filled) along PC1 and PC2, with other component scores set to zero, shown in the y–z plane on a sagittal slice of the 7-T ex vivo brain (Edlow et al., 2019); segments connect the same landmark. (d) Contribution of each landmark coordinate to the PCA, summed across components as absolute loading multiplied by explained-variance ratio. These values describe variance structure, not supervised predictive importance (see Supplementary Figure S6).

### Figure 3. Held-out STN localization across acquisitions and against registration.

File: `Figure_3.png`

(a) Prediction residuals relative to each hemisphere's reference STN location, projected onto the y–z plane, for the clinical pair (1.5 T with gadolinium and 7 T; LHSCPD/SNSX) and the external pair (3 T and 7 T; 3T-7T). Each acquisition contributes 20 hemispheres from 10 participants. The dashed circle marks a projected distance of 1 mm; x error is not shown. (b) Empirical cumulative distributions of three-dimensional Euclidean error for the four acquisitions. (c) Within-participant differences in Euclidean error (1.5 T − 7 T and 3 T − 7 T). Points are paired hemispheres, shaded shapes kernel density estimates, thick bars interquartile ranges, and open circles medians; p values are paired Wilcoxon signed-rank tests (1.5 T vs. 7 T, p = 0.12; 3 T vs. 7 T, p = 0.15). A non-significant difference does not establish equivalence. (d) Paired Euclidean errors of atlas-based deformable registration (ANTs/Lead-DBS) and AFID-based prediction on the clinical 1.5-T gadolinium-enhanced and 7-T acquisitions (n = 20 hemispheres each); gray lines connect matched hemispheres. (e) Registration error minus AFID error (positive values favour the AFID model); paired Wilcoxon signed-rank tests, 1.5 T with gadolinium p < 0.001 (p = 0.00085), 7 T p = 0.60.

### Figure 4. Robustness to annotation variability and extension to additional targets.

File: `Figure_4.png`

(a) STN predictions from 38 independent AFID annotation sets by 10 raters on the MNI152NLin2009bAsym template (76 hemisphere predictions; coral points) and the reference STN coordinates (stars), overlaid on an axial slice of the 7-T ex vivo brain (Edlow et al., 2019) with the STN atlas (gray; DISTAL). (b) Mean AFID annotation error versus STN prediction error (circles, right; squares, left hemisphere; Pearson r = 0.47, p < 0.001; ordinary least-squares slope = 0.46). The shaded band is the 95% confidence interval of the fitted mean and does not account for repeated raters and hemispheres. (c) Empirical cumulative error distributions under leave-one-participant-out cross-validation for the selected architecture of each additional target: ElasticNet for cZI (64 hemisphere estimates, 32 folds; median 0.81 mm), RandomForest for fct (64, 32 folds; median 0.77 mm), and Lasso for RN (62, 31 folds; median 0.46 mm). (d) Median cross-validated error of all ten candidate architectures for each target; outlined cells mark the lowest median per target. Model selection and error estimation use the same cross-validation, without independent held-out confirmation, so these values should not be compared directly with the STN held-out results.

## Supplementary figures

### Supplementary Figure S1. The 16-AFID protocol subset used in this study.

File: `Figure_S1.png`

(a) Sagittal, coronal and axial views of the MNI152NLin2009bAsym template with the 16 retained AFIDs, numbered by their AFIDs protocol index and colored by anatomical group (commissural, midbrain, brainstem, diencephalic). Points not on a displayed plane are projected onto it. (b) Protocol number, abbreviation and full name of each landmark. Positions are the AFIDs ground-truth annotation on the template (Lau et al., 2019).

### Supplementary Figure S2. Model comparison across all four coordinate-to-coordinate targets.

File: `Figure_S2.png`

For each target (subthalamic nucleus [STN], caudal zona incerta [cZI], fasciculus cerebellothalamicus [fct], and red nucleus [RN]), ten candidate regression models were screened with the same preprocessing pipeline and model roster (Supplementary Tables S2–S5). Each row shows one model's median Euclidean distance (dot), interquartile range (thick segment) and full range (thin line) over hemisphere estimates; the selected model (lowest median ED) is drawn in the target's color and its median is printed above it. Ranges extending beyond the plotted axis (MLP) are truncated with an arrow and their true maximum is given. STN: 100 hemisphere estimates from 50 participants, each averaged over 5 repeats of 4-fold participant-grouped cross-validation. cZI and fct: 64 hemisphere estimates from 32 participants (32-fold leave-one-participant-out). RN: 62 hemisphere estimates from 31 participants (31 folds). Holm p: Holm-Bonferroni-corrected paired Wilcoxon signed-rank tests against the selected model, computed on participant means for STN (n = 50) and on hemisphere estimates for cZI, fct and RN. These are internal cross-validation results; the STN held-out test is reported in main-text Figure 3.

### Supplementary Figure S3. Ridge regression versus XGBoost for the primary STN model, paired per participant.

File: `Figure_S3.png`

Squared error along x, y and z and Euclidean distance (ED) for each of the 50 training participants (hemispheres averaged; out-of-fold predictions, 5 × 4-fold participant-grouped cross-validation). Lines connect a participant's two models; thick bars show the IQR and ticks the median. p values are paired Wilcoxon signed-rank tests, Bonferroni corrected across the four metrics.

### Supplementary Figure S4. Leave-one-AFID-out landmark reconstruction.

File: `Figure_S4.png`

Each of the 11 encoded landmarks was predicted (x, y, z) from the remaining ten with the STN pipeline (StandardScaler → PCA 99% → Ridge), using 5 × 4-fold cross-validation grouped by participant on the 50 training participants. Points are the 100 hemisphere examples (ED averaged over repeats); shapes are kernel density estimates, thick bars the IQR and open circles the median, which is also printed above each landmark. Landmarks are ordered by median error; LVAC and LVPC (ventricular, right of the dashed line) are excluded from the pooled midbrain summary (nine landmarks: median 0.80 mm, IQR 0.34–1.33 mm; seven landmarks excluding AC and PC: median 1.00 mm, IQR 0.67–1.53 mm). *AC and PC are exact mirror images in AC–PC/MCP space (ACy = −PCy; x = z = 0), so each is reconstructed trivially from the other. Errors above 5 mm are drawn at 5 mm.

### Supplementary Figure S5. Sensitivity of the STN pipeline.

File: `Figure_S5.png`

(a) Median ED against the number of retained principal components, for PCA thresholds of 80–99.9% and without PCA. The production threshold (99%) is circled. (b) Held-out median ED against the number of training participants (25 random draws per size; line, median over draws; band, 10th–90th percentile). Test values use the 80 held-out STNs.

### Supplementary Figure S6. Landmark importance for STN prediction.

File: `Figure_S6.png`

(a) Change in cross-validated median ED when each of the 11 encoded landmarks is removed and the pipeline refit (positive values indicate the landmark helps). (b) Outer-fold ED against subset size when greedy forward selection is nested inside cross-validation (line, median; band, IQR). Selection uses only the outer training fold, so the curve is not biased by selection.

## Anatomical sources

- MNI152NLin2009bAsym T1/T2 templates: ICBM 2009b asymmetric distribution (https://www.bic.mni.mcgill.ca/~vfonov/icbm/2009/mni_icbm152_nlin_asym_09b_nifti.zip; Fonov et al., 2009).
- STN surfaces and contours in Figure 4a: DISTAL atlas distributed with Lead-DBS (Ewert et al., 2018), thresholded at probability 0.5.
- Figure 1 uses a 200-µm 7-T template (`Synthesized_FLASH25_in_MNI_v2_200um.nii.gz`, MNI152NLin2009bAsym space), which is not in this repository, and target masks from the SNSX32NLin2020Asym template in MNI152NLin2009bAsym space (`resources/segmentations_binary_MNIspace/`). A provenance citation for both is an open author item (docs/OPEN_ISSUES.md).
