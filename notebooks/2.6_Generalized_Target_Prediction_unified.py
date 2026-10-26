"""
2.6 Generalized Target Prediction - UNIFIED methodology for cZI, fct, RN.

Purpose
-------
The cZI/fct/RN secondary-target models were originally trained (by a different
collaborator, see thrower19/all_target_model_data on GitHub) with a
methodology that differed from the primary STN model
(notebooks/2.4_Data_Split_and_ML.ipynb) in several important ways: per-model
bespoke preprocessing (some models got PCA + scaling, some didn't, some used
fixed/ad hoc hyperparameters instead of tuned ones), a non-identical 10-model
roster, a buggy block-order GroupKFold grouping (afidutils.create_list, which
silently leaked mirrored hemisphere pairs across folds), and no retained
per-fold results (so no pairwise statistical model comparison was possible).

This script re-runs cZI/fct/RN with the SAME methodology as the (bug-fixed)
STN pipeline:
    - Single shared StandardScaler -> PCA(n_components=0.99) fit fresh within
      each training fold, augmented with an "other subjects" pool (all AFIDs
      subjects that are NOT part of the current target's cohort) for PCA
      stability -- exactly mirroring df_afids_proc_other in 2.4.
    - The exact same 10-candidate-model roster as STN's Table S5:
      Ridge, BayesianRidge, ElasticNet, RandomForest, Lasso, PLS, XGBoost,
      SVR, KNN, MLP.
    - Nested CV hyperparameter tuning for every model (per output axis),
      using the corrected subject-index-based GroupKFold grouping (NOT
      afidutils.create_list's block-order assumption).
    - Per-fold results retained, so a Holm-Bonferroni-corrected pairwise
      Wilcoxon signed-rank comparison vs. the winning model can be computed,
      exactly mirroring how data/tableS4_model_comparison_corrected.csv was
      produced for STN.

Deliberate, scientifically-motivated DIFFERENCE from STN kept in place:
    - Outer CV is 32-fold (31 for RN) Leave-One-Group-Out (LOGO-CV), not a
      4-fold GroupKFold. STN uses 4 outer folds because its training cohort
      is large (100 subjects); cZI/fct/RN only have ~32 healthy-control
      subjects, too few to split into a meaningful held-out 4-fold structure
      without badly underpowering both partitions. LOGO-CV is the
      appropriate choice at this sample size and is NOT a rigor shortfall.
    - No independent held-out test-set confirmation step (STN has one) --
      there is no spare data to hold out at N=32.

Raw ground-truth recovery
--------------------------
The 16-AFID input coordinates for these 32 subjects already exist locally
(data/coordinate/AFIDs/SNSXCN/{afids_groundtruth,afids_mcp}) -- confirmed to
be the same cohort used by the original collaborator (byte-identical native
T1w coordinates, same subject list). The cZI/fct/RN ground-truth centroid
targets are NOT local; they were recovered from
github.com/thrower19/all_target_model_data (vendored in this repository at
data/secondary_targets_thrower19/), which
stores them as raw native-T1w-space electrode-contact fcsv files derived from
expert segmentation. This script reconstructs the SAME MCP-aligned space
used by the primary STN pipeline by recovering the exact AC-PC+MCP affine
transform matrix used to produce each subject's local `afids_mcp` fcsv
(stored alongside it as an ITK/Slicer .txt transform file) and applying that
SAME per-subject matrix to the raw native-space cZI/fct/RN centroids. This
was verified: re-applying the recovered matrix to the raw native AFIDs
reproduces the local afids_mcp fcsv coordinates to ~1e-5 mm.

RN-specific note: sub-C008 is excluded (31 of 32 subjects, 31 outer folds),
matching the manuscript's stated "31 of 32 training subjects... 62 folds".
This was confirmed both from the original collaborator's notebook (explicit
`drop_for_now = ['sub-C008'] # EXCLUDED because MISSING right GT expert
annotation`) and independently from the raw data itself: sub-C008's RN
fcsv has a right-side value that is an exact mirror-image placeholder of the
left-side value (not a real expert annotation), unlike cZI/fct where sub-C008
has genuinely distinct L/R coordinates.
"""
import sys
import time
import json
import warnings
from pathlib import Path
from glob import glob

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, uniform, loguniform, randint
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import (
    GroupKFold, LeaveOneGroupOut, RandomizedSearchCV, GridSearchCV,
)
from sklearn.linear_model import (
    Ridge, RidgeCV, Lasso, LassoCV, ElasticNet, ElasticNetCV, BayesianRidge,
)
from sklearn.cross_decomposition import PLSRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS_DIR = REPO_ROOT / "notebooks"
sys.path.insert(0, str(NOTEBOOKS_DIR))
import afidutils  # noqa: E402

THROWER_REPO = REPO_ROOT / "data/secondary_targets_thrower19"

RANDOM_STATE = 42
INNER_K = 5            # inner-CV folds for hyperparameter tuning (documented choice, see report)
PCA_VARIANCE = 0.99


# --------------------------------------------------------------------------
# 1. Recover the AC-PC + MCP transform matrix used to build afids_mcp
# --------------------------------------------------------------------------
def parse_itk_affine_txt(path):
    """Recover the RAS 4x4 affine matrix (as used by transform_afids/coords @
    matrix.T convention) from a Slicer/ITK AffineTransform_double_3_3 .txt
    file, inverting the exact `generate_slicer_file` encoding used to write
    it (LPS<->RAS flip via D=diag(-1,-1,1,1), then matrix inversion)."""
    params = None
    with open(path) as f:
        for line in f:
            if line.startswith("Parameters:"):
                params = np.array([float(x) for x in line.split(":")[1].split()])
    rot = params[:9].reshape(3, 3)
    trans = params[9:12]
    lps = np.eye(4)
    lps[:3, :3] = rot
    lps[:3, 3] = trans
    D = np.diag([-1.0, -1.0, 1.0, 1.0])
    ras_inmatrix = D @ lps @ D
    matrix = np.linalg.inv(ras_inmatrix)
    return matrix


def get_subject_transform(subjid):
    txt_glob = glob(str(
        REPO_ROOT / "data/coordinate/AFIDs/SNSXCN/afids_mcp"
        / subjid / "anat" / "*.txt"
    ))
    assert len(txt_glob) == 1, f"expected exactly 1 transform file for {subjid}, got {txt_glob}"
    return parse_itk_affine_txt(txt_glob[0])


# --------------------------------------------------------------------------
# 2. Load raw native-space target centroids and transform to MCP space
# --------------------------------------------------------------------------
def load_raw_target_centroid(fcsv_path):
    """Parse a 2-row {target}_contacts.fcsv (r_<target>, l_<target> labels)
    and return native-space (r_xyz, l_xyz) as np arrays, keyed off the
    'label' column (not row position) for robustness."""
    df = pd.read_csv(fcsv_path, sep=",", header=2)
    r_row = df[df["label"].str.lower().str.startswith("r_")]
    l_row = df[df["label"].str.lower().str.startswith("l_")]
    assert len(r_row) == 1 and len(l_row) == 1, f"unexpected row layout in {fcsv_path}"
    r_xyz = r_row[["x", "y", "z"]].to_numpy()[0]
    l_xyz = l_row[["x", "y", "z"]].to_numpy()[0]
    return r_xyz, l_xyz


def apply_transform(matrix, xyz):
    h = np.append(xyz, 1.0)
    return (matrix @ h)[:3]


def build_target_subject_ids(target):
    """Alphabetically sorted subject ids with usable ground truth for this
    target. RN drops sub-C008 (verified mirror-placeholder right side)."""
    centroid_dir = THROWER_REPO / "training_data/training_targets" / f"{target}_GT_centroids"
    files = sorted(glob(str(centroid_dir / "sub-*.fcsv")))
    ids = []
    for f in files:
        sid = Path(f).name.split("_")[0]
        # Restrict to the local 32-subject SNSXCN cohort (same as STN's
        # AFID feature source); thrower19 repo has 2 extra subjects
        # (sub-C033/034) not present in the local SNSXCN afids_mcp cohort.
        if (REPO_ROOT / "data/coordinate/AFIDs/SNSXCN/afids_mcp" / sid).exists():
            ids.append(sid)
    ids = sorted(ids)
    if target == "RN":
        assert "sub-C008" in ids
        ids.remove("sub-C008")
    return ids


def build_target_dataframe(target, subject_ids):
    """Build df_target_proc: r-block (all subjects) then l-block (all
    subjects, x mirrored), indexed by subject id, columns x/y/z -- mirrors
    df_stn_proc's construction in 2.4 exactly."""
    centroid_dir = THROWER_REPO / "training_data/training_targets" / f"{target}_GT_centroids"
    rows_r, rows_l = [], []
    for sid in subject_ids:
        fcsv = centroid_dir / f"{sid}_{target}_contacts.fcsv"
        r_xyz, l_xyz = load_raw_target_centroid(fcsv)
        matrix = get_subject_transform(sid)
        r_mcp = apply_transform(matrix, r_xyz)
        l_mcp = apply_transform(matrix, l_xyz)
        rows_r.append(r_mcp)
        rows_l.append(l_mcp)
    df_r = pd.DataFrame(rows_r, index=subject_ids, columns=["x", "y", "z"])
    df_l = pd.DataFrame(rows_l, index=subject_ids, columns=["x", "y", "z"])
    df_l["x"] *= -1  # mirror left hemisphere x, matching flip_and_concatenate_hemispheres convention
    df_target_proc = pd.concat([df_r, df_l], axis=0)
    return df_target_proc


# --------------------------------------------------------------------------
# 3. Build the full multi-dataset AFID feature pool (same as 2.4 cell 3)
# --------------------------------------------------------------------------
def build_full_afids_pool():
    afids_path = REPO_ROOT / "data/coordinate/AFIDs"
    afids_fcsv = sorted(glob(str(afids_path / "*/afids_mcp/*/*/*MCP*.fcsv")))
    df_subs_afids = pd.concat([afidutils.fcsvtodf(f)[0] for f in afids_fcsv], axis=0)
    df_afids_proc = afidutils.flip_and_concatenate_hemispheres(df_subs_afids)
    return df_afids_proc


# --------------------------------------------------------------------------
# 4. Model roster + nested-CV hyperparameter search specs (per output axis)
# --------------------------------------------------------------------------
def fit_predict_axis(model_name, X_train, y_train_axis, X_test, inner_cv, n_pca_comp):
    """Nested-CV tune (where applicable) + fit + predict a single output axis
    for the given model family. Returns y_pred_axis (test-set predictions)."""
    if model_name == "Ridge":
        est = RidgeCV(alphas=np.logspace(-3, 6, 200), fit_intercept=True,
                       scoring="neg_mean_squared_error", cv=inner_cv)
        est.fit(X_train, y_train_axis)
        return est.predict(X_test)

    elif model_name == "Lasso":
        est = LassoCV(alphas=np.logspace(-3, 1, 50), cv=inner_cv,
                       max_iter=10000, random_state=RANDOM_STATE)
        est.fit(X_train, y_train_axis)
        return est.predict(X_test)

    elif model_name == "ElasticNet":
        est = ElasticNetCV(l1_ratio=[.1, .3, .5, .7, .9, .95, .99, 1.0],
                            alphas=np.logspace(-3, 1, 30), cv=inner_cv,
                            max_iter=10000, random_state=RANDOM_STATE)
        est.fit(X_train, y_train_axis)
        return est.predict(X_test)

    elif model_name == "BayesianRidge":
        # No meaningful fixed grid to nested-CV tune -- BayesianRidge's
        # regularization (alpha/lambda precision priors) is itself estimated
        # via evidence maximization during .fit(), i.e. it is already
        # "self-tuning". Fit directly with defaults (documented judgment
        # call; consistent with how it is commonly used).
        est = BayesianRidge()
        est.fit(X_train, y_train_axis)
        return est.predict(X_test)

    elif model_name == "PLS":
        max_comp = max(1, min(10, X_train.shape[1], X_train.shape[0] - 1))
        grid = {"n_components": list(range(1, max_comp + 1))}
        search = GridSearchCV(PLSRegression(), grid, cv=inner_cv,
                               scoring="neg_mean_squared_error", n_jobs=-1)
        search.fit(X_train, y_train_axis)
        return search.predict(X_test).ravel()

    elif model_name == "RandomForest":
        # NOTE: n_iter/n_estimators range reduced from an initial pilot
        # (randint(100,500), n_iter=20 -> ~51s/fold) to keep total runtime
        # tractable across 95 outer folds x 3 targets; still a real
        # randomized nested-CV search, not a fixed/ad hoc setting.
        dist = {
            "n_estimators": randint(50, 300),
            "max_depth": randint(2, 12),
            "min_samples_leaf": randint(1, 5),
            "max_features": uniform(0.3, 0.7),
        }
        search = RandomizedSearchCV(
            RandomForestRegressor(random_state=RANDOM_STATE),
            param_distributions=dist, n_iter=12, cv=inner_cv,
            scoring="neg_mean_squared_error", random_state=RANDOM_STATE, n_jobs=-1)
        search.fit(X_train, y_train_axis)
        return search.predict(X_test)

    elif model_name == "XGBoost":
        dist = {
            "n_estimators": randint(50, 400),
            "max_depth": randint(2, 10),
            "learning_rate": uniform(0.01, 0.3),
            "subsample": uniform(0.6, 0.4),
            "colsample_bytree": uniform(0.6, 0.4),
            "gamma": uniform(0, 3),
            "reg_alpha": uniform(0, 1),
            "reg_lambda": uniform(0, 2),
        }
        search = RandomizedSearchCV(
            XGBRegressor(objective="reg:squarederror", n_jobs=1,
                         random_state=RANDOM_STATE, verbosity=0),
            param_distributions=dist, n_iter=20, cv=inner_cv,
            scoring="neg_mean_squared_error", random_state=RANDOM_STATE, n_jobs=-1)
        search.fit(X_train, y_train_axis)
        return search.predict(X_test)

    elif model_name == "SVR":
        dist = {
            "C": loguniform(1e-2, 1e3),
            "epsilon": uniform(0.01, 1.0),
            "gamma": loguniform(1e-4, 1e1),
        }
        search = RandomizedSearchCV(
            SVR(kernel="rbf"), param_distributions=dist, n_iter=20, cv=inner_cv,
            scoring="neg_mean_squared_error", random_state=RANDOM_STATE, n_jobs=-1)
        search.fit(X_train, y_train_axis)
        return search.predict(X_test)

    elif model_name == "KNN":
        grid = {"n_neighbors": list(range(1, 11)), "weights": ["uniform", "distance"]}
        search = GridSearchCV(KNeighborsRegressor(), grid, cv=inner_cv,
                               scoring="neg_mean_squared_error", n_jobs=-1)
        search.fit(X_train, y_train_axis)
        return search.predict(X_test)

    elif model_name == "MLP":
        # NOTE: max_iter reduced from an initial pilot (max_iter=2000,
        # early_stopping=True -> ~45s/fold, mostly non-converging runs since
        # early_stopping's internal validation split is unreliable at N~30)
        # to max_iter=500 with early_stopping disabled, to keep total runtime
        # tractable across 95 outer folds x 3 targets. Still a real
        # randomized nested-CV search over architecture/regularization.
        dist = {
            "hidden_layer_sizes": [(16,), (32,), (16, 16), (32, 16), (64,)],
            "alpha": loguniform(1e-5, 1e-1),
            "learning_rate_init": loguniform(1e-4, 1e-1),
        }
        search = RandomizedSearchCV(
            MLPRegressor(max_iter=500, random_state=RANDOM_STATE, early_stopping=False),
            param_distributions=dist, n_iter=10, cv=inner_cv,
            scoring="neg_mean_squared_error", random_state=RANDOM_STATE, n_jobs=-1)
        search.fit(X_train, y_train_axis)
        return search.predict(X_test)

    else:
        raise ValueError(model_name)


MODEL_ROSTER = ["Ridge", "BayesianRidge", "ElasticNet", "RandomForest", "Lasso",
                "PLS", "XGBoost", "SVR", "KNN", "MLP"]


# --------------------------------------------------------------------------
# 5. Outer LOGO-CV loop
# --------------------------------------------------------------------------
def run_target(target):
    print(f"\n{'='*70}\nTarget: {target}\n{'='*70}")

    subject_ids = build_target_subject_ids(target)
    n_subj = len(subject_ids)
    print(f"  {n_subj} subjects with usable ground truth: {subject_ids[:3]}...{subject_ids[-3:]}")

    df_target_proc = build_target_dataframe(target, subject_ids)
    df_afids_full = build_full_afids_pool()

    # BUGFIX-consistent grouping: select via .loc on the (duplicate-indexed)
    # full AFID pool, then read groups directly off the resulting index --
    # do NOT use afidutils.create_list (block-order bug).
    df_afids_target = df_afids_full.loc[subject_ids]
    df_target_sel = df_target_proc.loc[subject_ids]
    groups = df_afids_target.index.to_numpy()

    # hemisphere labels, built with an identical .loc selection for perfect
    # row-order correspondence with X/y/groups
    hemi_series = pd.Series(
        ["R"] * len(subject_ids) + ["L"] * len(subject_ids), index=subject_ids * 2
    )
    hemi_labels = hemi_series.loc[subject_ids].to_numpy()

    X_other = df_afids_full.drop(index=subject_ids).to_numpy()

    X = df_afids_target.to_numpy()
    y = df_target_sel.to_numpy()

    assert X.shape[0] == y.shape[0] == len(groups) == len(hemi_labels) == 2 * n_subj

    logo = LeaveOneGroupOut()
    n_folds = logo.get_n_splits(groups=groups)
    print(f"  outer folds (LOGO-CV): {n_folds}")

    records = []
    t0 = time.time()
    for fold_idx, (train_idx, test_idx) in enumerate(
        tqdm(logo.split(X, y, groups=groups), total=n_folds, desc=f"{target} outer folds")
    ):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        inner_groups = groups[train_idx]
        hemi_test = hemi_labels[test_idx]

        # === Single shared preprocessing pipeline (StandardScaler -> PCA(0.99)) ===
        X_train_c = np.concatenate([X_train, X_other], axis=0)
        scaler = StandardScaler().fit(X_train_c)
        pca = PCA(n_components=PCA_VARIANCE).fit(scaler.transform(X_train_c))
        X_train_pca = pca.transform(scaler.transform(X_train))
        X_test_pca = pca.transform(scaler.transform(X_test))

        n_unique_inner = len(np.unique(inner_groups))
        inner_k = min(INNER_K, n_unique_inner)
        inner_cv = list(GroupKFold(n_splits=inner_k).split(X_train_pca, groups=inner_groups))

        for model_name in MODEL_ROSTER:
            y_pred = np.column_stack([
                fit_predict_axis(model_name, X_train_pca, y_train[:, ax], X_test_pca,
                                  inner_cv, pca.n_components_)
                for ax in range(3)
            ])
            err = y_test - y_pred
            ed = np.sqrt((err ** 2).sum(axis=1))
            for row_i in range(len(test_idx)):
                records.append({
                    "model": model_name,
                    "fold": fold_idx,
                    "hemisphere": hemi_test[row_i],
                    "ed": ed[row_i],
                    "err_x": err[row_i, 0],
                    "err_y": err[row_i, 1],
                    "err_z": err[row_i, 2],
                })

    elapsed = time.time() - t0
    print(f"  {target} done in {elapsed/60:.1f} min")
    df_results = pd.DataFrame(records)
    return df_results, elapsed


# --------------------------------------------------------------------------
# 6. Summary stats: median/mean+-SD, Holm-Bonferroni Wilcoxon vs winner
# --------------------------------------------------------------------------
def summarize(df_results):
    piv = df_results.pivot_table(index=["fold", "hemisphere"], columns="model", values="ed")
    means = piv.mean().sort_values()
    medians = piv.median()
    sds = piv.std()

    # Winner = lowest MEAN ED, per the task brief. Cross-checked against the
    # existing (old-methodology) cZI/fct/RN figures: figdata/build_figS3_fct
    # _10model.py hardcodes winner_model="RandomForest" for fct, which is
    # NOT the lowest-median model in data/fct_logocv_results.csv
    # (BayesianRidge has the lower median, 0.784 vs 0.828) but IS the
    # lowest-MEAN model (0.823 vs BayesianRidge's 0.832) -- confirming mean
    # ED, not median, is the actual established winner-selection convention
    # for these three targets. Both metrics are reported below regardless.
    winner = means.index[0]
    winner_scores = piv[winner].values

    rows = []
    pvals = []
    other_models = [m for m in means.index if m != winner]
    for m in other_models:
        diff = winner_scores - piv[m].values
        if np.allclose(diff, 0):
            p = 1.0
        else:
            _, p = wilcoxon(diff)
        pvals.append(p)

    _, p_holm, _, _ = multipletests(pvals, method="holm")
    holm_map = dict(zip(other_models, p_holm))

    for m in means.index:
        rows.append({
            "model": m,
            "median_ED": medians[m],
            "mean_ED": means[m],
            "sd_ED": sds[m],
            f"p_holm_vs_{winner.lower()}": np.nan if m == winner else holm_map[m],
        })
    return pd.DataFrame(rows), winner


def harmonized_table(df_results):
    """Manuscript-facing table (data/table_<target>_final_harmonized.csv):
    models sorted by median ED, winner = lowest MEDIAN ED, Holm-corrected
    Wilcoxon signed-rank p vs that winner, rounded for the supplement."""
    piv = df_results.pivot_table(index=["fold", "hemisphere"], columns="model", values="ed")
    stats = df_results.groupby("model").ed.agg(["median", "mean", "std"]).sort_values("median")
    winner = stats.index[0]
    others = [m for m in stats.index if m != winner]
    pvals = [wilcoxon(piv[winner] - piv[m]).pvalue for m in others]
    holm = dict(zip(others, multipletests(pvals, method="holm")[1]))
    return pd.DataFrame({
        "Model": stats.index,
        "Median ED": stats["median"].round(3).values,
        "Mean ED": stats["mean"].round(3).values,
        "SD": stats["std"].round(3).values,
        "Holm p (vs winner)": ["\u2014" if m == winner else round(holm[m], 4) for m in stats.index],
    })


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
if __name__ == "__main__":
    targets = sys.argv[1:] if len(sys.argv) > 1 else ["cZI", "fct", "RN"]
    data_dir = REPO_ROOT / "data"
    timing = {}

    for target in targets:
        df_results, elapsed = run_target(target)
        timing[target] = elapsed

        out_csv = data_dir / f"{target}_logocv_results_unified.csv"
        df_results.to_csv(out_csv, index=False)
        print(f"  saved {out_csv}")

        summary, winner = summarize(df_results)
        summary_csv = data_dir / f"tableS4_{target}_model_comparison_unified.csv"
        summary.to_csv(summary_csv, index=False)
        print(f"  saved {summary_csv}")
        print(f"  winner (lowest mean ED): {winner}")
        print(summary.to_string(index=False))

        harmonized_csv = data_dir / f"table_{target}_final_harmonized.csv"
        harmonized_table(df_results).to_csv(harmonized_csv, index=False)
        print(f"  saved {harmonized_csv}")

    print("\nTotal runtime (s):", json.dumps(timing))
