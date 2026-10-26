"""Supplementary analyses for the primary STN model.

All analyses use the production pipeline (StandardScaler -> PCA(0.99) -> RidgeCV,
PCA fit on the training partition + the AFID-only "other" pool) and group CV folds
by the TRUE subject ID read off the row index (never afidutils.create_list, whose
block-order labels leak mirrored hemisphere pairs when rows are paired-adjacent).

  screen     10-model STN screen (Table S2 / Figs S2-S3), same roster and per-model
             tuning as the cZI/fct/RN screens (2.6_..._unified.py), repeated
             4-fold GroupKFold outer CV, 3-fold GroupKFold inner CV.
  baselines  Held-out test set: model vs. training-mean and literature-consensus
             (12, -2, -4 mm from MCP) STN coordinates.
  pca        PCA variance-threshold sensitivity (CV + test).
  curve      Learning curve: test error vs. number of training subjects.
  ablation   Augmentation ablation (hemisphere mirroring, AFID-only PCA pool).
  landmarks  Landmark ablation importance + nested-CV subset-size curve.
  bootstrap  Person-level cluster bootstrap CI for the reported test-set median ED.
  loafo      Leave-one-AFID-out landmark reconstruction (Fig S5): each landmark's x,y,z
             predicted from the other 10 landmarks.

Run from notebooks/:  python supplementary_analyses.py [analysis ...]
Outputs: ../data/supplementary/*.csv
"""
import sys
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
from glob import glob

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, mannwhitneyu
from statsmodels.stats.multitest import multipletests
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import GroupKFold

import afidutils
from reproduce_test_set import load_training as load, BASE
from reproduce_landmark_importance import LANDMARKS, cols_for

import importlib.util
_spec = importlib.util.spec_from_file_location("unified", Path(__file__).parent / "2.6_Generalized_Target_Prediction_unified.py")
unified = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(unified)

OUT = Path("../data/supplementary"); OUT.mkdir(exist_ok=True)
ALPHAS = np.logspace(-3, 6, 200)
CONSENSUS = np.array([12.0, -2.0, -4.0])   # lateral, AP, SI relative to MCP (mm)
SEED = 42


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
def get_data():
    af, stn = load()
    train = pd.read_csv(f"{BASE}/train_data.csv")["subjid"].tolist()
    test_all = pd.read_csv(f"{BASE}/test_data.csv")["subjid"].tolist()
    other = af.drop(index=[i for i in test_all if i in af.index]).drop(index=train)
    X = af.loc[train]
    y = stn.loc[train].to_numpy()
    return af, stn, X, y, other, test_all


def test_cases(stn):
    """The 80 held-out STN evaluations (4 cohorts x 10 people x 2 hemispheres).
    1.5T-gad and 3T AFIDs are scored against the paired 7T STN labels, exactly as in
    manuscript_figure_set/source/prepare_data.py. Rows: (cohort, person, hemi, X row, truth)."""
    test_all = pd.read_csv(f"{BASE}/test_data.csv")["subjid"].tolist()
    rows = []
    for ds in ["SNSXPD", "SNSXPD1.5T", "3T7T_3T", "3T7T_7T"]:
        for f in sorted(glob(f"{BASE}/AFIDs/{ds}/afids_mcp/*/anat/*groundtruth*.fcsv")):
            s = f.split("/")[-3]
            if s not in test_all:
                continue
            g = s.replace("sub-R", "sub-P") if ds == "SNSXPD1.5T" else \
                s.replace("sub-", "sub-R") if ds == "3T7T_3T" else s
            d = afidutils.fcsvtodf(f)[0].map(afidutils.make_zero)
            Xs = afidutils.flip_and_concatenate_hemispheres(d)
            truth = stn.loc[g].to_numpy()          # row 0 = R, row 1 = L (mirrored)
            for h in range(2):
                rows.append(dict(cohort=ds, person=g, hemi="RL"[h],
                                 x=Xs.iloc[h].to_numpy(), t=truth[h]))
    assert len(rows) == 80
    return rows


def fit_ridge(Xtr, ytr, groups, Xextra=None, pca_var=0.99):
    """Production pipeline. pca_var=None disables PCA."""
    Xpca_fit = np.vstack([Xtr, Xextra]) if Xextra is not None else Xtr
    sc = StandardScaler().fit(Xpca_fit)
    pca = PCA(pca_var).fit(sc.transform(Xpca_fit)) if pca_var else None
    tf = (lambda A: pca.transform(sc.transform(A))) if pca else sc.transform
    A = tf(Xtr)
    inner = list(GroupKFold(3).split(A, groups=groups))
    models = [Ridge(alpha=RidgeCV(alphas=ALPHAS, scoring="neg_mean_squared_error", cv=inner)
                    .fit(A, ytr[:, j]).alpha_).fit(A, ytr[:, j]) for j in range(3)]
    ncomp = pca.n_components_ if pca else Xtr.shape[1]
    return (lambda B: np.column_stack([m.predict(tf(B)) for m in models])), ncomp


def test_ed(predict, cases):
    P = predict(np.vstack([c["x"] for c in cases]))
    T = np.vstack([c["t"] for c in cases])
    return np.sqrt(((P - T) ** 2).sum(1))


def cv_ed(X, y, groups, Xextra, pca_var=0.99, n_repeats=5):
    """Repeated grouped 4-fold CV; returns per-row ED averaged over repeats."""
    Xa = X if isinstance(X, np.ndarray) else X.to_numpy()
    acc = np.zeros(len(Xa)); rng = np.random.RandomState(SEED)
    for _ in range(n_repeats):
        perm_groups = pd.Series(groups).map(dict(zip(np.unique(groups), rng.permutation(len(np.unique(groups)))))).to_numpy()
        for tr, te in GroupKFold(4).split(Xa, groups=perm_groups):
            pred, _ = fit_ridge(Xa[tr], y[tr], groups[tr], Xextra, pca_var)
            acc[te] += np.sqrt(((pred(Xa[te]) - y[te]) ** 2).sum(1))
    return acc / n_repeats


def iqr(v):
    return f"{np.median(v):.2f} [{np.percentile(v,25):.2f}–{np.percentile(v,75):.2f}]"


# --------------------------------------------------------------------------
# analyses
# --------------------------------------------------------------------------
def screen(n_repeats=5):
    """10-model STN screen. Per-hemisphere ED is averaged over repeats; paired
    Wilcoxon tests use per-SUBJECT means (n=50; hemispheres are not independent)."""
    _, _, X, y, other, _ = get_data()
    Xa, groups, Xo = X.to_numpy(), X.index.to_numpy(), other.to_numpy()
    hemi = np.tile(["R", "L"], len(Xa) // 2)          # .loc[train] gives paired-adjacent rows
    assert (X.index[::2] == X.index[1::2]).all()
    rng = np.random.RandomState(SEED); ug = np.unique(groups)
    recs = []
    for rep in range(n_repeats):
        pg = pd.Series(groups).map(dict(zip(ug, rng.permutation(len(ug))))).to_numpy()
        for fold, (tr, te) in enumerate(GroupKFold(4).split(Xa, groups=pg)):
            Xc = np.vstack([Xa[tr], Xo])
            sc = StandardScaler().fit(Xc); pca = PCA(0.99).fit(sc.transform(Xc))
            A, B = pca.transform(sc.transform(Xa[tr])), pca.transform(sc.transform(Xa[te]))
            inner = list(GroupKFold(3).split(A, groups=groups[tr]))
            for m in unified.MODEL_ROSTER:
                P = np.column_stack([unified.fit_predict_axis(m, A, y[tr, j], B, inner, pca.n_components_)
                                     for j in range(3)])
                err = P - y[te]
                for i, r in enumerate(te):
                    recs.append(dict(model=m, repeat=rep, fold=fold, subject=groups[r], hemisphere=hemi[r],
                                     n_pcs=pca.n_components_, err_x=err[i, 0], err_y=err[i, 1], err_z=err[i, 2],
                                     ed=np.linalg.norm(err[i])))
            print(f"  repeat {rep} fold {fold} done", flush=True)
    df = pd.DataFrame(recs)
    df.to_csv(OUT / "stn_model_screen_per_fold.csv", index=False)

    # per-hemisphere (averaged over repeats) -> per-subject
    hem = df.groupby(["model", "subject", "hemisphere"])[["ed", "err_x", "err_y", "err_z"]] \
            .agg({"ed": "mean", "err_x": lambda v: (v ** 2).mean(), "err_y": lambda v: (v ** 2).mean(),
                  "err_z": lambda v: (v ** 2).mean()}).rename(columns={"err_x": "se_x", "err_y": "se_y", "err_z": "se_z"}) \
            .reset_index()
    hem.to_csv(OUT / "stn_model_screen_per_hemisphere.csv", index=False)
    subj = hem.groupby(["model", "subject"])[["ed", "se_x", "se_y", "se_z"]].mean().reset_index()
    piv = subj.pivot(index="subject", columns="model", values="ed")
    hpiv = hem.pivot_table(index=["subject", "hemisphere"], columns="model", values="ed")
    order = hpiv.median().sort_values().index.tolist()
    others = [m for m in order if m != "Ridge"]
    p = [wilcoxon(piv["Ridge"] - piv[m]).pvalue for m in others]
    holm = dict(zip(others, multipletests(p, method="holm")[1]))
    tab = pd.DataFrame([dict(model=m, median_ED=hpiv[m].median(), mean_ED=hpiv[m].mean(), sd_ED=hpiv[m].std(),
                             p_holm_vs_ridge=holm.get(m, np.nan)) for m in order])
    tab.to_csv(OUT / "tableS_stn_model_screen.csv", index=False)
    print(tab.to_string(index=False))

    # Ridge vs XGBoost per axis (Fig S5), per-subject paired, Bonferroni over 4 metrics
    rx = []
    for met in ["se_x", "se_y", "se_z", "ed"]:
        a = subj[subj.model == "Ridge"].set_index("subject")[met]
        b = subj[subj.model == "XGBoost"].set_index("subject")[met].loc[a.index]
        rx.append(dict(metric=met, ridge_median=a.median(), xgb_median=b.median(),
                       p=wilcoxon(a - b).pvalue, p_bonf=min(1, 4 * wilcoxon(a - b).pvalue),
                       frac_subjects_ridge_better=float((a < b).mean())))
    rx = pd.DataFrame(rx); rx.to_csv(OUT / "ridge_vs_xgb.csv", index=False); print(rx.to_string(index=False))

    # HC vs PD CV error (Ridge, per subject)
    r = piv["Ridge"]; hc, pd_ = r[r.index.str.startswith("sub-C")], r[r.index.str.startswith("sub-P")]
    hp = pd.DataFrame([dict(group="HC", n=len(hc), median_ED=hc.median()), dict(group="PD", n=len(pd_), median_ED=pd_.median()),
                       dict(group="MWU p", n=np.nan, median_ED=mannwhitneyu(hc, pd_).pvalue)])
    hp.to_csv(OUT / "hc_vs_pd_cv_error.csv", index=False); print(hp.to_string(index=False))


def baselines():
    """Model (deployed pkl == training-only pipeline) vs. coordinate baselines on the 80 test STNs."""
    _, stn, X, y, _, _ = get_data()
    pr = pd.read_csv("../figures/source_data/predictions.csv")
    pr = pr[pr.method == "AFIDs"].copy()
    T = pr[["truth_x", "truth_y", "truth_z"]].to_numpy()
    mean_xyz = y.mean(0)
    pr["ED_trainmean"] = np.linalg.norm(T - mean_xyz, axis=1)
    pr["ED_consensus"] = np.linalg.norm(T - CONSENSUS, axis=1)
    rows = []
    for name, d in [("All", pr)] + list(pr.groupby("dataset", sort=False)):
        per = d.groupby("subject")[["ED", "ED_trainmean", "ED_consensus"]].mean()   # person-level pairing
        rows.append(dict(cohort=name, n_stn=len(d), model=iqr(d.ED), train_mean=iqr(d.ED_trainmean),
                         consensus=iqr(d.ED_consensus),
                         p_model_vs_trainmean=wilcoxon(per.ED - per.ED_trainmean).pvalue,
                         p_model_vs_consensus=wilcoxon(per.ED - per.ED_consensus).pvalue))
    tab = pd.DataFrame(rows)
    tab.to_csv(OUT / "test_baselines.csv", index=False)
    pr.to_csv(OUT / "test_baselines_per_stn.csv", index=False)
    print("training-mean STN (MCP space):", np.round(mean_xyz, 2)); print(tab.to_string(index=False))


def pca_sensitivity():
    _, stn, X, y, other, _ = get_data()
    cases = test_cases(stn); g = X.index.to_numpy()
    rows = []
    for v in [0.80, 0.90, 0.95, 0.99, 0.995, 0.999, None]:
        pred, k = fit_ridge(X.to_numpy(), y, g, other.to_numpy(), v)
        cv = cv_ed(X, y, g, other.to_numpy(), v)
        te = test_ed(pred, cases)
        rows.append(dict(variance=("none (33 features)" if v is None else v), n_components=k,
                         cv_median_ED=np.median(cv), test_median_ED=np.median(te),
                         test_q25=np.percentile(te, 25), test_q75=np.percentile(te, 75)))
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(OUT / "pca_threshold_sensitivity.csv", index=False)


def learning_curve(n_draws=25):
    _, stn, X, y, other, _ = get_data()
    cases = test_cases(stn); subs = X.index.unique().to_numpy()
    rng = np.random.RandomState(SEED); rows = []
    for n in [8, 12, 16, 20, 25, 30, 35, 40, 45, 50]:
        for d in range(n_draws if n < len(subs) else 1):
            pick = rng.choice(subs, n, replace=False)
            Xs = X.loc[pick]
            pred, k = fit_ridge(Xs.to_numpy(), stn.loc[pick].to_numpy(), Xs.index.to_numpy(), other.to_numpy())
            te = test_ed(pred, cases)
            rows.append(dict(n_train_subjects=n, draw=d, n_components=k, test_median_ED=np.median(te),
                             test_mean_ED=te.mean()))
        print(n, np.median([r["test_median_ED"] for r in rows if r["n_train_subjects"] == n]), flush=True)
    pd.DataFrame(rows).to_csv(OUT / "learning_curve.csv", index=False)


def ablation():
    _, stn, X, y, other, _ = get_data()
    cases = test_cases(stn); g = X.index.to_numpy()
    right = np.arange(0, len(X), 2)                    # original (non-mirrored) right-hemisphere rows
    rows = []
    for mirror in [True, False]:
        for pool in [True, False]:
            idx = np.arange(len(X)) if mirror else right
            Xo = other.to_numpy() if pool else None
            Xs, ys, gs = X.to_numpy()[idx], y[idx], g[idx]
            pred, k = fit_ridge(Xs, ys, gs, Xo)
            te = test_ed(pred, cases)
            cv = cv_ed(Xs, ys, gs, Xo)
            rows.append(dict(mirroring=mirror, pca_pool=pool, n_train_rows=len(idx), n_components=k,
                             cv_median_ED=np.median(cv), test_median_ED=np.median(te),
                             test_q25=np.percentile(te, 25), test_q75=np.percentile(te, 75)))
            print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(OUT / "augmentation_ablation.csv", index=False)


def landmarks():
    """Same three analyses as reproduce_landmark_importance.py, with true-subject groups."""
    _, _, X, y, other, _ = get_data()
    g = X.index.to_numpy(); cols = list(X.columns)

    def fold_ed(c, tr, te, Xsrc=X, ysrc=y, gsrc=g):
        pred, _ = fit_ridge(Xsrc[c].values[tr], ysrc[tr], gsrc[tr], other[c].values)
        return list(np.sqrt(((pred(Xsrc[c].values[te]) - ysrc[te]) ** 2).sum(1)))

    outer = list(GroupKFold(4).split(X, groups=g))
    def cv_med(lms):
        c = cols_for(lms, cols); return float(np.median(sum((fold_ed(c, tr, te) for tr, te in outer), [])))

    full = cv_med(LANDMARKS)
    abl = pd.DataFrame([dict(landmark=lm, dED=cv_med([l for l in LANDMARKS if l != lm]) - full) for lm in LANDMARKS])
    abl["full_model_median_ED"] = full
    abl.sort_values("dED", ascending=False).to_csv(OUT / "landmark_ablation.csv", index=False)
    print(abl.sort_values("dED", ascending=False).to_string(index=False))

    size_ed = {k: [] for k in range(1, len(LANDMARKS) + 1)}; picks = []
    for tr_o, te_o in outer:
        Xo, yo, go = X.iloc[tr_o], y[tr_o], g[tr_o]
        inner = list(GroupKFold(3).split(Xo, groups=go))
        sel, rem = [], LANDMARKS.copy()
        while rem:
            scored = sorted((np.median(sum((fold_ed(cols_for(sel + [lm], cols), a, b, Xo, yo, go) for a, b in inner), [])), lm)
                            for lm in rem)
            sel.append(scored[0][1]); rem.remove(scored[0][1])
            c = cols_for(sel, cols)
            pred, _ = fit_ridge(Xo[c].values, yo, go, other[c].values)
            size_ed[len(sel)] += list(np.sqrt(((pred(X.iloc[te_o][c].values) - y[te_o]) ** 2).sum(1)))
        picks.append(sel)
    curve = pd.DataFrame([dict(k=k, median_outer_ED=np.median(v), q25=np.percentile(v, 25), q75=np.percentile(v, 75))
                          for k, v in size_ed.items()])
    curve.to_csv(OUT / "landmark_subset_size_nestedCV.csv", index=False); print(curve.to_string(index=False))
    pd.DataFrame(picks, index=[f"fold{i}" for i in range(4)]).to_csv(OUT / "landmark_selection_order.csv")
    print(pd.DataFrame(picks).to_string())


def bootstrap(n=10000):
    """Cluster bootstrap over the 20 test PEOPLE (1.5T/7T and 3T/7T scans of one person share a cluster)."""
    pr = pd.read_csv("../figures/source_data/predictions.csv")
    pr = pr[pr.method == "AFIDs"]
    by = {p: d.ED.to_numpy() for p, d in pr.groupby("subject")}
    people = list(by); rng = np.random.RandomState(SEED)
    boots = [np.median(np.concatenate([by[p] for p in rng.choice(people, len(people))])) for _ in range(n)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    res = dict(n_people=len(people), n_stn=len(pr), median_ED=pr.ED.median(), sd=pr.ED.std(),
               q25=pr.ED.quantile(.25), q75=pr.ED.quantile(.75), ci95_lo=lo, ci95_hi=hi,
               pct_under_2mm=float((pr.ED < 2).mean()))
    pd.DataFrame([res]).to_csv(OUT / "test_bootstrap_ci.csv", index=False); print(res)


def leave_one_afid_out(n_repeats=5):
    """Leave-one-AFID-out reconstruction: each of the 11 encoded landmarks is predicted (x, y, z)
    from the remaining 10 with the production pipeline (AFID-only pool joins the PCA fit, restricted
    to the predictor columns). 5 x 4-fold CV grouped by the TRUE participant ID (X index) on the 50
    training participants (100 rows), permuted exactly as in cv_ed.

    Degeneracy: in AC-PC/MCP space ACx = ACz = PCx = PCz = 0 and ACy = -PCy, so AC is an exact
    linear function of PC (and vice versa); 'determined_by_others' flags any target that the
    predictors reproduce exactly (OLS residual ~ 0), and 'target_dof' counts non-constant axes.
    Reference: the training-fold mean position of the landmark (paired Wilcoxon on participant
    means, Holm across the 11 landmarks)."""
    _, _, X, _, other, _ = get_data()
    g = X.index.to_numpy(); hemi = np.tile(["R", "L"], len(X) // 2)
    assert (X.index[::2] == X.index[1::2]).all()
    rows, summ = [], []
    for lm in LANDMARKS:
        tcols = [lm + a for a in "xyz"]; pcols = [c for c in X.columns if c not in tcols]
        Xa, ya, Xo = X[pcols].to_numpy(), X[tcols].to_numpy(), other[pcols].to_numpy()
        A1 = np.column_stack([np.ones(len(Xa)), Xa])
        ols_res = np.abs(ya - A1 @ np.linalg.lstsq(A1, ya, rcond=None)[0]).max()
        rng = np.random.RandomState(SEED); ug = np.unique(g); fold_eds, ncomp = [], []
        for rep in range(n_repeats):
            pg = pd.Series(g).map(dict(zip(ug, rng.permutation(len(ug))))).to_numpy()
            for fold, (tr, te) in enumerate(GroupKFold(4).split(Xa, groups=pg)):
                pred, k = fit_ridge(Xa[tr], ya[tr], g[tr], Xo); ncomp.append(k)
                err = pred(Xa[te]) - ya[te]; base_err = ya[tr].mean(0) - ya[te]
                ed = np.linalg.norm(err, axis=1); fold_eds.append(ed.mean())
                for i, r in enumerate(te):
                    rows.append(dict(landmark=lm, subject=g[r], hemisphere=hemi[r], repeat=rep, fold=fold,
                                     err_x=err[i, 0], err_y=err[i, 1], err_z=err[i, 2], ed=ed[i],
                                     ed_trainmean=np.linalg.norm(base_err[i])))
        d = pd.DataFrame([r for r in rows if r["landmark"] == lm])
        per = d.groupby(["subject", "hemisphere"])[["ed", "ed_trainmean"]].mean()   # mean over repeats
        summ.append(dict(landmark=lm, midbrain9=lm not in ("LVAC", "LVPC"), n_rows=len(per),
                         n_participants=per.index.get_level_values(0).nunique(),
                         target_dof=int((ya.std(0) > 1e-9).sum()), determined_by_others=bool(ols_res < 1e-6),
                         n_pcs_median=int(np.median(ncomp)), median_ED=per.ed.median(),
                         q25_ED=per.ed.quantile(.25), q75_ED=per.ed.quantile(.75), mean_ED=per.ed.mean(),
                         sd_ED=per.ed.std(), fold_mean_ED=np.mean(fold_eds), fold_sd_ED=np.std(fold_eds, ddof=1),
                         trainmean_median_ED=per.ed_trainmean.median(),
                         p_vs_trainmean=wilcoxon(*per.groupby(level=0)[["ed", "ed_trainmean"]].mean().T.to_numpy()).pvalue))
        print(summ[-1], flush=True)
    p_holm = multipletests([r["p_vs_trainmean"] for r in summ], method="holm")[1]   # participant means, n = 50
    for r, ph in zip(summ, p_holm):
        r["p_holm_vs_trainmean"] = ph
    rows = pd.DataFrame(rows); rows.to_csv(OUT / "leave_one_afid_out_per_row.csv", index=False)
    per = rows.groupby(["landmark", "subject", "hemisphere"])[["ed", "ed_trainmean"]].mean().reset_index()
    for name, keep in [("POOLED_9_midbrain", [l for l in LANDMARKS if l not in ("LVAC", "LVPC")]),
                       ("POOLED_7_midbrain_excl_AC_PC", [l for l in LANDMARKS if l not in ("LVAC", "LVPC", "AC", "PC")]),
                       ("POOLED_11_all", LANDMARKS)]:
        v = per[per.landmark.isin(keep)]
        summ.append(dict(landmark=name, n_rows=len(v), n_participants=v.subject.nunique(), median_ED=v.ed.median(),
                         q25_ED=v.ed.quantile(.25), q75_ED=v.ed.quantile(.75), mean_ED=v.ed.mean(), sd_ED=v.ed.std(),
                         trainmean_median_ED=v.ed_trainmean.median()))
    tab = pd.DataFrame(summ); tab.to_csv(OUT / "leave_one_afid_out.csv", index=False)
    print(tab.to_string(index=False))


ANALYSES = dict(screen=screen, baselines=baselines, pca=pca_sensitivity, curve=learning_curve,
                ablation=ablation, landmarks=landmarks, bootstrap=bootstrap, loafo=leave_one_afid_out)

if __name__ == "__main__":
    for a in (sys.argv[1:] or ANALYSES):
        print(f"\n===== {a} =====", flush=True)
        ANALYSES[a]()
