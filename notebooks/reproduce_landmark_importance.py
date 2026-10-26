"""Definitive AFID-landmark importance & subset analysis for STN prediction.

Settles "which landmarks matter and why" with three unbiased analyses on the
production pipeline (StandardScaler -> PCA(0.99) -> RidgeCV), all subject-grouped:

  1. ABLATION importance  - remove each landmark, measure increase in STN CV ED.
  2. NESTED-CV size curve  - select subset INSIDE inner folds, score on the held-out
     outer fold -> unbiased performance vs subset size (no selection bias).
  3. SELECTION STABILITY   - how often each landmark is chosen across outer folds /
     bootstrap resamples -> which landmarks are *consistently* important.

Run (from any directory):  python notebooks/reproduce_landmark_importance.py
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from glob import glob
from pathlib import Path
import sys
from collections import Counter
sys.path.insert(0, str(Path(__file__).resolve().parent))
import afidutils
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import GroupKFold

REPO = Path(__file__).resolve().parents[1]
BASE = REPO / "data/coordinate"
ALPHAS = np.logspace(-1, 3, 40)          # deployed alphas live ~4.6-11.6; this brackets them
LANDMARKS = ["AC", "PC", "ICS", "PMJ", "SIPF", "SLMS", "ILMS", "IMS", "MB", "LVAC", "LVPC"]
RNG = np.random.RandomState(42)


def load():
    stn = pd.concat([afidutils.fcsvtodf(s)[0] for s in sorted(glob(f"{BASE}/STN/*/stn_mcp/*/*/*MCP*.fcsv"))])
    stn.columns = ["RSTNx", "LSTNx", "RSTNy", "LSTNy", "RSTNz", "LSTNz"]; stn["LSTNx"] *= -1
    r = stn[["RSTNx", "RSTNy", "RSTNz"]].rename(columns=lambda c: c[1:])
    l = stn[["LSTNx", "LSTNy", "LSTNz"]].rename(columns=lambda c: c[1:])
    stn_p = pd.concat([r, l])
    af = pd.concat([afidutils.fcsvtodf(f)[0] for f in sorted(glob(f"{BASE}/AFIDs/*/afids_mcp/*/*/*MCP*.fcsv"))])
    return afidutils.flip_and_concatenate_hemispheres(af), stn_p


def cols_for(lms, all_cols):
    return [c for c in all_cols if any(c.startswith(lm) and c[len(lm):len(lm)+1] in "xyz" for lm in lms)]


def fold_ed(Xtr, ytr, Xte, yte, Xex):
    sc, pca = StandardScaler(), PCA(0.99)
    pca.fit(sc.fit_transform(np.vstack([Xtr, Xex])))
    A, B = pca.transform(sc.transform(Xtr)), pca.transform(sc.transform(Xte))
    pred = np.column_stack([Ridge(alpha=RidgeCV(alphas=ALPHAS).fit(A, ytr[:, j]).alpha_)
                            .fit(A, ytr[:, j]).predict(B) for j in range(3)])
    return list(np.sqrt(((yte - pred) ** 2).sum(1)))


def cv_median(lms, X, y, Xex, groups, all_cols):
    c = cols_for(lms, all_cols)
    eds = []
    for tr, te in GroupKFold(4).split(X, groups=groups):
        eds += fold_ed(X[c].values[tr], y[tr], X[c].values[te], y[te], Xex[c].values)
    return float(np.median(eds))


def main():
    df_af, df_stn = load()
    train = pd.read_csv(f"{BASE}/train_data.csv")["subjid"].tolist()
    test_all = pd.read_csv(f"{BASE}/test_data.csv")["subjid"].tolist()
    other = df_af.drop(index=[i for i in test_all if i in df_af.index]).drop(index=train)
    X = df_af.loc[train]; y = df_stn.loc[train].to_numpy()
    groups = afidutils.create_list(len(train)); cols = list(df_af.columns)

    full = cv_median(LANDMARKS, X, y, other, groups, cols)
    print(f"Full {len(LANDMARKS)}-landmark model: median CV ED = {full:.3f} mm\n")

    # 1. Ablation importance
    print("== 1. Ablation importance (ED increase when landmark removed) ==")
    abl = []
    for lm in LANDMARKS:
        ed = cv_median([l for l in LANDMARKS if l != lm], X, y, other, groups, cols)
        abl.append((lm, ed - full))
    for lm, d in sorted(abl, key=lambda t: -t[1]):
        print(f"  -{lm:5s}  dED = {d:+.3f} mm")

    # 2. Nested-CV size curve (unbiased): select on inner folds, score outer fold
    print("\n== 2. Nested-CV unbiased performance vs subset size ==")
    outer = list(GroupKFold(4).split(X, groups=groups))
    size_ed = {k: [] for k in range(1, len(LANDMARKS) + 1)}
    picks_per_fold = []
    for tr_idx, te_idx in outer:
        Xtr_o, ytr_o = X.iloc[tr_idx], y[tr_idx]
        gtr = [groups[i] for i in tr_idx]
        sel, remaining = [], LANDMARKS.copy()
        # greedy on inner CV of the OUTER-TRAIN only
        while remaining:
            scored = []
            for lm in remaining:
                c = cols_for(sel + [lm], cols)
                eds = []
                for itr, ite in GroupKFold(3).split(Xtr_o, groups=gtr):
                    eds += fold_ed(Xtr_o[c].values[itr], ytr_o[itr],
                                   Xtr_o[c].values[ite], ytr_o[ite], other[c].values)
                scored.append((np.median(eds), lm))
            scored.sort(); sel.append(scored[0][1]); remaining.remove(scored[0][1])
            # score this subset size on the held-out OUTER fold
            c = cols_for(sel, cols)
            ed_out = fold_ed(Xtr_o[c].values, ytr_o, X.iloc[te_idx][c].values, y[te_idx], other[c].values)
            size_ed[len(sel)] += ed_out
        picks_per_fold.append(sel)
    for k in range(1, len(LANDMARKS) + 1):
        print(f"  k={k:2d}: median outer ED = {np.median(size_ed[k]):.3f} mm")
    best_k = min(range(1, len(LANDMARKS) + 1), key=lambda k: np.median(size_ed[k]))
    print(f"  -> unbiased best size k={best_k} ({np.median(size_ed[best_k]):.3f} mm); full={full:.3f}")

    # 3. Selection stability: frequency each landmark appears in first-5 picks across folds
    print("\n== 3. Selection stability (appearance in top-5 across outer folds) ==")
    top5 = Counter()
    for sel in picks_per_fold:
        top5.update(sel[:5])
    for lm, c in top5.most_common():
        print(f"  {lm:5s}: {c}/{len(picks_per_fold)} folds")
    print(f"\n  per-fold first-5 picks: {[s[:5] for s in picks_per_fold]}")


if __name__ == "__main__":
    main()
