"""Reconstructed greedy forward AFID-subset selection on the PRODUCTION pipeline.

Replaces the earlier subset analysis, which used raw/scaled features (no PCA) and
therefore reported errors on a different scale than the headline model. Here every
candidate subset is evaluated inside the production StandardScaler -> PCA(0.99) ->
RidgeCV pipeline with subject-grouped 4-fold CV, so the numbers are directly
comparable to the full-model median CV ED (~0.81 mm).

Run (from any directory):  python notebooks/reproduce_subset_selection.py
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from glob import glob
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import afidutils
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import GroupKFold

REPO = Path(__file__).resolve().parents[1]
BASE = REPO / "data/coordinate"
ALPHAS = np.logspace(-3, 6, 200)
LANDMARKS = ["AC", "PC", "ICS", "PMJ", "SIPF", "SLMS", "ILMS", "IMS", "MB", "LVAC", "LVPC"]


def load():
    stn_fcsv = sorted(glob(f"{BASE}/STN/*/stn_mcp/*/*/*MCP*.fcsv"))
    af_fcsv = sorted(glob(f"{BASE}/AFIDs/*/afids_mcp/*/*/*MCP*.fcsv"))
    df_stn = pd.concat([afidutils.fcsvtodf(s)[0] for s in stn_fcsv], axis=0)
    df_stn.columns = ["RSTNx", "LSTNx", "RSTNy", "LSTNy", "RSTNz", "LSTNz"]
    df_stn["LSTNx"] *= -1
    r = df_stn[["RSTNx", "RSTNy", "RSTNz"]].rename(columns=lambda c: c[1:])
    l = df_stn[["LSTNx", "LSTNy", "LSTNz"]].rename(columns=lambda c: c[1:])
    df_stn_proc = pd.concat([r, l], axis=0)
    df_af = pd.concat([afidutils.fcsvtodf(f)[0] for f in af_fcsv], axis=0)
    return afidutils.flip_and_concatenate_hemispheres(df_af), df_stn_proc


def cv_median_ed(X, y, Xex, groups):
    eds = []
    for tr, te in GroupKFold(4).split(X, groups=groups):
        sc, pca = StandardScaler(), PCA(0.99)
        pca.fit(sc.fit_transform(np.vstack([X[tr], Xex])))
        Xtp, Xep = pca.transform(sc.transform(X[tr])), pca.transform(sc.transform(X[te]))
        gi = [groups[i] for i in tr]; inner = list(GroupKFold(3).split(Xtp, groups=gi))
        pred = np.column_stack([
            Ridge(alpha=RidgeCV(alphas=ALPHAS, cv=inner).fit(Xtp, y[tr][:, j]).alpha_)
            .fit(Xtp, y[tr][:, j]).predict(Xep) for j in range(3)])
        eds += list(np.sqrt(((y[te] - pred) ** 2).sum(1)))
    return float(np.median(eds))


def cols_for(lms, all_cols):
    return [c for c in all_cols if any(c.startswith(lm) and c[len(lm):len(lm)+1] in "xyz" for lm in lms)]


def main():
    df_af, df_stn = load()
    train = pd.read_csv(f"{BASE}/train_data.csv")["subjid"].tolist()
    test_all = pd.read_csv(f"{BASE}/test_data.csv")["subjid"].tolist()
    other = df_af.drop(index=[i for i in test_all if i in df_af.index]).drop(index=train)
    Xtr = df_af.loc[train]; y = df_stn.loc[train].to_numpy()
    groups = afidutils.create_list(len(train))
    all_cols = list(df_af.columns)

    full = cv_median_ed(Xtr[all_cols].to_numpy(), y, other[all_cols].to_numpy(), groups)
    print(f"Full {len(LANDMARKS)}-landmark model: median CV ED = {full:.3f} mm\n")

    selected, table = [], []
    remaining = LANDMARKS.copy()
    while remaining:
        scored = []
        for lm in remaining:
            cols = cols_for(selected + [lm], all_cols)
            scored.append((cv_median_ed(Xtr[cols].to_numpy(), y, other[cols].to_numpy(), groups), lm))
        scored.sort()
        best_ed, best_lm = scored[0]
        selected.append(best_lm); remaining.remove(best_lm)
        table.append({"step": len(selected), "added": best_lm,
                      "subset": "+".join(selected), "median_CV_ED": round(best_ed, 3)})
        print(f"  step {len(selected):2d}: +{best_lm:5s} -> {best_ed:.3f} mm")

    out = pd.DataFrame(table)
    out.to_csv(REPO / "data/tableS3_subset_selection_corrected.csv", index=False)
    print(f"\nFull model: {full:.3f} mm | best subset size by ED: "
          f"{out.loc[out.median_CV_ED.idxmin(),'step']} landmarks "
          f"({out.median_CV_ED.min():.3f} mm)")
    print("Saved -> data/tableS3_subset_selection_corrected.csv")


if __name__ == "__main__":
    main()
