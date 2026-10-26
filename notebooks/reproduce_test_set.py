"""Held-out test-set evaluation of the deployed STN model.

Regenerates the headline test-set metrics reported in the manuscript: the
deployed model (resources/models/afids_stn_model_all_data.pkl, trained on the
SNSX training subjects only) is applied to every held-out test scan and scored
against the STN ground truth. The test set is 20 people scanned twice
(10 SNSXPD people at 3 T and 1.5 T + gadolinium; 10 3T-7T people at 3 T and
7 T) = 40 scans = 80 STNs. A person-level cluster bootstrap gives the 95% CI
of the median ED (both scans of one person form one cluster).

For reference, the pipeline is also refit from the training subjects with
the current code (afids_pipeline.AfidsTargetPredictor, groups = true subject
ids). The deployed model was fit with an earlier version of the code
(scikit-learn 1.6.1, different ridge alpha grid), so the refit is close to,
but not identical with, the deployed model; the manuscript numbers come from
the deployed model.

Run (from any directory):  python notebooks/reproduce_test_set.py
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from glob import glob
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import afidutils
from afids_pipeline import AfidsTargetPredictor

REPO = Path(__file__).resolve().parents[1]
BASE = REPO / "data/coordinate"
MODEL = REPO / "resources/models/afids_stn_model_all_data.pkl"
DATASETS = ["SNSXPD", "SNSXPD1.5T", "3T7T_3T", "3T7T_7T"]
SEED = 42


def read_xyz(path):
    """Rows of a Slicer .fcsv as an (n, 3) array; for STN files row 0 = right,
    row 1 = left (left x is mirrored to match the model's output convention)."""
    return pd.read_csv(path, comment="#", header=None).iloc[:, 1:4].to_numpy(float)


def stn_truth(dataset, subject):
    """Ground-truth STN for a test scan. The 1.5 T scans (sub-Rxxx) share the
    3 T label of the same person (sub-Pxxx); the 3T-7T 3 T scans (sub-NN) share
    the 7 T label (sub-RNN)."""
    person = (subject.replace("sub-R", "sub-P") if dataset == "SNSXPD1.5T"
              else subject.replace("sub-", "sub-R") if dataset == "3T7T_3T" else subject)
    files = glob(f"{BASE}/STN/*/stn_mcp/{person}/anat/*groundtruth*.fcsv")
    assert len(files) == 1, files
    t = read_xyz(files[0]); t[1, 0] *= -1
    return person, t


def load_training():
    stn_fcsv = sorted(glob(f"{BASE}/STN/*/stn_mcp/*/*/*MCP*.fcsv"))
    af_fcsv = sorted(glob(f"{BASE}/AFIDs/*/afids_mcp/*/*/*MCP*.fcsv"))
    df_stn = pd.concat([afidutils.fcsvtodf(s)[0] for s in stn_fcsv], axis=0)
    df_stn.columns = ["RSTNx", "LSTNx", "RSTNy", "LSTNy", "RSTNz", "LSTNz"]
    df_stn["LSTNx"] *= -1
    r = df_stn[["RSTNx", "RSTNy", "RSTNz"]].rename(columns=lambda c: c[1:])
    l = df_stn[["LSTNx", "LSTNy", "LSTNz"]].rename(columns=lambda c: c[1:])
    df_stn_proc = pd.concat([r, l], axis=0)
    df_af = pd.concat([afidutils.fcsvtodf(f)[0] for f in af_fcsv], axis=0)
    df_af_proc = afidutils.flip_and_concatenate_hemispheres(df_af)
    return df_af_proc, df_stn_proc


def main():
    test = pd.read_csv(BASE / "test_data.csv")["subjid"].tolist()
    rows = []
    for ds in DATASETS:
        for f in sorted(glob(f"{BASE}/AFIDs/{ds}/afids_mcp/*/anat/*groundtruth*.fcsv")):
            subject = Path(f).parts[-3]
            if subject not in test:
                continue
            person, t = stn_truth(ds, subject)
            p = afidutils.predict_stn_coordinates(f, str(MODEL))
            for h, hemi in enumerate("RL"):
                rows.append(dict(dataset=ds, scan=subject, person=person, hemisphere=hemi,
                                 ED=float(np.linalg.norm(p[h] - t[h])),
                                 **{f"abserr_{a}": float(abs(p[h, j] - t[h, j])) for j, a in enumerate("xyz")},
                                 **{f"pred_{a}": float(p[h, j]) for j, a in enumerate("xyz")}))
    df = pd.DataFrame(rows)
    assert len(df) == 80 and df.person.nunique() == 20, (len(df), df.person.nunique())

    print(f"Deployed model: {MODEL.relative_to(REPO)}")
    print(f"Test set: {df.person.nunique()} people, {df.scan.nunique()} scans, {len(df)} STNs\n")
    for name, d in [("All", df)] + list(df.groupby("dataset", sort=False)):
        print(f"  {name:11s} n={len(d):2d}  median ED {d.ED.median():.2f} mm "
              f"[IQR {d.ED.quantile(.25):.2f}-{d.ED.quantile(.75):.2f}]  <2 mm {100*(d.ED < 2).mean():.1f}%")
    for a in "xyz":
        print(f"  MAE {a}: median {df[f'abserr_{a}'].median():.2f}  mean {df[f'abserr_{a}'].mean():.2f} mm")

    by = {p: d.ED.to_numpy() for p, d in df.groupby("person")}
    people = list(by); rng = np.random.RandomState(SEED)
    boots = [np.median(np.concatenate([by[p] for p in rng.choice(people, len(people))])) for _ in range(10000)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    print(f"  median ED 95% CI (person-level bootstrap): {lo:.2f}-{hi:.2f} mm")

    # Consistency check: refit the production pipeline on the training subjects.
    df_af, df_stn = load_training()
    train = pd.read_csv(BASE / "train_data.csv")["subjid"].tolist()
    other = df_af.drop(index=[i for i in test if i in df_af.index]).drop(index=train)
    Xtr = df_af.loc[train]
    refit = AfidsTargetPredictor("STN").fit(Xtr.to_numpy(), df_stn.loc[train].to_numpy(),
                                            groups=Xtr.index.to_numpy(), X_extra=other.to_numpy())
    max_diff, ed_refit = 0.0, []
    for (ds, scan), d in df.groupby(["dataset", "scan"]):
        f = glob(f"{BASE}/AFIDs/{ds}/afids_mcp/{scan}/anat/*groundtruth*.fcsv")[0]
        x = afidutils.flip_and_concatenate_hemispheres(afidutils.fcsvtodf(f)[0].map(afidutils.make_zero))
        p = refit.predict(x.to_numpy())
        max_diff = max(max_diff, float(np.abs(p - d[["pred_x", "pred_y", "pred_z"]].to_numpy()).max()))
        ed_refit += list(np.linalg.norm(p - stn_truth(ds, scan)[1], axis=1))
    print(f"\nRefit with current code: median test ED {np.median(ed_refit):.2f} mm; "
          f"max |prediction difference| vs deployed model {max_diff:.2f} mm")


if __name__ == "__main__":
    main()
