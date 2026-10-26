"""Why the AC/PC leave-one-out results are byte-identical: it is not a bug.

AC-PC (MCP-centred) space is *defined* by AC, PC and the midsagittal plane. Once the
data are transformed into that frame, several landmark coordinates are fixed by
construction and carry no subject-to-subject variance:

    ACx = ACz = PCx = PCz = 0      (AC and PC define the y-axis)
    PMJx = 0                        (PMJ helps define the midsagittal plane)

and, critically, AC and PC are placed symmetrically about the mid-commissural point,
so their single informative coordinate is one number seen twice with opposite signs:

    ACy = -PCy  exactly

AC and PC therefore share exactly ONE degree of freedom between them: half the AC-PC
distance. Dropping AC and dropping PC leave design matrices that differ only by the
sign of a single column, which StandardScaler + PCA remove -- so the two ablations
must produce identical predictions. This script verifies each claim.

Run (from any directory):  python notebooks/verify_acpc_degeneracy.py
"""
import numpy as np
import pandas as pd
from pathlib import Path

RAW = Path(__file__).resolve().parents[1] / "data/coordinate/AFIDs/df_afids_mcp_raw.csv"
LM_IDX = {"AC": 1, "PC": 2, "ICS": 3, "PMJ": 4, "SIPF": 5, "SLMS": 6,
          "ILMS": 8, "IMS": 11, "MB": 12, "LVAC": 15, "LVPC": 17}

raw = pd.read_csv(RAW, index_col=0)
print(f"{len(raw)} subjects x {len(LM_IDX)} landmarks, AC-PC (MCP-centred) space\n")

# 1 -- which coordinates are constant by construction?
print("1. Coordinates with zero variance (fixed by the AC-PC frame):")
const = []
for lm, i in LM_IDX.items():
    for a in "xyz":
        col = raw[f"{a}_{i}"]
        if col.std() < 1e-9:
            const.append(f"{lm}{a}")
            print(f"     {lm}{a:<2s} = {col.iloc[0]:.6f} for every subject")
print(f"   -> {len(const)} of {3 * len(LM_IDX)} coordinates carry no information\n")

# 2 -- the AC/PC identity
acy, pcy = raw["y_1"].values, raw["y_2"].values
print("2. AC and PC are the same number twice:")
print(f"     ACy == -PCy exactly : {np.array_equal(acy, -pcy)}")
print(f"     max |ACy + PCy|     : {np.abs(acy + pcy).max():.2e} mm")
print(f"     corr(ACy, PCy)      : {np.corrcoef(acy, pcy)[0, 1]:+.6f}")
print(f"     ACy = half the AC-PC distance; mean AC-PC = {2 * acy.mean():.2f} mm\n")

# 3 -- the consequence for leave-one-out
print("3. Consequence: dropping AC vs dropping PC gives the same PCA scores.")
feat = [f"{a}_{i}" for i in LM_IDX.values() for a in "xyz"]
X = raw[[c for c in feat if raw[c].std() >= 1e-9]]


def standardized(drop_idx):
    """Design matrix with one landmark removed, z-scored (what the pipeline sees)."""
    keep = [c for c in X.columns if not c.endswith(f"_{drop_idx}")]
    Z = X[keep].values
    return (Z - Z.mean(0)) / Z.std(0)


no_ac, no_pc = standardized(1), standardized(2)
print(f"     shapes                    : {no_ac.shape} and {no_pc.shape}")
print(f"     equal up to a column sign : {np.allclose(np.abs(no_ac), np.abs(no_pc))}")

# The two designs are X and XD with D = diag(1,..,-1,..,1) flipping the AC/PC column.
# Then (XD)'(XD) = D X'X D, so the eigenvalues match and the eigenvectors are D v --
# giving scores (XD)(Dv) = Xv. The Gram differs, but everything the model sees does not.
from sklearn.decomposition import PCA

s_ac = PCA(n_components=0.99, svd_solver="full").fit_transform(no_ac)
s_pc = PCA(n_components=0.99, svd_solver="full").fit_transform(no_pc)
print(f"     same number of components : {s_ac.shape[1] == s_pc.shape[1]} "
      f"({s_ac.shape[1]} PCs)")
# components are identical up to an arbitrary per-component sign (an SVD convention)
same = np.allclose(np.abs(s_ac), np.abs(s_pc), atol=1e-8)
print(f"     identical PCA scores      : {same}  <- Ridge therefore sees the same input")
print("\n   The AC/PC leave-one-out result is a property of the coordinate frame,")
print("   not a duplication bug. AC and PC cannot be ablated independently:")
print("   removing either one leaves the other carrying the same single number.")
