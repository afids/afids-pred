import numpy as np
import pandas as pd
import pickle
import sys
import os

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.utils import check_random_state
from sklearn.model_selection._split import _RepeatedSplits

sys.path.insert(0, os.path.dirname(__file__))
import afidutils


class RepeatedGroupKFold(_RepeatedSplits):
    """Repeated Group K-Fold cross validator."""

    def __init__(self, *, n_splits=5, n_repeats=10, random_state=None):
        super().__init__(
            GroupKFold, n_repeats=n_repeats,
            random_state=random_state, n_splits=n_splits)

    def split(self, X, y=None, groups=None):
        n_repeats = self.n_repeats
        rng = check_random_state(self.random_state)
        for idx in range(n_repeats):
            cv = self.cv(random_state=rng, shuffle=True, **self.cvargs)
            for train_index, test_index in cv.split(X, y, groups):
                yield train_index, test_index


class AfidsTargetPredictor:
    """
    Trains and applies a StandardScaler -> PCA -> Ridge pipeline to predict
    3D brain target coordinates from AFID feature vectors.
    """

    def __init__(self, target_name, pca_variance=0.99):
        self.target_name = target_name
        self.pca_variance = pca_variance
        self.scaler = None
        self.pca = None
        self.ridge_x = None
        self.ridge_y = None
        self.ridge_z = None

    def fit(self, X_afids, y_target, groups=None, X_extra=None):
        """
        Fit the pipeline (StandardScaler -> PCA -> Ridge per axis).

        Parameters
        ----------
        X_afids : np.ndarray, shape (n_samples, n_features)
        y_target : np.ndarray, shape (n_samples, 3)
        groups : array-like, optional — group labels for hemisphere grouping
        X_extra : np.ndarray, optional — extra AFIDs data for more stable PCA
        """
        # Regularization search grid. A sane, log-spaced range is sufficient;
        # the previous (-100, 100) grid was pathologically wide and wasteful.
        alphas = np.logspace(-3, 6, 200)

        # Augment training data for PCA fitting if extra data provided
        X_for_pca = np.concatenate([X_afids, X_extra], axis=0) if X_extra is not None else X_afids

        self.scaler = StandardScaler()
        self.pca = PCA(n_components=self.pca_variance)

        X_scaled_aug = self.scaler.fit_transform(X_for_pca)
        self.pca.fit(X_scaled_aug)

        X_pca = self.pca.transform(self.scaler.transform(X_afids))

        # Inner CV for alpha selection
        if groups is not None:
            inner_cv = list(GroupKFold(n_splits=3).split(X_pca, groups=groups))
        else:
            inner_cv = 3

        ridge_models = []
        for i in range(3):
            rcv = RidgeCV(alphas=alphas,
                          scoring='neg_mean_squared_error', cv=inner_cv)
            rcv.fit(X_pca, y_target[:, i])
            ridge = Ridge(alpha=rcv.alpha_)
            ridge.fit(X_pca, y_target[:, i])
            ridge_models.append(ridge)

        self.ridge_x, self.ridge_y, self.ridge_z = ridge_models
        print(f"Fitted {self.target_name} predictor. PCA components: {self.pca.n_components_}")
        return self

    def predict(self, X_afids):
        """
        Predict target coordinates.

        Parameters
        ----------
        X_afids : np.ndarray, shape (n_samples, n_features)

        Returns
        -------
        y_pred : np.ndarray, shape (n_samples, 3)
        """
        X = self.pca.transform(self.scaler.transform(X_afids))
        return np.column_stack([
            self.ridge_x.predict(X),
            self.ridge_y.predict(X),
            self.ridge_z.predict(X),
        ])

    def evaluate(self, X_afids, y_target, groups=None, n_splits=4, n_repeats=10):
        """
        Evaluate pipeline using RepeatedGroupKFold CV.

        Returns a dict with mean and std of ED and per-axis MSE.
        """
        results = cross_validate_pipeline(
            X_afids, y_target, groups,
            n_splits=n_splits, n_repeats=n_repeats,
            pca_variance=self.pca_variance,
        )
        summary = {
            'ED_mean': results['ED'].mean(),
            'ED_std': results['ED'].std(),
            'mse_x_mean': results['mse_x'].mean(),
            'mse_y_mean': results['mse_y'].mean(),
            'mse_z_mean': results['mse_z'].mean(),
        }
        return summary

    def save(self, path):
        """Save predictor to a pickle file."""
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        print(f"Saved {self.target_name} predictor to {path}")

    @classmethod
    def load(cls, path):
        """Load predictor from a pickle file."""
        with open(path, 'rb') as f:
            obj = pickle.load(f)
        return obj


def train_from_fcsvs(
    afids_fcsv_list,
    target_fcsv_list,
    target_col_names,
    output_model_path,
    target_name='target',
    pca_variance=0.99,
    extra_afids_fcsv_list=None,
):
    """
    Load AFID and target fcsv files, apply hemisphere flipping, train and save an
    AfidsTargetPredictor.

    Parameters
    ----------
    afids_fcsv_list : list of str
        Paths to AFID fcsv files (one per subject).
    target_fcsv_list : list of str
        Paths to target fcsv files matching the AFID files.
    target_col_names : list of str
        Column names to assign to the loaded target fcsv (e.g. ['Rx','Lx','Ry','Ly','Rz','Lz']).
        Must have 6 entries corresponding to right-x, left-x, right-y, left-y, right-z, left-z.
    output_model_path : str
        Path to write the trained predictor pickle.
    target_name : str
        Label for the target (used for print messages).
    pca_variance : float
        Fraction of variance retained by PCA.
    extra_afids_fcsv_list : list of str, optional
        Additional AFID fcsv files (no paired targets) used to stabilise PCA fitting.

    Returns
    -------
    predictor : AfidsTargetPredictor
    """
    print(f"Loading {len(afids_fcsv_list)} AFID files...")
    df_afids = pd.concat([afidutils.fcsvtodf(f)[0] for f in afids_fcsv_list], axis=0)
    df_afids_proc = afidutils.flip_and_concatenate_hemispheres(df_afids)

    print(f"Loading {len(target_fcsv_list)} target files...")
    df_target_raw = pd.concat([afidutils.fcsvtodf(f)[0] for f in target_fcsv_list], axis=0)
    df_target_raw.columns = target_col_names

    rx, lx = target_col_names[0], target_col_names[1]
    ry, ly = target_col_names[2], target_col_names[3]
    rz, lz = target_col_names[4], target_col_names[5]

    # Mirror left hemisphere x-coordinate
    df_target_raw[lx] *= -1

    df_target_r = df_target_raw[[rx, ry, rz]].rename(columns=lambda c: c[1:])
    df_target_l = df_target_raw[[lx, ly, lz]].rename(columns=lambda c: c[1:])
    df_target_proc = pd.concat([df_target_r, df_target_l], axis=0)

    X = df_afids_proc.to_numpy()
    y = df_target_proc.to_numpy()

    groups = afidutils.create_list(X.shape[0] // 2)

    X_extra = None
    if extra_afids_fcsv_list:
        print(f"Loading {len(extra_afids_fcsv_list)} extra AFID files for PCA...")
        df_extra = pd.concat([afidutils.fcsvtodf(f)[0] for f in extra_afids_fcsv_list], axis=0)
        X_extra = afidutils.flip_and_concatenate_hemispheres(df_extra).to_numpy()

    predictor = AfidsTargetPredictor(target_name=target_name, pca_variance=pca_variance)
    predictor.fit(X, y, groups=groups, X_extra=X_extra)
    predictor.save(output_model_path)

    return predictor


def predict_target_coordinates(subject_afid_path, model_path):
    """
    Run a saved AfidsTargetPredictor on a single subject's AFID fcsv file.

    Parameters
    ----------
    subject_afid_path : str
        Path to subject's AFID fcsv file (ACPC + MCP centered space).
    model_path : str
        Path to a saved AfidsTargetPredictor pickle.

    Returns
    -------
    y_pred : np.ndarray, shape (2, 3)
        Predicted coordinates: row 0 = right hemisphere, row 1 = left hemisphere.
    """
    df = afidutils.fcsvtodf(subject_afid_path)[0]
    df = df.map(afidutils.make_zero)
    df_proc = afidutils.flip_and_concatenate_hemispheres(df)

    predictor = AfidsTargetPredictor.load(model_path)
    return predictor.predict(df_proc.to_numpy())


def cross_validate_pipeline(
    X_afids,
    y_target,
    groups,
    X_extra=None,
    n_splits=4,
    n_repeats=10,
    pca_variance=0.99,
    random_state=42,
):
    """
    Run repeated group K-fold CV and return per-fold metrics.

    Parameters
    ----------
    X_afids : np.ndarray, shape (n_samples, n_features)
    y_target : np.ndarray, shape (n_samples, 3)
    groups : array-like, shape (n_samples,)
    X_extra : np.ndarray, optional — extra AFIDs for PCA stabilisation
    n_splits : int
    n_repeats : int
    pca_variance : float
    random_state : int

    Returns
    -------
    results_df : pd.DataFrame with columns fold, mse_x, mse_y, mse_z, ED
    """
    alphas = np.logspace(-3, 6, 200)
    rkf = RepeatedGroupKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    rows = []

    for fold_idx, (train_idx, test_idx) in enumerate(rkf.split(X_afids, y_target, groups=groups)):
        X_train, X_test = X_afids[train_idx], X_afids[test_idx]
        y_train, y_test = y_target[train_idx], y_target[test_idx]

        # Augment with extra data for PCA fitting
        X_train_aug = np.concatenate([X_train, X_extra], axis=0) if X_extra is not None else X_train

        scaler = StandardScaler()
        pca = PCA(n_components=pca_variance)
        pca.fit(scaler.fit_transform(X_train_aug))

        X_train_pca = pca.transform(scaler.transform(X_train))
        X_test_pca = pca.transform(scaler.transform(X_test))

        groups_inner = [groups[i] for i in train_idx]
        inner_cv = list(GroupKFold(n_splits=3).split(X_train_pca, groups=groups_inner))

        ridge_models = []
        for i in range(3):
            rcv = RidgeCV(alphas=alphas,
                          scoring='neg_mean_squared_error', cv=inner_cv)
            rcv.fit(X_train_pca, y_train[:, i])
            ridge = Ridge(alpha=rcv.alpha_)
            ridge.fit(X_train_pca, y_train[:, i])
            ridge_models.append(ridge)

        y_pred = np.column_stack([r.predict(X_test_pca) for r in ridge_models])

        mse_x = ((y_test[:, 0] - y_pred[:, 0]) ** 2).mean()
        mse_y = ((y_test[:, 1] - y_pred[:, 1]) ** 2).mean()
        mse_z = ((y_test[:, 2] - y_pred[:, 2]) ** 2).mean()
        ed = np.sqrt(((y_test - y_pred) ** 2).sum(axis=1)).mean()

        rows.append({'fold': fold_idx + 1, 'mse_x': mse_x, 'mse_y': mse_y, 'mse_z': mse_z, 'ED': ed})

    return pd.DataFrame(rows)
