"""Out-of-fold k-nearest-neighbour price feature (design 7.1).

"What do the k nearest training listings charge?" is one of the strongest price signals, and
one of the easiest to leak: computed naively, every training listing would see its own price.
This transformer is built to be leak-free inside a scikit-learn Pipeline:

* fit_transform(X, y)  (training data, called by Pipeline.fit): each row's value is the median
  target of its k nearest neighbours computed OUT OF FOLD: the row's own fold (grouped by
  location, so a listing's other snapshots share its fold) never contributes.
* transform(X)  (calibration, test, new listings): neighbours come from the full training set;
  a neighbour at distance zero (the listing itself, if it was in training) is skipped.

Because it lives in the pipeline, cross-validation in 04_modeling refits it on each fold's
training rows only, and the saved model carries its training neighbours with it, so
05_model_evaluation and the prediction tool compute it exactly the same way.

The value is the median of the model-scale target (log1p price for our models): the median of
log prices is the log of the median price, so it is the neighbours' median price on log scale.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import BallTree

COORDS = ["latitude", "longitude"]


def _coords(X, fill=None) -> np.ndarray:
    a = (X[COORDS].to_numpy(float) if isinstance(X, pd.DataFrame) else np.asarray(X, float)).copy()
    if fill is not None:
        for j in range(2):
            a[np.isnan(a[:, j]), j] = fill[j]
    return a


class KNNPriceFeature(TransformerMixin, BaseEstimator):
    """Median target of the k nearest training listings (haversine), out-of-fold on training."""

    def __init__(self, k: int = 15, n_splits: int = 5):
        self.k = k
        self.n_splits = n_splits

    def fit(self, X, y):
        if y is None:
            raise ValueError("KNNPriceFeature needs the target: use it inside a Pipeline that is "
                             "fitted with y")
        raw = _coords(X)
        self.fill_ = np.nanmedian(raw, axis=0)
        self.coords_ = np.radians(_coords(X, self.fill_))
        self.y_ = np.asarray(y, float)
        self.tree_ = BallTree(self.coords_, metric="haversine")
        return self

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y)
        n = len(self.y_)
        out = np.empty(n)
        # Group rows by location so a listing's other snapshots never act as its neighbours
        groups = pd.factorize(pd.Series(map(tuple, np.round(self.coords_, 9))))[0]
        n_splits = min(self.n_splits, len(np.unique(groups)))
        for fit_idx, oof_idx in GroupKFold(n_splits=n_splits).split(self.coords_, groups=groups):
            tree = BallTree(self.coords_[fit_idx], metric="haversine")
            k = min(self.k, len(fit_idx))
            _, nn = tree.query(self.coords_[oof_idx], k=k)
            out[oof_idx] = np.median(self.y_[fit_idx][nn], axis=1)
        return out.reshape(-1, 1)

    def transform(self, X):
        pts = np.radians(_coords(X, self.fill_))
        k = min(self.k + 1, len(self.y_))
        dist, nn = self.tree_.query(pts, k=k)
        out = np.empty(len(pts))
        for i in range(len(pts)):
            keep = nn[i][dist[i] > 1e-12][: self.k]       # skip the listing itself if present
            out[i] = np.median(self.y_[keep]) if len(keep) else np.median(self.y_)
        return out.reshape(-1, 1)

    def get_feature_names_out(self, input_features=None):
        return np.array(["knn_price"])
