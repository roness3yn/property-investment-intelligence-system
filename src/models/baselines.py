"""Baselines B0 and B1 (design chapter 6), shared by notebooks 03 and 04.

B1 is the README's official baseline: "training-set median by neighbourhood and room type,
global-median fallback". `MedianGroupBaseline` implements exactly that (it is the class
from notebook 03). `HierarchicalMedian` is the design's refined version (section 6.1:
minimum group size, room-type fallback); switching B1 to it is a team decision that must
be made, and pre-registered, before the test set is scored.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

BASELINE_GROUPS = ["neighbourhood_cleansed", "room_type"]


class GlobalMedianBaseline:
    """B0: predicts the training median for every listing."""

    def fit(self, X, y):
        self.median_ = float(np.median(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.median_)


class MedianGroupBaseline:
    """B1 (README): median by group, global-median fallback for unseen groups."""

    def __init__(self, group_cols=tuple(BASELINE_GROUPS)):
        self.group_cols = list(group_cols)

    def fit(self, X, y):
        data = X[self.group_cols].assign(_y=np.asarray(y))
        self.group_medians_ = data.groupby(self.group_cols)["_y"].median()
        self.global_median_ = float(np.median(y))
        return self

    def predict(self, X):
        keys = pd.MultiIndex.from_frame(X[self.group_cols])
        preds = self.group_medians_.reindex(keys).to_numpy()
        return np.where(np.isnan(preds), self.global_median_, preds)

    def is_fallback(self, X):
        keys = pd.MultiIndex.from_frame(X[self.group_cols])
        return np.isnan(self.group_medians_.reindex(keys).to_numpy())


class HierarchicalMedian:
    """Design 6.1: median by (neighbourhood, room_type); backs off to room_type, then global."""

    LEVELS = (("neighbourhood_cleansed", "room_type"), ("room_type",))

    def __init__(self, levels=LEVELS, min_count=10):
        self.levels, self.min_count = levels, min_count

    def fit(self, X, y):
        d = X.assign(_y=np.asarray(y))
        self.tables_ = []
        for cols in self.levels:
            g = d.groupby(list(cols), observed=True)["_y"].agg(["median", "size"])
            keep = g["size"] >= self.min_count
            self.tables_.append((list(cols), g.loc[keep, "median"]))
        self.global_ = float(np.median(y))
        return self

    def predict(self, X):
        pred = pd.Series(np.nan, index=X.index)
        for cols, table in self.tables_:
            keys = pd.MultiIndex.from_frame(X[cols]) if len(cols) > 1 else X[cols[0]]
            vals = pd.Series(table.reindex(keys).to_numpy(), index=X.index)
            pred = pred.fillna(vals)
        return pred.fillna(self.global_).to_numpy()
