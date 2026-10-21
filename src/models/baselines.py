"""Baselines B0 and B1 (design chapter 6), shared by notebooks 04_modeling and 05_model_evaluation.

B1, the official baseline the learned models must beat by at least 10% (H1), is the design's
hierarchical median (section 6.1): the training median by neighbourhood x room type, used only
for groups with at least `B1_MIN_COUNT` training listings; smaller or unseen groups back off
to the room type's median, then to the global median. `make_b1()` builds it, so both
notebooks use exactly the same definition.

`MedianGroupBaseline` is the earlier, simpler version (neighbourhood x room type, global
fallback, no minimum group size). It is kept as the comparison row "B1_simple": the
difference between the two shows how much a careful baseline changes the bar.

All medians are of the nightly listing price, computed on the training partition only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

BASELINE_GROUPS = ["neighbourhood_cleansed", "room_type"]
B1_MIN_COUNT = 10                     # pre-registered with the hypotheses (notebook 05, §0.1)
B1_DESCRIPTION = (f"training median by neighbourhood x room type (groups with at least "
                  f"{B1_MIN_COUNT} training listings), else room-type median, else global median")
B1_SIMPLE_DESCRIPTION = "training median by neighbourhood x room type, else global median"


class GlobalMedianBaseline:
    """B0: predicts the training median for every listing."""

    def fit(self, X, y):
        self.median_ = float(np.median(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.median_)


class MedianGroupBaseline:
    """B1_simple: median by group, global-median fallback for unseen groups, no minimum size."""

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
    """B1 (design 6.1): median by (neighbourhood, room_type) for groups with >= min_count
    training listings; otherwise the room-type median (if that has >= min_count); otherwise
    the global median."""

    LEVELS = (("neighbourhood_cleansed", "room_type"), ("room_type",))

    def __init__(self, levels=LEVELS, min_count=B1_MIN_COUNT):
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

    def _level_values(self, X):
        """One array of looked-up medians per level (NaN where the level does not apply)."""
        out = []
        for cols, table in self.tables_:
            keys = pd.MultiIndex.from_frame(X[cols]) if len(cols) > 1 else X[cols[0]]
            out.append(pd.Series(table.reindex(keys).to_numpy(), index=X.index))
        return out

    def predict(self, X):
        pred = pd.Series(np.nan, index=X.index)
        for vals in self._level_values(X):
            pred = pred.fillna(vals)
        return pred.fillna(self.global_).to_numpy()

    def fallback_level(self, X) -> np.ndarray:
        """Which level answered each row: e.g. 'neighbourhood_cleansed x room_type',
        'room_type' or 'global'."""
        level = pd.Series("global", index=X.index, dtype=object)
        for (cols, _), vals in reversed(list(zip(self.tables_, self._level_values(X)))):
            level[vals.notna()] = " x ".join(cols)
        return level.to_numpy()

    def is_fallback(self, X) -> np.ndarray:
        """True where the finest level (neighbourhood x room type) did not answer."""
        return self.fallback_level(X) != " x ".join(self.tables_[0][0])


def make_b1() -> HierarchicalMedian:
    """The official B1 (design 6.1), identical in notebooks 04 and 05."""
    return HierarchicalMedian(min_count=B1_MIN_COUNT)


def make_b1_simple() -> MedianGroupBaseline:
    """The earlier B1, kept as a comparison row."""
    return MedianGroupBaseline(BASELINE_GROUPS)
