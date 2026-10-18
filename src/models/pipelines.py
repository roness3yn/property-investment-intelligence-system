"""Preprocessing and candidate models (notebook 03 cells 16-17 as functions).

Notebook 04 needs these to refit variants on the training partition (ablations, seed
stability, a candidate that 03 did not save). Hyperparameters are never chosen here: a
variant of a saved model is made with `with_features`, which keeps the saved model's
settings and swaps only the column lists.
"""
from __future__ import annotations

import numpy as np
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Simplest first. The selection rule (design 13.2) picks the simplest model whose paired
# MAE difference from the best one has a CI that includes 0.
COMPLEXITY_ORDER = ["B0_global_median", "B1_group_median", "ridge", "hist_gbm"]


def make_preprocessor(numeric, numeric_indicator, categorical) -> ColumnTransformer:
    numeric_pipe = Pipeline([("impute", SimpleImputer(strategy="median")),
                             ("scale", StandardScaler())])
    numeric_indicator_pipe = Pipeline([("impute", SimpleImputer(strategy="median",
                                                                add_indicator=True)),
                                       ("scale", StandardScaler())])
    categorical_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=10)),
    ])
    return ColumnTransformer([
        (name, pipe, list(cols)) for name, pipe, cols in [
            ("num", numeric_pipe, numeric),
            ("num_ind", numeric_indicator_pipe, numeric_indicator),
            ("cat", categorical_pipe, categorical),
        ] if cols
    ])


def make_model(name: str, preprocessor, seed: int = 42) -> Pipeline:
    """The candidates of notebook 03, with its default settings."""
    estimators = {
        "ridge": lambda: Ridge(alpha=1.0),
        "hist_gbm": lambda: HistGradientBoostingRegressor(random_state=seed),
    }
    if name not in estimators:
        raise KeyError(f"unknown model {name!r}; known: {sorted(estimators)}")
    return Pipeline([("prep", clone(preprocessor)), ("model", estimators[name]())])


def with_features(pipe: Pipeline, numeric, numeric_indicator, categorical,
                  seed: int | None = None) -> Pipeline:
    """Unfitted copy of `pipe` with new column lists (and optionally a new seed)."""
    new = clone(pipe).set_params(prep=make_preprocessor(numeric, numeric_indicator, categorical))
    if seed is not None and "random_state" in new.named_steps["model"].get_params():
        new.set_params(model__random_state=seed)
    return new


def has_seed(pipe: Pipeline) -> bool:
    """True if the estimator's result depends on its random_state.

    Linear models expose random_state only for stochastic solvers that Ridge(alpha=1.0) does
    not use, so they count as deterministic.
    """
    est = pipe.named_steps["model"]
    return ("random_state" in est.get_params()
            and not type(est).__module__.startswith("sklearn.linear_model"))


class PricePredictor:
    """Wraps a fitted pipeline so .predict returns currency per night (undoes log1p)."""

    def __init__(self, pipe, log_target: bool = True):
        self.pipe, self.log_target = pipe, log_target

    def fit(self, X, y):
        self.pipe.fit(X, np.log1p(y) if self.log_target else y)
        return self

    def predict(self, X):
        p = self.pipe.predict(X)
        return np.expm1(p) if self.log_target else p


def smearing_factor(pipe, X, y, groups, n_splits: int = 5) -> float:
    """Duan (1983) smearing factor mean(exp(e_i)) from out-of-fold log residuals on train.

    Multiply the plain back-transformed prediction by this factor to estimate the conditional
    mean, which is what RMSE rewards (design 7.3). Folds are grouped by listing.
    """
    from sklearn.model_selection import GroupKFold

    z = np.log1p(np.asarray(y, float))
    oof = np.empty_like(z)
    for tr, va in GroupKFold(n_splits=n_splits).split(X, z, groups):
        oof[va] = clone(pipe).fit(X.iloc[tr], z[tr]).predict(X.iloc[va])
    return float(np.mean(np.exp(z - oof)))
