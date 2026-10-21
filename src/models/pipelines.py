"""Preprocessing and candidate models, shared by 04_modeling and 05_model_evaluation.

The model ladder (design 6): ridge (L1), hist_gbm (G1, default settings) and hist_gbm_tuned
(G2: tuned in 04_modeling, plus the out-of-fold kNN price feature of design 7.1).

05_model_evaluation refits variants on the training partition (ablations, seed stability, a
candidate without a saved artifact). Hyperparameters are never chosen there: a variant of a
saved model is made with `with_features`, which keeps the saved model's settings (including
its kNN k) and swaps only the column lists.
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

from src.models.knn_features import COORDS, KNNPriceFeature

# Simplest first. The selection rule (design 13.2) picks the simplest model whose paired
# MAE difference from the best one has a CI that includes 0.
COMPLEXITY_ORDER = ["B0_global_median", "B1_group_median", "ridge", "hist_gbm", "hist_gbm_tuned"]

# G2 settings used when no tuned parameters are available (synthetic demo, refit fallback)
DEFAULT_KNN_K = 15
DEFAULT_G2_PARAMS = {"learning_rate": 0.05, "max_leaf_nodes": 31, "min_samples_leaf": 20,
                     "l2_regularization": 0.1, "max_features": 1.0}


def make_preprocessor(numeric, numeric_indicator, categorical,
                      knn_k: int | None = None) -> ColumnTransformer:
    """Column-wise preprocessing. With knn_k, adds the out-of-fold kNN price feature
    (design 7.1), computed from latitude/longitude; only if both are among the features."""
    numeric_pipe = Pipeline([("impute", SimpleImputer(strategy="median")),
                             ("scale", StandardScaler())])
    numeric_indicator_pipe = Pipeline([("impute", SimpleImputer(strategy="median",
                                                                add_indicator=True)),
                                       ("scale", StandardScaler())])
    categorical_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=10)),
    ])
    steps = [(name, pipe, list(cols)) for name, pipe, cols in [
        ("num", numeric_pipe, numeric),
        ("num_ind", numeric_indicator_pipe, numeric_indicator),
        ("cat", categorical_pipe, categorical),
    ] if cols]
    if knn_k and set(COORDS) <= set(numeric) | set(numeric_indicator):
        steps.append(("knn", KNNPriceFeature(k=int(knn_k)), COORDS))
    return ColumnTransformer(steps)


def knn_k_of(pipe: Pipeline) -> int | None:
    """The kNN k a pipeline uses, or None if it has no kNN price feature."""
    for name, transformer, _ in pipe.named_steps["prep"].transformers:
        if name == "knn":
            return int(transformer.k)
    return None


def make_model(name: str, preprocessor, seed: int = 42, params: dict | None = None) -> Pipeline:
    """The candidates of the model ladder. `params` only applies to hist_gbm_tuned (G2); the
    preprocessor for G2 should include the kNN feature (make_preprocessor(..., knn_k=k))."""
    g2 = {**DEFAULT_G2_PARAMS, **(params or {})}
    estimators = {
        "ridge": lambda: Ridge(alpha=1.0),
        "hist_gbm": lambda: HistGradientBoostingRegressor(random_state=seed),
        "hist_gbm_tuned": lambda: HistGradientBoostingRegressor(
            random_state=seed, max_iter=1000, early_stopping=True, validation_fraction=0.1,
            n_iter_no_change=20, **g2),
    }
    if name not in estimators:
        raise KeyError(f"unknown model {name!r}; known: {sorted(estimators)}")
    return Pipeline([("prep", clone(preprocessor)), ("model", estimators[name]())])


def with_features(pipe: Pipeline, numeric, numeric_indicator, categorical,
                  seed: int | None = None, knn="keep") -> Pipeline:
    """Unfitted copy of `pipe` with new column lists (and optionally a new seed).

    knn="keep" keeps the pipeline's kNN feature (dropped automatically if latitude/longitude
    are removed); None removes it; an int sets k.
    """
    k = knn_k_of(pipe) if knn == "keep" else knn
    new = clone(pipe).set_params(prep=make_preprocessor(numeric, numeric_indicator, categorical,
                                                        knn_k=k))
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
