"""Budgeted, grouped and recorded hyperparameter search for G2 (design 8).

The design suggests Optuna's TPE sampler. To avoid a new dependency this uses a seeded
random search over the same kind of space, which is a well-established alternative for small
budgets (Bergstra & Bengio, 2012). The design's other requirements are kept:

* fixed trial budget, fixed seed: the search is reproducible;
* grouped CV inside the training partition, on the same folds as every other rung;
* the objective is the reported metric, MAE in currency per night (not log-scale error);
* early stopping inside each fit instead of tuning the number of trees;
* the kNN price feature's k is tuned jointly, and recomputed out of fold inside each fold;
* every trial is recorded; the full history belongs in the appendix.

Selecting the best of N configurations by CV score makes that score optimistic. The honest
number comes from the untouched test partition in 05_model_evaluation.
"""
from __future__ import annotations

import math
import time

import numpy as np
import pandas as pd
from sklearn.base import clone

# name -> (low, high, scale, type)
SEARCH_SPACE = {
    "learning_rate": (0.01, 0.2, "log", float),
    "max_leaf_nodes": (15, 255, "log", int),
    "min_samples_leaf": (10, 200, "log", int),
    "l2_regularization": (1e-3, 10.0, "log", float),
    "max_features": (0.5, 1.0, "linear", float),
    "knn_k": (5, 50, "linear", int),
}


def sample_params(rng: np.random.Generator, space: dict = SEARCH_SPACE) -> dict:
    out = {}
    for name, (lo, hi, scale, kind) in space.items():
        if scale == "log":
            v = math.exp(rng.uniform(math.log(lo), math.log(hi)))
        else:
            v = rng.uniform(lo, hi)
        out[name] = int(round(v)) if kind is int else float(v)
    return out


def random_search(build_pipe, X: pd.DataFrame, y_price: pd.Series, folds, to_model, from_model,
                  n_trials: int = 30, seed: int = 0, space: dict = SEARCH_SPACE,
                  verbose: bool = True) -> pd.DataFrame:
    """Evaluate `n_trials` random configurations on the given folds.

    build_pipe(params) -> unfitted Pipeline; to_model / from_model convert between price and
    the model's target scale (log1p / expm1). Returns one row per trial: the parameters, the
    MAE per fold, mean, std and seconds. The best trial is the row with the lowest mean MAE.
    """
    rng = np.random.default_rng(seed)
    y_price = pd.Series(np.asarray(y_price, float), index=X.index)
    rows = []
    for t in range(n_trials):
        params = sample_params(rng, space)
        t0 = time.perf_counter()
        maes = []
        for tr, va in folds:
            pipe = clone(build_pipe(params))
            pipe.fit(X.iloc[tr], to_model(y_price.iloc[tr]))
            pred = from_model(pipe.predict(X.iloc[va]))
            maes.append(float(np.mean(np.abs(y_price.iloc[va].to_numpy() - pred))))
        rows.append({"trial": t, **params, **{f"MAE fold {i + 1}": m for i, m in enumerate(maes)},
                     "MAE mean": float(np.mean(maes)), "MAE std": float(np.std(maes, ddof=1)),
                     "seconds": time.perf_counter() - t0})
        if verbose:
            best = min(r["MAE mean"] for r in rows)
            print(f"trial {t + 1:3d}/{n_trials}: MAE {rows[-1]['MAE mean']:8.2f}  (best {best:.2f})")
    return pd.DataFrame(rows).set_index("trial")


def best_params(trials: pd.DataFrame, space: dict = SEARCH_SPACE) -> dict:
    row = trials.loc[trials["MAE mean"].idxmin()]
    return {k: (int(row[k]) if space[k][3] is int else float(row[k])) for k in space}


def rough_importance(trials: pd.DataFrame, space: dict = SEARCH_SPACE) -> pd.DataFrame:
    """Spearman correlation of each parameter (log scale where searched on log) with the trial
    MAE: a rough guide to which settings mattered, not a formal importance measure."""
    out = {}
    for name, (_, _, scale, _) in space.items():
        x = np.log(trials[name]) if scale == "log" else trials[name]
        out[name] = pd.Series(x).corr(trials["MAE mean"], method="spearman")
    t = pd.DataFrame({"spearman_rho_with_MAE": out})
    t["abs"] = t["spearman_rho_with_MAE"].abs()
    return t.sort_values("abs", ascending=False).drop(columns="abs")
