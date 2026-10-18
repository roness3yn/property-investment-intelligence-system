"""Conformalized quantile regression (CQR), design chapter 9.

Notebook 03 (blueprint section 11) fits the interval model: quantile models on the training
partition, conformity scores on the calibration partition (one row per listing). It saves
the result with `save_cqr`, and notebook 04 loads it with `load_cqr` to evaluate coverage,
width and interval score on the test partition.

Everything works in log space (log1p), so bounds stay positive and Q acts multiplicatively.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.base import clone


def fit_cqr(lo_model, hi_model, X_tr, y_tr, X_cal, y_cal, alpha: float = 0.2) -> dict:
    """Fit the two quantile models on train, compute Q on calibration (design 9.1).

    lo_model / hi_model: unfitted estimators already configured for the alpha/2 and
    1 - alpha/2 quantiles (e.g. HistGradientBoostingRegressor(loss="quantile", quantile=0.1)).
    X_cal / y_cal must hold one row per listing (exchangeability).
    """
    z_tr, z_cal = np.log1p(np.asarray(y_tr)), np.log1p(np.asarray(y_cal))
    lo = clone(lo_model).fit(X_tr, z_tr)
    hi = clone(hi_model).fit(X_tr, z_tr)
    scores = np.maximum(lo.predict(X_cal) - z_cal, z_cal - hi.predict(X_cal))
    n = len(scores)
    level = min(1.0, np.ceil((1 - alpha) * (n + 1)) / n)
    q = float(np.quantile(scores, level, method="higher"))
    return {"lo": lo, "hi": hi, "q": q, "alpha": alpha, "n_calibration": n}


def predict_interval(cqr: dict, X) -> tuple[np.ndarray, np.ndarray]:
    """[lower, upper] in currency per night; bounds sorted in case the quantiles cross."""
    lo = np.expm1(cqr["lo"].predict(X) - cqr["q"])
    hi = np.expm1(cqr["hi"].predict(X) + cqr["q"])
    return np.minimum(lo, hi), np.maximum(lo, hi)


def save_cqr(cqr: dict, path: Path | str) -> None:
    joblib.dump(cqr, path)


def load_cqr(path: Path | str) -> dict:
    cqr = joblib.load(path)
    assert {"lo", "hi", "q", "alpha"} <= set(cqr), f"{path} is not a CQR artifact"
    return cqr
