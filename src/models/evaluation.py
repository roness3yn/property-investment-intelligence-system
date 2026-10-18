"""Evaluation metrics and statistics for notebook 04 (design chapters 14-16).

* Point metrics (chapter 14): MAE (primary), RMSE, MedAE, typical relative error,
  hit rate within +/-20%, bias, R^2 on log price.
* Interval metrics: coverage, mean width, interval score (Gneiting & Raftery 2007).
* Cluster bootstrap (chapter 15): the resampling unit is the listing, not the row, and
  model comparisons are paired (same test listings for both models).
* Segment tables with reliability flags (chapter 16.1).

Every metric that is a transform of a per-row mean gets a bootstrap CI. MedAE and R^2 are
reported as point values only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-9


# --------------------------------------------------------------------------- per-row pieces
def per_row_point(y, yhat) -> dict:
    """Per-row quantities whose means define the bootstrap-able point metrics."""
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    err = yhat - y
    return {
        "abs_err": np.abs(err),
        "sq_err": err ** 2,
        "signed_err": err,
        "abs_log_ratio": np.abs(np.log(np.clip(yhat, EPS, None)) - np.log(np.clip(y, EPS, None))),
        "hit20": (np.abs(yhat / np.clip(y, EPS, None) - 1) <= 0.2).astype(float),
    }


def per_row_interval(y, lo, hi, alpha) -> dict:
    y, lo, hi = (np.asarray(a, float) for a in (y, lo, hi))
    width = hi - lo
    score = width + (2 / alpha) * (lo - y) * (y < lo) + (2 / alpha) * (y - hi) * (y > hi)
    return {"covered": ((y >= lo) & (y <= hi)).astype(float), "width": width,
            "interval_score": score}


# metric name -> (per-row key, transform of the mean)
POINT_METRICS = {
    "MAE": ("abs_err", lambda m: m),
    "RMSE": ("sq_err", np.sqrt),
    "bias": ("signed_err", lambda m: m),
    "typical_rel_err": ("abs_log_ratio", lambda m: np.exp(m) - 1),
    "hit_rate_20": ("hit20", lambda m: m),
}
INTERVAL_METRICS = {
    "coverage": ("covered", lambda m: m),
    "mean_width": ("width", lambda m: m),
    "interval_score": ("interval_score", lambda m: m),
}


def point_metrics(y, yhat) -> dict:
    """All chapter-14 point metrics as plain numbers."""
    r = per_row_point(y, yhat)
    out = {name: float(f(r[key].mean())) for name, (key, f) in POINT_METRICS.items()}
    out["MedAE"] = float(np.median(r["abs_err"]))
    ly, lp = np.log(np.asarray(y, float)), np.log(np.clip(np.asarray(yhat, float), EPS, None))
    out["R2_log"] = float(1 - ((ly - lp) ** 2).sum() / ((ly - ly.mean()) ** 2).sum())
    return out


def interval_metrics(y, lo, hi, alpha) -> dict:
    r = per_row_interval(y, lo, hi, alpha)
    return {name: float(f(r[key].mean())) for name, (key, f) in INTERVAL_METRICS.items()}


# --------------------------------------------------------------------------- cluster bootstrap
def _group_sums(values: np.ndarray, groups) -> tuple[np.ndarray, np.ndarray]:
    """Sum of `values` (rows x k) and row count per group (listing)."""
    codes, _ = pd.factorize(pd.Series(groups), sort=True)
    k = values.shape[1]
    sums = np.zeros((codes.max() + 1, k))
    for j in range(k):
        sums[:, j] = np.bincount(codes, weights=values[:, j])
    return sums, np.bincount(codes).astype(float)


def _resampled_means(sums, counts, n_boot, seed, chunk=200):
    """Bootstrap distribution of column means, resampling groups with replacement."""
    rng = np.random.default_rng(seed)
    g = len(counts)
    out = np.empty((n_boot, sums.shape[1]))
    for start in range(0, n_boot, chunk):
        b = min(chunk, n_boot - start)
        idx = rng.integers(0, g, size=(b, g))
        n = counts[idx].sum(1)
        for j in range(sums.shape[1]):
            out[start:start + b, j] = sums[idx, j].sum(1) / n
    return out


def bootstrap_ci(per_row: np.ndarray, groups, transform=lambda m: m, n_boot=2000, seed=0,
                 level=0.95):
    """Point estimate and CI for transform(mean(per_row)), resampling listings."""
    sums, counts = _group_sums(np.asarray(per_row, float)[:, None], groups)
    point = transform(sums[:, 0].sum() / counts.sum())
    boots = transform(_resampled_means(sums, counts, n_boot, seed)[:, 0])
    a = (1 - level) / 2 * 100
    lo, hi = np.percentile(boots, [a, 100 - a])
    return float(point), float(lo), float(hi)


def bootstrap_delta(per_row_a, per_row_b, groups, transform=lambda m: m, n_boot=2000, seed=0,
                    level=0.95):
    """Paired difference transform(mean a) - transform(mean b) with a cluster-bootstrap CI.

    With transform=identity and per-row absolute errors this is design 15's
    cluster_bootstrap_delta: MAE(a) - MAE(b). Negative means a has the lower error.
    """
    vals = np.column_stack([np.asarray(per_row_a, float), np.asarray(per_row_b, float)])
    sums, counts = _group_sums(vals, groups)
    point = transform(sums[:, 0].sum() / counts.sum()) - transform(sums[:, 1].sum() / counts.sum())
    m = _resampled_means(sums, counts, n_boot, seed)
    deltas = transform(m[:, 0]) - transform(m[:, 1])
    a = (1 - level) / 2 * 100
    lo, hi = np.percentile(deltas, [a, 100 - a])
    return float(point), float(lo), float(hi)


def metric_table(y, preds: dict, groups, n_boot=2000, seed=0, mean_preds: dict | None = None
                 ) -> pd.DataFrame:
    """Headline table: every point metric for every candidate, with 95% CIs where defined.

    mean_preds: optional {name: smearing-corrected predictions}. RMSE is computed from these
    (design 7.3: RMSE targets the conditional mean); all other metrics use `preds`.
    """
    mean_preds = mean_preds or {}
    rows = []
    for name, yhat in preds.items():
        r = per_row_point(y, yhat)
        r["sq_err"] = per_row_point(y, mean_preds.get(name, yhat))["sq_err"]
        row = {"approach": name}
        for metric, (key, f) in POINT_METRICS.items():
            p, lo, hi = bootstrap_ci(r[key], groups, f, n_boot, seed)
            row[metric], row[f"{metric} CI low"], row[f"{metric} CI high"] = p, lo, hi
        pm = point_metrics(y, yhat)
        row["MedAE"], row["R2_log"] = pm["MedAE"], pm["R2_log"]
        rows.append(row)
    return pd.DataFrame(rows).set_index("approach")


def paired_mae_deltas(y, preds: dict, reference: str, groups, n_boot=2000, seed=0) -> pd.DataFrame:
    """MAE(candidate) - MAE(reference) with paired cluster-bootstrap CIs, and % change."""
    ref_err = per_row_point(y, preds[reference])["abs_err"]
    ref_mae = ref_err.mean()
    rows = []
    for name, yhat in preds.items():
        if name == reference:
            continue
        err = per_row_point(y, yhat)["abs_err"]
        d, lo, hi = bootstrap_delta(err, ref_err, groups, n_boot=n_boot, seed=seed)
        rows.append({"approach": name, "MAE": err.mean(), f"MAE {reference}": ref_mae,
                     "delta": d, "delta CI low": lo, "delta CI high": hi,
                     "change (%)": 100 * d / ref_mae,
                     "CI excludes 0": (hi < 0) or (lo > 0)})
    return pd.DataFrame(rows).set_index("approach")


# --------------------------------------------------------------------------- segments
def segment_table(df: pd.DataFrame, by, min_n: int = 50) -> pd.DataFrame:
    """Design 16.1. Expects columns abs_err_model, abs_err_b1, signed_err_model and,
    if intervals exist, covered and width. Small segments are flagged, not hidden."""
    g = df.groupby(by, observed=True)
    t = pd.DataFrame({
        "n": g.size(),
        "listings": g["id"].nunique() if "id" in df.columns else g.size(),
        "MAE_model": g["abs_err_model"].mean(),
        "MAE_B1": g["abs_err_b1"].mean(),
        "bias": g["signed_err_model"].mean(),
    })
    if "covered" in df.columns:
        t["coverage"] = g["covered"].mean()
        t["width"] = g["width"].mean()
    t["gain_vs_B1_%"] = 100 * (1 - t["MAE_model"] / t["MAE_B1"])
    t["reliable"] = t["n"] >= min_n
    return t.sort_values("n", ascending=False)


def review_band(n_reviews: pd.Series) -> pd.Series:
    """Design 16.1 review-count bands: 0, 1-9, 10+."""
    return pd.cut(n_reviews.fillna(0), [-0.5, 0.5, 9.5, np.inf], labels=["0", "1-9", "10+"])


def host_type(listings_count: pd.Series) -> pd.Series:
    """Single-listing hosts vs multi-listing (professional) hosts."""
    return np.where(listings_count.fillna(1) > 1, "multi-listing host", "single-listing host")


# --------------------------------------------------------------------------- drift
def psi(expected, actual, bins: int = 10) -> float:
    """Population stability index of `actual` against `expected` (design 17.1).

    Bins are the deciles of `expected`. Rule of thumb: < 0.1 stable, 0.1-0.25 moderate
    shift, > 0.25 large shift.
    """
    e = pd.Series(expected, dtype=float).dropna().to_numpy()
    a = pd.Series(actual, dtype=float).dropna().to_numpy()
    if len(e) == 0 or len(a) == 0:
        return float("nan")
    edges = np.unique(np.quantile(e, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:                       # (almost) constant feature
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    pe = np.histogram(e, edges)[0] / len(e)
    pa = np.histogram(a, edges)[0] / len(a)
    pe, pa = np.clip(pe, 1e-4, None), np.clip(pa, 1e-4, None)
    return float(((pa - pe) * np.log(pa / pe)).sum())
