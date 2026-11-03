"""Tests for src/models/relative_metrics.py (no data or models needed).

Run from the repository root:   python -m pytest tests/test_relative_metrics.py -q
"""
import numpy as np
import pandas as pd

from src.models.relative_metrics import by_price_band, relative_metrics, summary_table


def test_known_values():
    y = np.array([100.0, 200.0, 400.0, 50.0])
    p = np.array([110.0, 150.0, 400.0, 100.0])
    r = relative_metrics(y, p)
    assert np.isclose(r["MAE"], (10 + 50 + 0 + 50) / 4)
    assert np.isclose(r["MAPE"], 100 * np.mean([0.10, 0.25, 0.0, 1.0]))
    assert np.isclose(r["MdAPE"], 100 * np.median([0.10, 0.25, 0.0, 1.0]))
    assert np.isclose(r["hit_rate_20"], 50.0)                       # 100->110 and 400->400
    expected = 100 * (np.exp(np.mean(np.abs(np.log(p / y)))) - 1)
    assert np.isclose(r["typical_rel_err"], expected)


def test_typical_relative_error_is_symmetric_but_mape_is_not():
    y = np.array([100.0])
    high, low = relative_metrics(y, [200.0]), relative_metrics(y, [50.0])
    assert np.isclose(high["typical_rel_err"], low["typical_rel_err"])   # x2 and /2 count the same
    assert high["MAPE"] == 100.0 and low["MAPE"] == 50.0                  # MAPE favours under-prediction


def test_perfect_prediction_and_bad_rows():
    r = relative_metrics([100, 200, np.nan, 0], [100, 200, 5, 5])        # NaN and zero prices ignored
    assert r["MAE"] == 0 and r["MAPE"] == 0 and r["hit_rate_20"] == 100 and r["typical_rel_err"] == 0


def test_tables():
    rng = np.random.default_rng(0)
    y = rng.lognormal(5.5, 0.5, 500)
    preds = {"B1": np.full(500, np.median(y)), "model": y * rng.lognormal(0, 0.2, 500)}
    s = summary_table(y, preds)
    assert list(s.index) == ["B1", "model"] and s.loc["model", "MAPE (%)"] < s.loc["B1", "MAPE (%)"]
    b = by_price_band(y, preds)
    assert len(b) == 10 and b["listings"].sum() == 500
    assert {"MAE model", "MdAPE model (%)", "typical rel. err. B1 (%)"} <= set(b.columns)
