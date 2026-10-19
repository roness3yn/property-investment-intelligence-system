"""Tests for the agent's prediction tool (src/models/predict.py).

Run from the repository root:   python -m pytest tests/test_predict.py -q
(without pytest:                python -m tests.test_predict)

Each test session builds a small temporary repository with synthetic listings, a frozen split,
a fitted ridge model and an interval model, so no real data or saved models are needed.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from src.models.data import make_synthetic_listings, prepare_listings
from src.models.feature_audit import apply_row_transforms, load_audit, select_features
from src.models.intervals import fit_cqr, save_cqr
from src.models.make_split import make_split, write_manifest
from src.models.pipelines import make_model, make_preprocessor
from src.models.predict import TOOL_SCHEMA, PredictionTool, SupportLimits

REPO_ROOT = Path(__file__).resolve().parents[1]
CONTRACT_KEYS = {"listing_id", "status", "status_reason", "currency", "predicted_median_price",
                 "interval_80", "observed_price", "position", "in_training_data", "top_drivers",
                 "drivers_method", "assumptions", "model_version", "split_version",
                 "data_snapshot_date", "evidence_id"}


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_repo(root: Path) -> Path:
    """A miniature repository laid out like the real one, trained on synthetic data."""
    (root / "data" / "processed").mkdir(parents=True)
    (root / "evaluation" / "splits").mkdir(parents=True)
    (root / "models").mkdir()
    shutil.copy(REPO_ROOT / "evaluation" / "feature_audit.csv", root / "evaluation")

    raw = make_synthetic_listings(n_listings=600, seed=7)
    data_file = root / "data" / "processed" / "listings_master.csv"
    raw.to_csv(data_file, index=False)

    audit = load_audit(root / "evaluation" / "feature_audit.csv")
    df = apply_row_transforms(prepare_listings(raw, verbose=False), audit)
    fs = select_features(audit, df.columns)
    manifest = make_split(df, seed=42)
    write_manifest(manifest, root / "evaluation" / "splits" / "split_test.json")
    train = df[df["id"].isin(manifest["train"])]
    calib = df[df["id"].isin(manifest["calibration"])].groupby("id").tail(1)

    prep = make_preprocessor(fs.numeric, fs.numeric_indicator, fs.categorical)
    pipe = make_model("ridge", prep).fit(train[fs.all], np.log1p(train["price"]))
    q = lambda a: make_model("hist_gbm", prep).set_params(
        model=HistGradientBoostingRegressor(loss="quantile", quantile=a, max_iter=50))
    cqr = fit_cqr(q(0.1), q(0.9), train[fs.all], train["price"], calib[fs.all], calib["price"])
    split_version = manifest["sha256"][:12]
    cqr.update(split_version=split_version, features=fs.all)
    save_cqr(cqr, root / "models" / "cqr_20260101_0000.joblib")

    art = root / "models" / "ridge_20260101_0000.joblib"
    joblib.dump(pipe, art)
    meta = {"model": "ridge", "version": "20260101_0000",
            "synthetic_data": False,   # the tool ignores synthetic artifacts; this is a test fixture
            "city": "dummy_city", "currency": "USD", "data_source": data_file.name,
            "data_sha256": _sha(data_file), "split_file": "split_test.json",
            "split_version": split_version, "log_target": True, "features": fs.all,
            "development_pick": "ridge", "artifact_sha256": _sha(art)}
    (root / "models" / "ridge_20260101_0000.json").write_text(json.dumps(meta))
    return root


_REPO = None


def repo() -> Path:
    global _REPO
    if _REPO is None:
        _REPO = build_repo(Path(tempfile.mkdtemp(prefix="predict_tool_")))
    return _REPO


def tool(**kw) -> PredictionTool:
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")            # "no model selected by 04 yet"
        return PredictionTool.load(repo(), verbose=False, **kw)


# ------------------------------------------------------------------------------ tests
def test_contract_for_known_listing():
    t = tool()
    lid = int(next(iter(set(t.listings["id"]) - t.train_ids)))
    r = t.predict_listing(lid)
    assert set(r) == CONTRACT_KEYS
    assert r["status"] == "ok" and r["status_reason"] is None
    lo, hi = r["interval_80"]
    assert lo <= r["predicted_median_price"] <= hi
    assert r["position"] in {"below_interval", "within_interval", "above_interval"}
    assert r["in_training_data"] is False
    assert r["evidence_id"] == f"pred:ridge_20260101_0000:{lid}:{r['data_snapshot_date']}"
    json.dumps(r)                                   # JSON-serialisable


def test_position_matches_interval():
    t = tool()
    for lid in t.listings["id"].drop_duplicates().head(40):
        r = t.predict_listing(int(lid))
        lo, hi = r["interval_80"]
        y = r["observed_price"]
        expected = "below_interval" if y < lo else "above_interval" if y > hi else "within_interval"
        assert r["position"] == expected


def test_latest_snapshot_is_default_and_snapshot_can_be_chosen():
    t = tool()
    lid = int(t.listings.groupby("id").size().idxmax())          # listing in both snapshots
    assert t.predict_listing(lid)["data_snapshot_date"] == "2026-09-15"
    assert t.predict_listing(lid, "2026-06-15")["data_snapshot_date"] == "2026-06-15"


def test_training_listing_is_flagged():
    t = tool()
    r = t.predict_listing(int(next(iter(t.train_ids))))
    assert r["in_training_data"] is True


def test_not_found():
    r = tool().predict_listing(10 ** 9)
    assert r["status"] == "not_found" and r["predicted_median_price"] is None


def test_unseen_neighbourhood_is_insufficient_evidence_and_withholds_prices():
    r = tool().predict_features({"neighbourhood_cleansed": "Atlantis",
                                 "room_type": "Entire home/apt", "accommodates": 4,
                                 "latitude": 40.7, "longitude": -74.0})
    assert r["status"] == "insufficient_evidence"
    assert "Atlantis" in r["status_reason"]
    assert r["predicted_median_price"] is None and r["interval_80"] is None


def test_small_neighbourhood_limit():
    r = tool(limits=SupportLimits(min_neighbourhood_listings=10 ** 6)).predict_features(
        {"neighbourhood_cleansed": "Downtown", "room_type": "Private room", "accommodates": 2})
    assert r["status"] == "insufficient_evidence" and "training listings" in r["status_reason"]


def test_hypothetical_listing_fills_coordinates_and_reports_assumptions():
    r = tool().predict_features({"neighbourhood_cleansed": "Downtown", "room_type": "Private room",
                                 "accommodates": 2, "bathrooms_text": "1 shared bath",
                                 "host_is_superhost": "t", "jacuzzi": True})
    assert r["status"] == "ok"
    text = " ".join(r["assumptions"])
    assert "typical location in Downtown" in text and "jacuzzi" in text
    assert r["evidence_id"].startswith("pred:ridge_20260101_0000:hypothetical:")


def test_drivers_shape_and_sign():
    t = tool()
    big = t.predict_features({"neighbourhood_cleansed": "Waterfront", "room_type": "Entire home/apt",
                              "accommodates": 6})
    drivers = {d["feature"]: d for d in big["top_drivers"]}
    assert len(big["top_drivers"]) <= 6
    assert drivers["accommodates"]["effect_pct"] > 0      # 6 guests vs a typical 3-4
    for d in big["top_drivers"]:
        assert set(d) == {"feature", "value", "typical", "effect_pct", "statement"}
        assert d["statement"].startswith(("+", "-"))


def test_invalid_input_never_raises():
    t = tool()
    for call in [{}, {"features": {"room_type": "x"}}, {"listing_id": "not-a-number"},
                 {"features": "nonsense"}]:
        r = t.handle_tool_call(call)
        assert r["status"] == "invalid_input" and r["status_reason"]


def test_evidence_log():
    log = Path(tempfile.mkdtemp()) / "evidence.jsonl"
    t = tool(evidence_log=log)
    lid = int(t.listings["id"].iloc[0])
    r = t.handle_tool_call({"listing_id": lid})
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    assert rows[-1]["evidence_id"] == r["evidence_id"]


def test_tampered_artifact_is_refused():
    t_repo = build_repo(Path(tempfile.mkdtemp(prefix="predict_tool_tamper_")))
    art = t_repo / "models" / "ridge_20260101_0000.joblib"
    art.write_bytes(art.read_bytes() + b"tampered")
    try:
        PredictionTool.load(t_repo, verbose=False)
    except ValueError as exc:
        assert "hash" in str(exc)
    else:
        raise AssertionError("a modified artifact must not load")


def test_tool_schema_is_valid():
    assert TOOL_SCHEMA["name"] == "predict_listing_price"
    props = TOOL_SCHEMA["input_schema"]["properties"]
    assert {"listing_id", "snapshot_date", "features"} <= set(props)
    assert "do not state any price" in TOOL_SCHEMA["description"]


if __name__ == "__main__":                      # runner for machines without pytest
    import sys
    import traceback
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except Exception:
            failed += 1
            print(f"FAIL  {name}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
