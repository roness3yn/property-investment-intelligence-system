"""Tests for the agent's prediction tool (src/models/predict.py) and src/models/cities.py.

Run from the repository root:   python -m pytest tests/test_predict.py -q
(without pytest:                python -m tests.test_predict)

Each test session builds a small temporary repository holding TWO cities (Amsterdam in EUR,
London in GBP) with synthetic listings, a frozen split per city, a fitted ridge model and an
interval model per city, laid out exactly as notebooks 03 and make_split.py write them.
The listings carry 03_Feature_Engineering's columns (competitor counts, amenity flags,
neighbourhood counts, and the price-derived columns the audit must exclude).
No real data or saved models are needed.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from src.models.cities import (city_data_sha256, city_slug, currency_for, split_filename)
from src.models.data import make_synthetic_listings, prepare_listings
from src.models.engineered import (AMENITY_FEATURES, COMPETITION_FEATURES, EARTH_RADIUS_KM,
                                   CompetitionIndex, amenity_features, neighbourhood_counts)
from src.models.feature_audit import apply_row_transforms, load_audit, select_features
from src.models.intervals import fit_cqr, save_cqr
from src.models.make_split import (ensure_snapshot_date, load_listings, make_split, select_city,
                                   write_manifest)
from src.models.pipelines import make_model, make_preprocessor
from src.models.predict import TOOL_SCHEMA, PredictionService, PredictionTool, SupportLimits

REPO_ROOT = Path(__file__).resolve().parents[1]
CITIES = {"Amsterdam": 0, "London": 100_000}          # city -> listing id offset
VERSION = "20260101_0000"
CONTRACT_KEYS = {"listing_id", "city", "status", "status_reason", "currency",
                 "predicted_median_price", "interval_80", "observed_price", "position",
                 "in_training_data", "top_drivers", "drivers_method", "assumptions",
                 "model_version", "split_version", "data_snapshot_date", "evidence_id"}


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


AMENITY_POOL = ["Wifi", "Kitchen", "Air conditioning", "Free parking on premises", "Pool table",
                "Dedicated workspace", "Washer", "Hair dryer"]
LEAKY = ["price_vs_neighbourhood_pct", "neighbourhood_median_price", "market_median_price",
         "price_numeric"]


def add_engineered(d: pd.DataFrame, seed: int) -> pd.DataFrame:
    """03_Feature_Engineering's columns for one city (definitions from src/models/engineered)."""
    rng = np.random.default_rng(seed)
    d = d.copy()
    d["host_id"] = rng.integers(1, 200, len(d))
    d["amenities"] = [json.dumps(list(rng.choice(AMENITY_POOL, rng.integers(0, 6), replace=False)))
                      for _ in range(len(d))]
    d = d.join(pd.DataFrame([amenity_features(v) for v in d["amenities"]], index=d.index))
    idx = CompetitionIndex(d)
    d = d.join(pd.DataFrame([idx.counts(r.latitude, r.longitude, r.room_type, exclude_self=True)
                             for r in d.itertuples()], index=d.index))
    d = d.join(neighbourhood_counts(d), on="neighbourhood_cleansed")
    price = pd.to_numeric(d["price"].str.replace(r"[$,]", "", regex=True))
    d["price_numeric"] = price                                        # leaky: must be excluded
    d["neighbourhood_median_price"] = price.groupby(d["neighbourhood_cleansed"]).transform("median")
    d["market_median_price"] = price.median()
    d["price_vs_neighbourhood_pct"] = 100 * (price / d["neighbourhood_median_price"] - 1)
    return d


def build_repo(root: Path) -> Path:
    """A miniature repository with two cities, trained as notebook 03 does it."""
    (root / "data" / "processed").mkdir(parents=True)
    (root / "evaluation" / "splits").mkdir(parents=True)
    (root / "models").mkdir()
    shutil.copy(REPO_ROOT / "evaluation" / "feature_audit.csv", root / "evaluation")

    frames = []
    for i, (city, offset) in enumerate(CITIES.items()):
        d = make_synthetic_listings(n_listings=500, seed=7 + i)
        d["id"] += offset
        d["location"] = city
        frames.append(add_engineered(d, seed=11 + i))
    data_file = root / "data" / "processed" / "listings_master.csv"
    pd.concat(frames, ignore_index=True).to_csv(data_file, index=False)

    audit = load_audit(root / "evaluation" / "feature_audit.csv")
    all_raw = ensure_snapshot_date(load_listings(data_file))
    for city in CITIES:
        slug = city_slug(city)
        raw = select_city(all_raw, city, "location")
        df = apply_row_transforms(prepare_listings(raw, verbose=False), audit)
        fs = select_features(audit, df.columns)
        manifest = make_split(df, seed=42)
        manifest["city"] = city
        write_manifest(manifest, root / "evaluation" / "splits" / split_filename(city))
        train = df[df["id"].isin(manifest["train"])]
        calib = df[df["id"].isin(manifest["calibration"])].groupby("id").tail(1)

        prep = make_preprocessor(fs.numeric, fs.numeric_indicator, fs.categorical)
        pipe = make_model("ridge", prep).fit(train[fs.all], np.log1p(train["price"]))
        q = lambda a: make_model("hist_gbm", prep).set_params(
            model=HistGradientBoostingRegressor(loss="quantile", quantile=a, max_iter=50))
        cqr = fit_cqr(q(0.1), q(0.9), train[fs.all], train["price"], calib[fs.all], calib["price"])
        split_version = manifest["sha256"][:12]
        cqr.update(split_version=split_version, features=fs.all, city=city)
        save_cqr(cqr, root / "models" / f"cqr_{slug}_{VERSION}.joblib")

        art = root / "models" / f"ridge_{slug}_{VERSION}.joblib"
        joblib.dump(pipe, art)
        meta = {"model": "ridge", "version": VERSION,
                "synthetic_data": False,  # the tool ignores synthetic artifacts; test fixture
                "city": city, "city_slug": slug, "currency": currency_for(city),
                "data_source": data_file.name, "data_sha256": city_data_sha256(raw),
                "split_file": split_filename(city), "split_version": split_version,
                "log_target": True, "features": fs.all, "development_pick": "ridge",
                "artifact_sha256": _sha(art)}
        (root / "models" / f"ridge_{slug}_{VERSION}.json").write_text(json.dumps(meta))
    return root


_REPO = None


def repo() -> Path:
    global _REPO
    if _REPO is None:
        _REPO = build_repo(Path(tempfile.mkdtemp(prefix="predict_tool_")))
    return _REPO


def quiet(fn, *args, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")            # "no model selected by 04 yet"
        return fn(*args, **kw)


def tool(city="Amsterdam", **kw) -> PredictionTool:
    return quiet(PredictionTool.load, repo(), city=city, verbose=False, **kw)


def service(**kw) -> PredictionService:
    return quiet(PredictionService.load, repo(), verbose=False, **kw)


# ------------------------------------------------------------------------------ cities.py
def test_city_helpers():
    assert city_slug("Greater Manchester") == "greater_manchester"
    assert split_filename("The Hague") == "split_v1_the_hague.json"
    assert currency_for("London") == "GBP" and currency_for("Prague") == "CZK"
    try:
        currency_for("Atlantis")
    except KeyError as exc:
        assert "CITY_CURRENCY" in str(exc)
    else:
        raise AssertionError("unknown city must fail")


def test_city_hash_ignores_row_and_column_order():
    d = make_synthetic_listings(n_listings=50)
    shuffled = d.sample(frac=1, random_state=3)[list(d.columns[::-1])]
    assert city_data_sha256(d) == city_data_sha256(shuffled)
    changed = d.copy()
    changed.loc[0, "price"] = "$1.00"
    assert city_data_sha256(changed) != city_data_sha256(d)


# ------------------------------------------------------------------------------ engineered features
def test_audit_excludes_price_derived_columns_and_includes_engineered():
    t = tool()
    assert not set(LEAKY) & set(t.features), "price-derived columns must never be features"
    assert set(COMPETITION_FEATURES) <= set(t.features)
    assert set(AMENITY_FEATURES) <= set(t.features)


def test_amenity_features():
    f = amenity_features('["Wifi", "Pool table", "Kitchenette", "Free parking on premises"]')
    assert f["amenities_count"] == 4 and f["has_wifi"] == 1 and f["has_kitchen"] == 1
    assert f["has_parking"] == 1 and f["has_pool"] == 1 and f["has_air_conditioning"] == 0
    assert amenity_features(None)["amenities_count"] == 0
    assert amenity_features("not a list")["amenities_count"] == 0


def test_competitor_counts_match_brute_force():
    rng = np.random.default_rng(0)
    pts = pd.DataFrame({"latitude": rng.uniform(52.30, 52.40, 400),
                        "longitude": rng.uniform(4.80, 4.95, 400),
                        "room_type": rng.choice(["Entire home/apt", "Private room"], 400)})
    lat, lon = 52.35, 4.88
    p1, p2 = np.radians([lat, lon]), np.radians(pts[["latitude", "longitude"]].to_numpy())
    hav = (np.sin((p2[:, 0] - p1[0]) / 2) ** 2
           + np.cos(p1[0]) * np.cos(p2[:, 0]) * np.sin((p2[:, 1] - p1[1]) / 2) ** 2)
    km = 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(hav))
    got = CompetitionIndex(pts).counts(lat, lon, "Private room")
    assert got["competitors_1km"] == int((km <= 1).sum())
    assert got["competitors_2km"] == int((km <= 2).sum())
    assert got["comparable_competitors_1km"] == int(((km <= 1) & (pts.room_type == "Private room")).sum())


def test_hypothetical_listing_gets_engineered_features():
    r = tool().predict_features({"neighbourhood_cleansed": "Downtown", "room_type": "Private room",
                                 "accommodates": 2, "amenities": ["Wifi", "Kitchen"]})
    assert r["status"] == "ok"
    text = " ".join(r["assumptions"])
    assert "competitor counts computed" in text and "amenity features computed" in text
    assert "neighbourhood counts taken from Downtown" in text
    assert "amenities" not in text.split("ignored unknown fields")[-1] if "ignored" in text else True
    for c in COMPETITION_FEATURES + AMENITY_FEATURES:
        assert c not in text.split("imputed by the model")[-1], f"{c} should not be imputed"


def test_more_competitors_nearby_in_dense_area():
    t = tool()
    dense = t.listings[["latitude", "longitude"]].median()
    far = t.listings[["latitude", "longitude"]].min()
    n_dense = t._competition.counts(dense.latitude, dense.longitude)["competitors_2km"]
    n_far = t._competition.counts(far.latitude, far.longitude)["competitors_2km"]
    assert n_dense > n_far


# ------------------------------------------------------------------------------ one city
def test_contract_for_known_listing():
    t = tool()
    lid = int(next(iter(t.listing_ids - t.train_ids)))
    r = t.predict_listing(lid)
    assert set(r) == CONTRACT_KEYS
    assert r["status"] == "ok" and r["status_reason"] is None and r["city"] == "Amsterdam"
    lo, hi = r["interval_80"]
    assert lo <= r["predicted_median_price"] <= hi
    assert r["position"] in {"below_interval", "within_interval", "above_interval"}
    assert r["in_training_data"] is False and r["currency"] == "EUR"
    assert r["model_version"] == f"ridge_amsterdam_{VERSION}"
    assert r["evidence_id"] == f"pred:ridge_amsterdam_{VERSION}:{lid}:{r['data_snapshot_date']}"
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
    assert t.predict_listing(int(next(iter(t.train_ids))))["in_training_data"] is True


def test_not_found():
    r = tool().predict_listing(10 ** 9)
    assert r["status"] == "not_found" and r["predicted_median_price"] is None


def test_unseen_neighbourhood_is_insufficient_evidence_and_withholds_prices():
    r = tool().predict_features({"neighbourhood_cleansed": "Atlantis",
                                 "room_type": "Entire home/apt", "accommodates": 4,
                                 "latitude": 40.7, "longitude": -74.0})
    assert r["status"] == "insufficient_evidence" and "Atlantis" in r["status_reason"]
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
    assert r["evidence_id"].startswith(f"pred:ridge_amsterdam_{VERSION}:hypothetical:")


def test_drivers_shape_and_sign():
    big = tool().predict_features({"neighbourhood_cleansed": "Waterfront",
                                   "room_type": "Entire home/apt", "accommodates": 6})
    drivers = {d["feature"]: d for d in big["top_drivers"]}
    assert len(big["top_drivers"]) <= 6
    assert drivers["accommodates"]["effect_pct"] > 0      # 6 guests vs a typical 3-4
    for d in big["top_drivers"]:
        assert set(d) == {"feature", "value", "typical", "effect_pct", "statement"}
        assert d["statement"].startswith(("+", "-"))


def test_invalid_input_never_raises():
    t = tool()
    for call in [{}, {"features": {"room_type": "x"}}, {"listing_id": "not-a-number"},
                 {"features": "nonsense"}, {"city": "London", "listing_id": 1}]:
        r = t.handle_tool_call(call)
        assert r["status"] == "invalid_input" and r["status_reason"]


def test_evidence_log():
    log = Path(tempfile.mkdtemp()) / "evidence.jsonl"
    t = tool(evidence_log=log)
    r = t.handle_tool_call({"listing_id": int(t.listings["id"].iloc[0])})
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    assert rows[-1]["evidence_id"] == r["evidence_id"]


def test_model_card_selection_removes_provisional_warning():
    card_dir = repo() / "reports" / "london"
    card_dir.mkdir(parents=True, exist_ok=True)
    (card_dir / "model_card_metrics.json").write_text(json.dumps(
        {"city": "London", "label": "CONFIRMATORY", "selected_model": "ridge"}))
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            PredictionTool.load(repo(), city="London", verbose=False)
        assert not [w for w in caught if "development pick" in str(w.message)]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            PredictionTool.load(repo(), city="Amsterdam", verbose=False)   # no card yet
        assert [w for w in caught if "development pick" in str(w.message)]
    finally:
        shutil.rmtree(card_dir)


# ------------------------------------------------------------------------------ several cities
def test_single_tool_needs_city_when_several_exist():
    try:
        quiet(PredictionTool.load, repo(), verbose=False)
    except ValueError as exc:
        assert "several cities" in str(exc)
    else:
        raise AssertionError("ambiguous city must fail")


def test_service_routes_by_listing_id_and_uses_city_currency():
    s = service()
    assert s.cities == ["Amsterdam", "London"]
    lid = int(next(iter(s.tools["London"].listing_ids)))
    r = s.handle_tool_call({"listing_id": lid})
    assert r["city"] == "London" and r["currency"] == "GBP" and r["status"] == "ok"
    assert r["model_version"] == f"ridge_london_{VERSION}"
    lid = int(next(iter(s.tools["Amsterdam"].listing_ids)))
    assert s.predict_listing(lid)["currency"] == "EUR"


def test_service_with_explicit_city():
    s = service()
    lid = int(next(iter(s.tools["Amsterdam"].listing_ids)))
    assert s.handle_tool_call({"city": "Amsterdam", "listing_id": lid})["status"] == "ok"
    r = s.handle_tool_call({"city": "London", "listing_id": lid})       # wrong city
    assert r["status"] == "not_found" and r["city"] == "London"


def test_service_errors_are_contract_shaped():
    s = service()
    calls = {"invalid_input": [{"city": "Atlantis", "listing_id": 1},
                               {"features": {"neighbourhood_cleansed": "Downtown",
                                             "room_type": "Private room", "accommodates": 2}},
                               {"listing_id": "abc"}],
             "not_found": [{"listing_id": 10 ** 9}]}
    for status, items in calls.items():
        for call in items:
            r = s.handle_tool_call(call)
            assert set(r) == CONTRACT_KEYS and r["status"] == status and r["status_reason"]
            assert r["predicted_median_price"] is None


def test_service_hypothetical_listing_with_city():
    r = service().handle_tool_call({"city": "London", "features": {
        "neighbourhood_cleansed": "Downtown", "room_type": "Private room", "accommodates": 2}})
    assert r["status"] == "ok" and r["currency"] == "GBP"


def test_changed_data_in_one_city_warns_only_for_that_city():
    t_repo = build_repo(Path(tempfile.mkdtemp(prefix="predict_tool_changed_")))
    data_file = t_repo / "data" / "processed" / "listings_master.csv"
    df = pd.read_csv(data_file)
    df.loc[df["location"] == "London", "price"] = "$999.00"             # London reprocessed
    df.to_csv(data_file, index=False)
    for city, expect_warning in [("Amsterdam", False), ("London", True)]:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            PredictionTool.load(t_repo, city=city, verbose=False)
        changed = [w for w in caught if "differ from those the model was trained on" in str(w.message)]
        assert bool(changed) == expect_warning, city


def test_tampered_artifact_is_refused():
    t_repo = build_repo(Path(tempfile.mkdtemp(prefix="predict_tool_tamper_")))
    art = t_repo / "models" / f"ridge_amsterdam_{VERSION}.joblib"
    art.write_bytes(art.read_bytes() + b"tampered")
    try:
        quiet(PredictionTool.load, t_repo, city="Amsterdam", verbose=False)
    except ValueError as exc:
        assert "hash" in str(exc)
    else:
        raise AssertionError("a modified artifact must not load")


def test_tool_schema_is_valid():
    assert TOOL_SCHEMA["name"] == "predict_listing_price"
    props = TOOL_SCHEMA["input_schema"]["properties"]
    assert {"city", "listing_id", "snapshot_date", "features"} <= set(props)
    assert "do not state any price" in TOOL_SCHEMA["description"]
    assert "currency" in TOOL_SCHEMA["description"]


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
