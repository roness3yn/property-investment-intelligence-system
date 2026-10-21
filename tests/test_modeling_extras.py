"""Tests for the target policy, the out-of-fold kNN price feature and the G2 search.

Run from the repository root:   python -m pytest tests/test_modeling_extras.py -q
(without pytest:                python -m tests.test_modeling_extras)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from src.models.knn_features import KNNPriceFeature
from src.models.pipelines import knn_k_of, make_model, make_preprocessor, with_features
from src.models.target_policy import (apply_target_policy, fit_target_policy,
                                      policy_scope_problem)
from src.models.tuning import best_params, random_search, rough_importance, sample_params


def listings(n=400, seed=0):
    rng = np.random.default_rng(seed)
    d = pd.DataFrame({"latitude": rng.uniform(52.30, 52.40, n),
                      "longitude": rng.uniform(4.80, 4.95, n),
                      "accommodates": rng.integers(1, 7, n).astype(float),
                      "room_type": rng.choice(["Entire home/apt", "Private room"], n),
                      "minimum_nights": rng.choice([1, 2, 3, 31, 60], n, p=[.4, .3, .2, .05, .05])})
    # price depends on location (east is pricier) and capacity
    d["price"] = (60 + 900 * (d.longitude - 4.80) + 25 * d.accommodates) * rng.lognormal(0, .15, n)
    return d


# ------------------------------------------------------------------ target policy
def test_policy_thresholds_come_from_training_only():
    train, test = listings(seed=1), listings(seed=2)
    test.loc[0, "price"] = 99_999                      # an extreme TEST price must not move the cap
    p = fit_target_policy(train)
    assert p["price_upper"] == float(np.quantile(train.price, 0.995))
    assert p["price_upper"] == fit_target_policy(train)["price_upper"]
    kept, counts = apply_target_policy(test, p)
    assert 99_999 not in kept.price.values
    assert counts["long-term rental"] == int((test.minimum_nights > 30).sum())
    assert counts["total removed"] == len(test) - len(kept)


def test_scope_problem_for_long_term_rentals():
    p = fit_target_policy(listings())
    assert "long-term rental" in policy_scope_problem({"minimum_nights": 60}, p)
    assert policy_scope_problem({"minimum_nights": 3}, p) is None
    assert policy_scope_problem({"minimum_nights": 3}, None) is None


# ------------------------------------------------------------------ kNN feature
def test_oof_feature_ignores_own_price():
    d = listings()
    y = np.log1p(d.price.to_numpy())
    a = KNNPriceFeature(k=10).fit_transform(d[["latitude", "longitude"]], y)
    y2 = y.copy()
    y2[0] = 100.0                                       # change ONE listing's own price wildly
    b = KNNPriceFeature(k=10).fit_transform(d[["latitude", "longitude"]], y2)
    assert a[0, 0] == b[0, 0], "a listing's own price must not enter its own feature"


def test_transform_skips_the_listing_itself():
    d = listings()
    y = np.log1p(d.price.to_numpy())
    f = KNNPriceFeature(k=1).fit(d[["latitude", "longitude"]], y)
    v = f.transform(d[["latitude", "longitude"]].iloc[[0]])[0, 0]
    assert v != y[0], "the nearest neighbour of a training listing must not be itself"


def test_knn_feature_tracks_local_prices():
    d = listings(n=800)
    y = np.log1p(d.price.to_numpy())
    feat = KNNPriceFeature(k=15).fit_transform(d[["latitude", "longitude"]], y)[:, 0]
    assert np.corrcoef(feat, y)[0, 1] > 0.3             # east/west price gradient is picked up


def test_pipeline_with_knn_and_with_features():
    d = listings()
    prep = make_preprocessor(["latitude", "longitude", "accommodates"], [], ["room_type"], knn_k=12)
    pipe = make_model("hist_gbm_tuned", prep, params={"max_leaf_nodes": 15})
    assert knn_k_of(pipe) == 12
    cols = ["latitude", "longitude", "accommodates", "room_type"]
    pipe.fit(d[cols], np.log1p(d.price))
    assert np.isfinite(pipe.predict(d[cols].iloc[:5])).all()
    no_space = with_features(pipe, ["accommodates"], [], ["room_type"])
    assert knn_k_of(no_space) is None                   # dropping coordinates drops kNN
    assert knn_k_of(with_features(pipe, ["latitude", "longitude"], [], [], knn=None)) is None
    assert knn_k_of(with_features(pipe, ["latitude", "longitude"], [], [])) == 12


def test_cv_with_knn_never_uses_validation_prices():
    d = listings()
    cols = ["latitude", "longitude", "accommodates", "room_type"]
    folds = list(GroupKFold(5).split(d, groups=np.arange(len(d))))
    tr, va = folds[0]
    prep = make_preprocessor(["latitude", "longitude", "accommodates"], [], ["room_type"], knn_k=10)
    p1 = make_model("hist_gbm_tuned", prep).fit(d[cols].iloc[tr], np.log1p(d.price.iloc[tr]))
    d2 = d.copy()
    d2.loc[d2.index[va], "price"] *= 50                 # validation prices change wildly
    p2 = make_model("hist_gbm_tuned", prep).fit(d2[cols].iloc[tr], np.log1p(d2.price.iloc[tr]))
    assert np.allclose(p1.predict(d[cols].iloc[va]), p2.predict(d[cols].iloc[va]))


# ------------------------------------------------------------------ tuning
def test_random_search_is_reproducible_and_recorded():
    d = listings()
    cols = ["latitude", "longitude", "accommodates", "room_type"]
    folds = list(GroupKFold(3).split(d, groups=np.arange(len(d))))

    def build(params):
        prep = make_preprocessor(["latitude", "longitude", "accommodates"], [], ["room_type"],
                                 knn_k=params["knn_k"])
        return make_model("hist_gbm_tuned", prep,
                          params={k: v for k, v in params.items() if k != "knn_k"})

    a = random_search(build, d[cols], d.price, folds, np.log1p, np.expm1, n_trials=3, seed=1,
                      verbose=False)
    b = random_search(build, d[cols], d.price, folds, np.log1p, np.expm1, n_trials=3, seed=1,
                      verbose=False)
    assert a["MAE mean"].round(6).equals(b["MAE mean"].round(6))
    assert {"MAE fold 1", "MAE fold 3", "knn_k", "learning_rate"} <= set(a.columns)
    bp = best_params(a)
    assert set(bp) == set(sample_params(np.random.default_rng(0)))
    assert len(rough_importance(a)) == len(bp)


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
