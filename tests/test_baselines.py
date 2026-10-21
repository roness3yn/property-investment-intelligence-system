"""Tests for the baselines (src/models/baselines.py).

Run from the repository root:   python -m pytest tests/test_baselines.py -q
(without pytest:                python -m tests.test_baselines)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.baselines import (B1_MIN_COUNT, GlobalMedianBaseline, HierarchicalMedian,
                                  make_b1, make_b1_simple)


def training_data():
    """Zuid/Entire: 12 listings; Zuid/Private: 2; Centrum/Private: 15; Centrum/Shared: 1."""
    rows = ([("Zuid", "Entire home/apt", p) for p in range(200, 320, 10)]        # 12, median 255
            + [("Zuid", "Private room", 60), ("Zuid", "Private room", 400)]       # 2 (too few)
            + [("Centrum", "Private room", p) for p in range(100, 250, 10)]       # 15, median 170
            + [("Centrum", "Shared room", 45)])                                   # 1 (too few)
    d = pd.DataFrame(rows, columns=["neighbourhood_cleansed", "room_type", "price"])
    return d[["neighbourhood_cleansed", "room_type"]], d["price"]


def query(rows):
    return pd.DataFrame(rows, columns=["neighbourhood_cleansed", "room_type"])


def test_b1_is_hierarchical_with_min_count():
    b1 = make_b1()
    assert isinstance(b1, HierarchicalMedian) and b1.min_count == B1_MIN_COUNT == 10


def test_large_group_uses_its_own_median():
    X, y = training_data()
    b1 = make_b1().fit(X, y)
    assert b1.predict(query([("Zuid", "Entire home/apt")]))[0] == 255
    assert b1.fallback_level(query([("Zuid", "Entire home/apt")]))[0] == "neighbourhood_cleansed x room_type"


def test_small_group_backs_off_to_room_type():
    X, y = training_data()
    b1 = make_b1().fit(X, y)
    # Zuid/Private has 2 listings -> city-wide private-room median (17 listings)
    private = y[X.room_type == "Private room"].median()
    q = query([("Zuid", "Private room")])
    assert b1.predict(q)[0] == private
    assert b1.fallback_level(q)[0] == "room_type" and b1.is_fallback(q)[0]


def test_rare_room_type_backs_off_to_global():
    X, y = training_data()
    b1 = make_b1().fit(X, y)
    q = query([("Centrum", "Shared room"), ("Nowhere", "Hotel room")])
    assert (b1.predict(q) == y.median()).all()
    assert list(b1.fallback_level(q)) == ["global", "global"]


def test_simple_b1_trusts_tiny_groups():
    X, y = training_data()
    simple = make_b1_simple().fit(X, y)
    assert simple.predict(query([("Centrum", "Shared room")]))[0] == 45      # one listing
    assert simple.predict(query([("Zuid", "Private room")]))[0] == 230       # two listings
    assert simple.predict(query([("Nowhere", "Hotel room")]))[0] == y.median()


def test_identical_when_every_group_is_large():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"neighbourhood_cleansed": rng.choice(["A", "B"], 400),
                      "room_type": rng.choice(["Entire home/apt", "Private room"], 400)})
    y = pd.Series(rng.lognormal(5, 0.4, 400))
    assert np.allclose(make_b1().fit(X, y).predict(X), make_b1_simple().fit(X, y).predict(X))


def test_b0():
    X, y = training_data()
    assert (GlobalMedianBaseline().fit(X, y).predict(X) == y.median()).all()


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
