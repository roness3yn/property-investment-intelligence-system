"""Listings preparation shared by notebooks 03 and 04 (and later the prediction tool).

Notebook 04 must score the test partition with exactly the same parsing and filtering
that notebook 03 used for training. Keeping the logic here, in one place, is what makes
that true (design chapter 12: "code in src/").

The functions reproduce section 2 of 03_modeling.ipynb:
  * price "$1,234.00" -> 1234.0
  * bathrooms and bathrooms_shared derived from bathrooms_text
  * unique key (city, snapshot_date, id)
  * rows with missing or non-positive price removed (no labels)
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

ID_COL = "id"
SNAPSHOT_COL = "snapshot_date"
TARGET = "price"
DICTIONARY_LISTING_FIELDS = [
    ID_COL, "latitude", "longitude", "neighbourhood_cleansed", "room_type",
    "accommodates", "bedrooms", "bathrooms_text", TARGET,
]


def parse_price(s: pd.Series) -> pd.Series:
    """'$1,234.00' -> 1234.0; numeric values pass through; unparseable -> NaN."""
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float)
    return pd.to_numeric(s.astype(str).str.replace(r"[^0-9.\-]", "", regex=True),
                         errors="coerce")


def parse_bathrooms(text) -> float:
    """'1.5 shared baths' -> 1.5; 'Half-bath' -> 0.5; missing -> NaN."""
    if pd.isna(text):
        return np.nan
    t = str(text).lower()
    if "half" in t and not re.search(r"\d", t):
        return 0.5
    m = re.search(r"\d+(\.\d+)?", t)
    return float(m.group()) if m else np.nan


def prepare_listings(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Section 2 of notebook 03 as a function: check, parse, derive, drop rows without a label."""
    missing = [c for c in DICTIONARY_LISTING_FIELDS + [SNAPSHOT_COL] if c not in df.columns]
    assert not missing, (f"Columns missing from the Listings table: {missing}. "
                         "Check the output of 02_preprocessing against data/data_dictionary.md.")
    df = df.copy()
    df[TARGET] = parse_price(df[TARGET])
    df[SNAPSHOT_COL] = pd.to_datetime(df[SNAPSHOT_COL])
    df["bathrooms"] = df["bathrooms_text"].map(parse_bathrooms)
    df["bathrooms_shared"] = (df["bathrooms_text"].astype(str).str.lower()
                              .str.contains("shared").astype(int))

    key = [c for c in ["city", SNAPSHOT_COL, ID_COL] if c in df.columns]
    dupes = df.duplicated(key)
    assert not dupes.any(), f"{dupes.sum()} duplicate rows on key {key}"

    bad_price = df[TARGET].isna() | (df[TARGET] <= 0)
    df = df.loc[~bad_price].reset_index(drop=True)
    if verbose:
        print(f"Rows removed for missing/non-positive price: {int(bad_price.sum())}")
        print(f"rows: {len(df):,}  listings: {df[ID_COL].nunique():,}  "
              f"snapshots: {df[SNAPSHOT_COL].nunique()}")
    return df


def make_synthetic_listings(n_listings: int = 1500, seed: int = 42,
                            forward_snapshot: str | None = None) -> pd.DataFrame:
    """Synthetic Listings table using the data dictionary's field names (same as notebook 03).

    Grain: snapshot_date + id (single city). Two snapshots, ~75% of listings repeated,
    raw-style price strings and bathrooms_text, so parsing and the grouped split are exercised.

    forward_snapshot: optional date of a third, later snapshot (60% of listings, prices up
    about 6%) used by notebook 04 to demonstrate the forward test. The first two snapshots
    are identical with or without it.
    """
    rng = np.random.default_rng(seed)
    neighbourhoods = ["Downtown", "Old Town", "Waterfront", "Riverside",
                      "North End", "Hill District", "Market Street"]
    hood_premium = dict(zip(neighbourhoods, [1.35, 1.2, 1.4, 1.0, 0.9, 0.85, 1.1]))
    room_types = ["Entire home/apt", "Private room", "Shared room", "Hotel room"]
    room_premium = {"Entire home/apt": 1.0, "Private room": 0.55,
                    "Shared room": 0.35, "Hotel room": 0.8}
    bath_choices = ["1 bath", "1.5 baths", "2 baths", "1 shared bath",
                    "1 private bath", "Half-bath", "2.5 baths"]

    base = pd.DataFrame({
        "id": np.arange(1, n_listings + 1),
        "latitude": rng.uniform(40.55, 40.85, n_listings),
        "longitude": rng.uniform(-74.10, -73.85, n_listings),
        "neighbourhood_cleansed": rng.choice(neighbourhoods, n_listings),
        "room_type": rng.choice(room_types, n_listings, p=[0.55, 0.35, 0.05, 0.05]),
        "accommodates": rng.integers(1, 7, n_listings),
        "bedrooms": rng.integers(0, 4, n_listings).astype(float),
        "bathrooms_text": rng.choice(bath_choices, n_listings),
    })
    base.loc[rng.random(n_listings) < 0.05, "bedrooms"] = np.nan  # realistic gaps

    # Optional Inside Airbnb columns (not in the dictionary), so the feature audit is exercised.
    base["property_type"] = rng.choice(
        ["Entire rental unit", "Private room in rental unit", "Entire condo",
         "Room in hotel", "Entire loft", "Tiny home"], n_listings,
        p=[0.45, 0.30, 0.12, 0.08, 0.04, 0.01])
    base["minimum_nights"] = rng.choice([1, 2, 3, 5, 30], n_listings, p=[0.35, 0.3, 0.2, 0.1, 0.05])
    base["number_of_reviews"] = rng.poisson(12, n_listings)
    base["review_scores_rating"] = np.round(rng.uniform(3.8, 5.0, n_listings), 2)
    base.loc[base["number_of_reviews"] == 0, "review_scores_rating"] = np.nan
    base.loc[rng.random(n_listings) < 0.12, "review_scores_rating"] = np.nan  # new listings
    base["host_is_superhost"] = rng.choice(["t", "f"], n_listings, p=[0.3, 0.7])
    base["availability_30"] = rng.integers(0, 31, n_listings)   # audit: ablation
    base["name"] = [f"Cozy place #{i}" for i in base["id"]]      # audit: defer

    # Price with real structure so the models have something to learn
    signal = (60 + 25 * base["accommodates"]
              * base["neighbourhood_cleansed"].map(hood_premium)
              * base["room_type"].map(room_premium))
    base_price = signal * rng.lognormal(0, 0.25, n_listings)

    snapshots = []
    for snap, frac in [("2026-06-15", 1.0), ("2026-09-15", 0.75)]:
        idx = base.sample(frac=frac, random_state=seed).index
        s = base.loc[idx].copy()
        s["snapshot_date"] = snap
        price = base_price.loc[idx] * rng.lognormal(0, 0.05, len(s))
        s["price"] = ["${:,.2f}".format(p) for p in price]  # raw Inside Airbnb style
        snapshots.append(s)
    if forward_snapshot is not None:          # drawn after the first two, so they are unchanged
        idx = base.sample(frac=0.6, random_state=seed + 1).index
        s = base.loc[idx].copy()
        s["snapshot_date"] = forward_snapshot
        price = base_price.loc[idx] * 1.06 * rng.lognormal(0, 0.08, len(s))
        s["price"] = ["${:,.2f}".format(p) for p in price]
        snapshots.append(s)
    return pd.concat(snapshots, ignore_index=True)
