"""One model per city: names, currencies and data fingerprints (README: one city, one currency).

Every file that belongs to a city carries the city's slug, so several cities can live side by
side without overwriting each other:

    evaluation/splits/split_v1_<slug>.json          frozen split           (make_split.py)
    models/<model>_<slug>_<version>.joblib/.json    trained candidates     (notebook 03)
    models/cqr_<slug>_<version>.joblib              interval model         (notebook 03)
    reports/<slug>/...                              evaluation results     (notebook 04)

Prices in Inside Airbnb are shown with "$" in every city, but they are in the local currency.
`currency_for` is the single place that knows which one.
"""
from __future__ import annotations

import hashlib
import re

import pandas as pd

# The locations in data/processed/listings_master.parquet (01_Data_Preparation)
CITY_CURRENCY = {
    **{c: "EUR" for c in [
        "Amsterdam", "Antwerp", "Athens", "Barcelona", "Bergamo", "Berlin", "Bologna",
        "Bordeaux", "Brussels", "Crete", "Dublin", "Euskadi", "Florence", "Ghent", "Girona",
        "Lisbon", "Lyon", "Madrid", "Mallorca", "Menorca", "Milan", "Munich", "Naples",
        "Paris", "Porto", "Puglia", "Riga", "Rome", "Rotterdam", "Sevilla", "Sicily",
        "South Aegean", "The Hague", "Thessaloniki", "Valencia", "Venice", "Vienna"]},
    **{c: "GBP" for c in ["Bristol", "Edinburgh", "Greater Manchester", "London"]},
    **{c: "CHF" for c in ["Geneva", "Vaud"]},
    "Prague": "CZK",
    "Copenhagen": "DKK",
    "Oslo": "NOK",
    "dummy_city": "USD",           # synthetic demo data
}


def currency_for(city: str) -> str:
    """Local currency of a city's listing prices; fails loudly for an unknown city."""
    try:
        return CITY_CURRENCY[city]
    except KeyError:
        raise KeyError(f"no currency recorded for {city!r}: add it to CITY_CURRENCY in "
                       "src/models/cities.py") from None


def city_slug(city: str) -> str:
    """'Greater Manchester' -> 'greater_manchester' (safe in file names)."""
    slug = re.sub(r"[^a-z0-9]+", "_", str(city).lower()).strip("_")
    if not slug:
        raise ValueError(f"cannot make a file name from city {city!r}")
    return slug


def split_filename(city: str, version: int = 1) -> str:
    return f"split_v{version}_{city_slug(city)}.json"


def city_data_sha256(df: pd.DataFrame, id_col: str = "id",
                     snapshot_col: str = "snapshot_date") -> str:
    """Fingerprint of ONE city's raw rows, independent of row and column order.

    Hashing the whole multi-city file would flag every city's models as stale whenever any
    city is added or reprocessed. Call this on the selected city's rows before preparation.
    """
    keys = [c for c in [id_col, snapshot_col] if c in df.columns]
    d = df.sort_values(keys, kind="mergesort").reset_index(drop=True)
    d = d[sorted(d.columns)].copy()
    floats = d.select_dtypes("floating").columns
    d[floats] = d[floats].round(9)                # ignore last-digit noise (e.g. CSV round trips)
    d = d.astype(str)                             # one dtype for every column
    h = hashlib.sha256()
    h.update("\x1f".join(d.columns).encode())
    h.update(pd.util.hash_pandas_object(d, index=False).values.tobytes())
    return h.hexdigest()
