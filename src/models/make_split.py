"""Frozen, grouped, stratified train / calibration / test split.

Design reference: Modelling & Evaluation Design v2, sections 4.1 and 4.2.

* All partitions are disjoint in listing id (a grouped split): every snapshot of a
  listing lands in the same partition.
* Listings are stratified by room type x price quintile of their FIRST snapshot, so
  rare segments are represented in every partition.
* The split is created once, by this script, and written to a manifest of listing ids
  plus a SHA-256 hash. Notebooks only load and verify the manifest; they never re-split.
* Re-running with the same data and seed must reproduce the same hash.

Usage (from the repo root):

    python -m src.models.make_split --data data/processed/listings_features.parquet --city Amsterdam

One split per city: this writes evaluation/splits/split_v1_amsterdam.json (see cities.py).

Notebook 03 reads train and calibration ids. Only notebook 04 reads test rows.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

PARTITIONS = ("train", "calibration", "test")


def _hash_partitions(manifest: dict) -> str:
    """SHA-256 over the three id lists only (seed and metadata are not hashed)."""
    body = {k: manifest[k] for k in PARTITIONS}
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()


def load_listings(path: Path | str) -> pd.DataFrame:
    """Read the processed Listings file (.parquet or .csv)."""
    path = Path(path)
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


def ensure_snapshot_date(df: pd.DataFrame, gap_days: int = 7) -> pd.DataFrame:
    """Guarantee a snapshot_date column: one date per Inside Airbnb release.

    Order of preference:
      1. an existing snapshot_date column;
      2. scrape_id: one date per scrape_id (the earliest last_scraped in it, else the
         YYYYMMDD prefix of scrape_id);
      3. last_scraped only: dates are grouped into releases wherever there is a gap of
         more than `gap_days`; each release is labelled with its first date.
    Notebook 02 should ideally provide snapshot_date; until then this is the single
    place where it is derived, shared by make_split and the notebooks.
    """
    df = df.copy()
    if "snapshot_date" in df.columns:
        df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])
        return df

    if "scrape_id" in df.columns:
        if "last_scraped" in df.columns:
            dates = pd.to_datetime(df["last_scraped"]).groupby(df["scrape_id"]).transform("min")
        else:
            dates = pd.to_datetime(df["scrape_id"].astype(str).str[:8], format="%Y%m%d")
        df["snapshot_date"] = dates.dt.normalize()
        return df

    if "last_scraped" in df.columns:
        scraped = pd.to_datetime(df["last_scraped"]).dt.normalize()
        days = pd.Series(sorted(scraped.dropna().unique()))
        new_release = days.diff().dt.days.fillna(gap_days + 1) > gap_days
        release_start = days.where(new_release).ffill()
        df["snapshot_date"] = scraped.map(dict(zip(days, release_start)))
        assert df["snapshot_date"].notna().all(), "rows with missing last_scraped"
        return df

    raise KeyError("No snapshot_date, scrape_id or last_scraped column: cannot tell snapshots "
                   "apart. Ask the data owner to add snapshot_date (one date per release).")


def select_city(df: pd.DataFrame, city: str | None, city_col: str = "location") -> pd.DataFrame:
    """Keep one city (the README: one city, one currency per model).

    If the city column is absent, the file is assumed to hold a single city.
    If it is present and holds several cities, `city` must name one of them.

    Matching ignores case, spaces and punctuation ("amsterdam" in the data matches
    CITY = "Amsterdam"), and the kept rows get the city column set to `city`, so every
    notebook and the prediction tool see the same spelling whatever the source files used.
    """
    if city_col not in df.columns:
        return df
    cities = sorted(df[city_col].dropna().astype(str).unique())
    if city is None:
        assert len(cities) == 1, (f"the file holds {len(cities)} cities {cities}: "
                                  f"pass the city to keep (--city / CITY)")
        return df
    from src.models.cities import city_slug          # local import: cities.py imports nothing from here
    slugs = df[city_col].astype(str).map(city_slug)
    keep = slugs == city_slug(city)
    assert keep.any(), f"city {city!r} not in the data; available: {cities}"
    out = df[keep].reset_index(drop=True)
    out[city_col] = city
    return out


def parse_price(s: pd.Series) -> pd.Series:
    """'$1,234.00' -> 1234.0; numeric values pass through; unparseable -> NaN."""
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float)
    return pd.to_numeric(s.astype(str).str.replace(r"[^0-9.\-]", "", regex=True), errors="coerce")


def make_split(df: pd.DataFrame, seed: int = 42, test: float = 0.20, calib: float = 0.10,
               id_col: str = "id", price_col: str = "price",
               room_col: str = "room_type", snapshot_col: str = "snapshot_date") -> dict:
    """Return a manifest {"train": [...], "calibration": [...], "test": [...], ...}.

    Stratum = room type x price quintile of the listing's first snapshot. Within each
    stratum, listing ids are shuffled and allocated test -> calibration -> train.
    Strata are processed in sorted order and ids are sorted before shuffling, so the
    result depends only on the data and the seed.
    """
    if not pd.api.types.is_numeric_dtype(df[price_col]):
        raise TypeError(f"'{price_col}' must be numeric (parse currency strings before splitting)")

    first = (df.sort_values([snapshot_col, id_col])
               .groupby(id_col, sort=True)
               .first())
    quintile = pd.qcut(first[price_col], 5, labels=False, duplicates="drop")
    # explicit labels (missing room type or price -> "NA"), independent of the pandas version
    q_label = quintile.map(lambda v: "NA" if pd.isna(v) else str(int(v)))
    strata = first[room_col].fillna("NA").astype(str) + "|q" + q_label

    rng = np.random.default_rng(seed)

    '''
    What parts looks like :

    By that point in make_split, parts holds the ids in the shuffled order they were allocated, as numpy integers:

    python
    parts = {
        "test":        [np.int64(812), np.int64(37),  np.int64(1204), ...],
        "calibration": [np.int64(95),  np.int64(640), ...],
        "train":       [np.int64(1311), np.int64(2),  np.int64(455), ...],
    }
    
    The result has the same three keys, each with a clean, sorted list:

    python
    manifest = {
        "test":        [12, 37, 58, 101, ...],
        "calibration": [9, 44, 95, ...],
        "train":       [2, 3, 5, 7, ...],

    }
    '''

    parts = {k: [] for k in PARTITIONS}
    for stratum in sorted(strata.unique()):
        ids = np.array(sorted(strata.index[strata == stratum]))
        rng.shuffle(ids)
        n_test = int(round(len(ids) * test))
        n_cal = int(round(len(ids) * calib))
        parts["test"] += ids[:n_test].tolist()
        parts["calibration"] += ids[n_test:n_test + n_cal].tolist()
        parts["train"] += ids[n_test + n_cal:].tolist()

    manifest = {k: sorted(int(i) for i in v) for k, v in parts.items()}
    manifest["sha256"] = _hash_partitions(manifest)
    manifest["seed"] = seed
    manifest["shares"] = {"test": test, "calibration": calib}
    manifest["stratified_by"] = f"{room_col} x {price_col} quintile of first snapshot"
    manifest["n_strata"] = int(strata.nunique())
    return manifest


    '''
    Why each part is there:

    int(i): the ids come out of a numpy array, 
    so they’re numpy.int64, not Python int. json.dumps 
    can’t handle numpy integers and raises TypeError: 
    Object of type int64 is not JSON serializable. 
    Converting each one makes the manifest saveable.

    sorted(...): the split depends only on which 
    listings are in each partition, not on the
      order the shuffle happened to visit them. 
      Sorting puts them in one fixed order, which matters for two reasons:

    A stable hash. The very next line hashes the manifest
      with json.dumps(..., sort_keys=True). sort_keys orders 
      the dictionary keys but not the items inside each list, 
      so without sorted() the same partitions listed in a different 
      order would give a different hash.
    Readable changes. If the split ever changes, a sorted list in 
    Git shows exactly which ids moved, instead of a reshuffled wall of numbers.

    The randomness still happens earlier, in rng.shuffle(ids), 
    which decides which listing lands in which partition. Sorting afterwards 
    doesn’t change any assignment; it only changes the order they’re written in.
    '''


def verify_manifest(manifest: dict) -> str:
    """Check the hash and disjointness. Returns the short split version (12 hex chars)."""
    assert _hash_partitions(manifest) == manifest["sha256"], \
        "split manifest hash mismatch: the file was edited or corrupted"
    sets = {k: set(manifest[k]) for k in PARTITIONS}
    for a, b in [("train", "calibration"), ("train", "test"), ("calibration", "test")]:
        overlap = sets[a] & sets[b]
        assert not overlap, f"{len(overlap)} listing ids in both {a} and {b}"
    return manifest["sha256"][:12]


def load_manifest(path: Path | str) -> dict:
    """Load and verify a manifest written by this module."""
    manifest = json.loads(Path(path).read_text())
    verify_manifest(manifest)
    return manifest


def write_manifest(manifest: dict, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=1))


def summarise(manifest: dict) -> pd.DataFrame:
    n = {k: len(manifest[k]) for k in PARTITIONS}
    total = sum(n.values())
    return pd.DataFrame({"listings": n, "share": {k: v / total for k, v in n.items()}})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", required=True, help="processed Listings file (.parquet or .csv)")
    ap.add_argument("--out", default=None,
                    help="manifest path; default evaluation/splits/split_v1_<city>.json")
    ap.add_argument("--city", default=None, help="city to keep, e.g. Amsterdam")
    ap.add_argument("--city-col", default="location")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test", type=float, default=0.20)
    ap.add_argument("--calib", type=float, default=0.10)
    ap.add_argument("--id-col", default="id")
    ap.add_argument("--force", action="store_true", help="overwrite an existing manifest")
    args = ap.parse_args()

    if args.out is None:
        if args.city is None:
            raise SystemExit("pass --city (one split per city), or --out for a single-city file")
        from src.models.cities import split_filename
        args.out = f"evaluation/splits/{split_filename(args.city)}"
    out = Path(args.out)
    if out.exists() and not args.force:
        raise SystemExit(f"{out} exists. The split is frozen; pass --force only on purpose.")

    df = load_listings(args.data)
    df = ensure_snapshot_date(df)
    df = select_city(df, args.city, args.city_col)
    df["price"] = parse_price(df["price"])
    snaps = sorted(df["snapshot_date"].dt.strftime("%Y-%m-%d").unique())
    print(f"loaded {len(df):,} rows, {df[args.id_col].nunique():,} listings, "
          f"city: {args.city or 'single-city file'}, snapshots: {snaps}")

    kwargs = dict(seed=args.seed, test=args.test, calib=args.calib, id_col=args.id_col)
    manifest = make_split(df, **kwargs)
    again = make_split(df, **kwargs)
    assert again["sha256"] == manifest["sha256"], "split is not reproducible"
    manifest["city"] = args.city            # metadata only: not part of the hash
    manifest["data_file"] = Path(args.data).name
    manifest["snapshots"] = snaps

    write_manifest(manifest, out)
    print(summarise(manifest).round(3))
    print(f"wrote {out}  sha256 {manifest['sha256'][:12]}  ({manifest['n_strata']} strata)")


if __name__ == "__main__":
    main()
