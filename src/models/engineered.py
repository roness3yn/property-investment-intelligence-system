"""Engineered features from notebooks/03_Feature_Engineering.ipynb, as reusable functions.

The data owner's notebook writes these columns into data/processed/listings_features.parquet,
so for EXISTING listings the models read them from the file. This module recomputes the
same definitions for a HYPOTHETICAL listing in the prediction tool ("a 2-bedroom in Zuid"),
where nobody can type in "competitors within 1 km". Keep the two in step: if the
definitions change in 03_Feature_Engineering, change them here too (tests compare them).

Only the features the feature audit includes are here. The price-derived columns of that
notebook (price_vs_neighbourhood_pct, neighbourhood_median_price, market_median_price)
contain the target and are never model features.
"""
from __future__ import annotations

import ast

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

EARTH_RADIUS_KM = 6371.0088
COMPETITOR_RADII_KM = {"competitors_500m": 0.5, "competitors_1km": 1.0, "competitors_2km": 2.0}
COMPARABLE_FEATURE, COMPARABLE_RADIUS_KM = "comparable_competitors_1km", 1.0
AMENITY_KEYWORDS = {
    "has_wifi": ["wifi", "wi-fi"],
    "has_air_conditioning": ["air conditioning"],
    "has_kitchen": ["kitchen"],
    "has_parking": ["parking"],
    "has_pool": ["pool"],
    "has_workspace": ["dedicated workspace"],
}
AMENITY_FEATURES = ["amenities_count", *AMENITY_KEYWORDS]
COMPETITION_FEATURES = [*COMPETITOR_RADII_KM, COMPARABLE_FEATURE]
NEIGHBOURHOOD_FEATURES = ["neighbourhood_listing_count", "neighbourhood_host_count"]


# ------------------------------------------------------------------ amenities
def parse_amenities(value) -> list[str]:
    """'["Wifi", "Kitchen"]' (Inside Airbnb text) or a list -> lower-case items."""
    if isinstance(value, (list, tuple, np.ndarray)):
        return [str(x).lower() for x in value]
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    try:
        parsed = ast.literal_eval(str(value))
        if isinstance(parsed, (list, tuple)):
            return [str(x).lower() for x in parsed]
    except (ValueError, SyntaxError):
        pass
    return []


def amenity_features(value) -> dict:
    """amenities_count and the has_* keyword flags for one listing."""
    items = parse_amenities(value)
    out = {"amenities_count": len(items)}
    for name, keywords in AMENITY_KEYWORDS.items():
        out[name] = int(any(k in item for item in items for k in keywords))
    return out


# ------------------------------------------------------------------ competition
class CompetitionIndex:
    """Ball trees over one city's listings, for competitor counts around any point.

    Matches 03_Feature_Engineering: haversine distance; all listings of the city for
    competitors_*; same room type for comparable_competitors_1km. A point that is itself
    one of the listings is excluded from its own count (exclude_self=True), as in that
    notebook; a hypothetical listing is not in the data, so nothing is excluded.
    """

    def __init__(self, listings: pd.DataFrame):
        geo = listings.dropna(subset=["latitude", "longitude"])
        geo = geo[geo["latitude"].between(-90, 90) & geo["longitude"].between(-180, 180)
                  & ~((geo["latitude"] == 0) & (geo["longitude"] == 0))]
        self._all = BallTree(np.radians(geo[["latitude", "longitude"]].to_numpy(float)),
                             metric="haversine")
        self._by_room = {
            rt: BallTree(np.radians(g[["latitude", "longitude"]].to_numpy(float)), metric="haversine")
            for rt, g in geo.dropna(subset=["room_type"]).groupby("room_type")
        }

    def counts(self, latitude: float, longitude: float, room_type=None,
               exclude_self: bool = False) -> dict:
        point = np.radians([[float(latitude), float(longitude)]])
        own = 1 if exclude_self else 0
        out = {name: int(self._all.query_radius(point, r=km / EARTH_RADIUS_KM,
                                                count_only=True)[0]) - own
               for name, km in COMPETITOR_RADII_KM.items()}
        tree = self._by_room.get(room_type)
        out[COMPARABLE_FEATURE] = (
            int(tree.query_radius(point, r=COMPARABLE_RADIUS_KM / EARTH_RADIUS_KM,
                                  count_only=True)[0]) - own if tree is not None else 0)
        return out


def neighbourhood_counts(listings: pd.DataFrame, neighbourhood_col: str = "neighbourhood_cleansed",
                         host_col: str = "host_id") -> pd.DataFrame:
    """Listings and distinct hosts per neighbourhood (one city), as in 03_Feature_Engineering."""
    g = listings.dropna(subset=[neighbourhood_col]).groupby(neighbourhood_col)
    out = pd.DataFrame({"neighbourhood_listing_count": g.size()})
    if host_col in listings.columns:
        out["neighbourhood_host_count"] = g[host_col].nunique()
    return out
