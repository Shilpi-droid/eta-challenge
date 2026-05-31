"""Submission interface — this is what Gobblecube's grader imports.

The grader will call `predict` once per held-out request. The signature below
is fixed; everything else (model type, preprocessing, etc.) is yours to change.
"""

from __future__ import annotations

import math
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np

_MODEL_PATH = Path(__file__).parent / "model.pkl"

with open(_MODEL_PATH, "rb") as _f:
    _ARTIFACT = pickle.load(_f)

_MODEL = _ARTIFACT["model"]
_ZONE_PAIR_DICT: dict[tuple[int, int], float] = _ARTIFACT["zone_pair_lookup"]
_HOUR_DOW_DICT: dict[tuple[int, int], float] = _ARTIFACT["hour_dow_lookup"]
_ZONE_PAIR_HOUR_DICT: dict[tuple[int, int, int], float] = _ARTIFACT["zone_pair_hour_lookup"]
_GLOBAL_MEAN: float = _ARTIFACT["global_mean"]
_ZONE_COORDS: dict[int, tuple[float, float]] = _ARTIFACT["zone_coords"]

_NYC_LAT, _NYC_LON = 40.7128, -74.0060  # fallback for unknown zones
_R = 6371.0  # Earth radius km


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * _R * math.asin(math.sqrt(a))


def predict(request: dict) -> float:
    """Predict trip duration in seconds.

    Input schema:
        {
            "pickup_zone":     int,   # NYC taxi zone, 1-265
            "dropoff_zone":    int,
            "requested_at":    str,   # ISO 8601 datetime
            "passenger_count": int,
        }
    """
    ts = datetime.fromisoformat(request["requested_at"])
    pickup = int(request["pickup_zone"])
    dropoff = int(request["dropoff_zone"])

    hour, dow = ts.hour, ts.weekday()
    p_lat, p_lon = _ZONE_COORDS.get(pickup, (_NYC_LAT, _NYC_LON))
    d_lat, d_lon = _ZONE_COORDS.get(dropoff, (_NYC_LAT, _NYC_LON))
    zone_pair_avg = _ZONE_PAIR_DICT.get((pickup, dropoff), _GLOBAL_MEAN)
    zone_pair_hour_avg = _ZONE_PAIR_HOUR_DICT.get((pickup, dropoff, hour), zone_pair_avg)

    x = np.array([[
        pickup, dropoff, hour, dow, ts.month,
        int(request["passenger_count"]),
        zone_pair_avg,
        _HOUR_DOW_DICT.get((hour, dow), _GLOBAL_MEAN),
        _haversine_km(p_lat, p_lon, d_lat, d_lon),
        zone_pair_hour_avg,
    ]], dtype=np.float32)

    return float(_MODEL.predict(x)[0])
