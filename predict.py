"""Submission interface — this is what Gobblecube's grader imports.

The grader will call `predict` once per held-out request. The signature below
is fixed; everything else (model type, preprocessing, etc.) is yours to change.
"""

from __future__ import annotations

import pickle
from datetime import datetime
from pathlib import Path

import numpy as np

_MODEL_PATH = Path(__file__).parent / "model.pkl"

with open(_MODEL_PATH, "rb") as _f:
    _ARTIFACT = pickle.load(_f)

# Support both old (bare model) and new (dict) artifact formats.
if isinstance(_ARTIFACT, dict):
    _MODEL = _ARTIFACT["model"]
    # Convert to a plain Python dict once at load time: O(1) per-call lookup,
    # no pandas overhead, and no dtype mismatch risk from MultiIndex int64 keys.
    _ZONE_PAIR_DICT: dict | None = {
        (int(k[0]), int(k[1])): float(v)
        for k, v in _ARTIFACT["zone_pair_lookup"].items()
    }
    _GLOBAL_MEAN: float | None = _ARTIFACT["global_mean"]
else:
    _MODEL = _ARTIFACT
    _ZONE_PAIR_DICT = None
    _GLOBAL_MEAN = None

# Disable xgboost's feature-name validation so we can predict on a bare
# numpy array (skips per-call DataFrame construction overhead).
if hasattr(_MODEL, "get_booster"):
    _MODEL.get_booster().feature_names = None



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

    row = [pickup, dropoff, ts.hour, ts.weekday(), ts.month, int(request["passenger_count"])]

    if _ZONE_PAIR_DICT is not None:
        row.append(_ZONE_PAIR_DICT.get((pickup, dropoff), _GLOBAL_MEAN))

    x = np.array([row], dtype=np.float32)
    return float(_MODEL.predict(x)[0])
