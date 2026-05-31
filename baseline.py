#!/usr/bin/env python
"""Baseline: gradient-boosted trees on six simple features + zone-pair average.

Trains in ~5 minutes on a laptop CPU. Produces `model.pkl` which `predict.py`
loads at inference.

Prerequisites:
    python data/download_data.py   # one-time, ~500 MB download

Run:
    python baseline.py             # trains and saves model.pkl

Your job is to replace this file with something better. The grader only cares
about `predict.py` — this file just needs to produce a `model.pkl` that
`predict.py` can load.
"""

from __future__ import annotations

import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

DATA_DIR = Path(__file__).parent / "data"
MODEL_PATH = Path(__file__).parent / "model.pkl"
EXPERIMENTS_PATH = Path(__file__).parent / "experiments.csv"

EXPERIMENT = "haversine distance feature"

FEATURES = [
    "pickup_zone",
    "dropoff_zone",
    "hour",
    "dow",
    "month",
    "passenger_count",
    "zone_pair_avg_duration",
    "hour_dow_avg_duration",
    "haversine_km",
]

_EARTH_RADIUS_KM = 6371.0


def haversine_km(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """Vectorised Haversine distance in kilometres."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * _EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def load_zone_coords() -> dict[int, tuple[float, float]]:
    """Load zone centroid lat/lon from the pre-built CSV."""
    centroids = pd.read_csv(DATA_DIR / "zone_centroids.csv")
    return {int(r.LocationID): (float(r.lat), float(r.lon)) for r in centroids.itertuples()}


def build_lookups(df: pd.DataFrame) -> tuple[pd.Series, pd.Series, float]:
    """Build zone-pair and (hour, dow) mean-duration lookups from training data."""
    global_mean = float(df["duration_seconds"].mean())
    zone_pair_lookup = df.groupby(["pickup_zone", "dropoff_zone"])["duration_seconds"].mean()
    ts = pd.to_datetime(df["requested_at"])
    tmp = df[["duration_seconds"]].copy()
    tmp["hour"] = ts.dt.hour
    tmp["dow"] = ts.dt.dayofweek
    hour_dow_lookup = tmp.groupby(["hour", "dow"])["duration_seconds"].mean()
    return zone_pair_lookup, hour_dow_lookup, global_mean


def engineer_features(
    df: pd.DataFrame,
    zone_pair_lookup: pd.Series,
    hour_dow_lookup: pd.Series,
    global_mean: float,
    zone_coords: dict[int, tuple[float, float]],
) -> pd.DataFrame:
    """Turn raw request columns into model features."""
    ts = pd.to_datetime(df["requested_at"])
    hour = ts.dt.hour.astype("int8")
    dow = ts.dt.dayofweek.astype("int8")

    # Zone-pair target encoding — vectorized merge
    lookup_df = zone_pair_lookup.rename("zone_pair_avg_duration").reset_index()
    zone_pair_avg = (
        df[["pickup_zone", "dropoff_zone"]]
        .merge(lookup_df, on=["pickup_zone", "dropoff_zone"], how="left")["zone_pair_avg_duration"]
        .fillna(global_mean)
        .astype("float32")
        .to_numpy()
    )

    # (hour, dow) target encoding — vectorized merge
    tmp = pd.DataFrame({"hour": hour.values, "dow": dow.values})
    lookup_hd = hour_dow_lookup.rename("hour_dow_avg_duration").reset_index()
    hour_dow_avg = (
        tmp.merge(lookup_hd, on=["hour", "dow"], how="left")["hour_dow_avg_duration"]
        .fillna(global_mean)
        .astype("float32")
        .to_numpy()
    )

    # Haversine distance between zone centroids
    default_lat, default_lon = 40.7128, -74.0060  # NYC centre fallback
    pickup_zones = df["pickup_zone"].astype(int).to_numpy()
    dropoff_zones = df["dropoff_zone"].astype(int).to_numpy()
    p_lat = np.array([zone_coords.get(z, (default_lat, default_lon))[0] for z in pickup_zones])
    p_lon = np.array([zone_coords.get(z, (default_lat, default_lon))[1] for z in pickup_zones])
    d_lat = np.array([zone_coords.get(z, (default_lat, default_lon))[0] for z in dropoff_zones])
    d_lon = np.array([zone_coords.get(z, (default_lat, default_lon))[1] for z in dropoff_zones])
    dist_km = haversine_km(p_lat, p_lon, d_lat, d_lon).astype("float32")

    return pd.DataFrame({
        "pickup_zone":            df["pickup_zone"].astype("int32"),
        "dropoff_zone":           df["dropoff_zone"].astype("int32"),
        "hour":                   hour,
        "dow":                    dow,
        "month":                  ts.dt.month.astype("int8"),
        "passenger_count":        df["passenger_count"].astype("int8"),
        "zone_pair_avg_duration": zone_pair_avg,
        "hour_dow_avg_duration":  hour_dow_avg,
        "haversine_km":           dist_km,
    })[FEATURES]


def main() -> None:
    train_path = DATA_DIR / "train.parquet"
    dev_path = DATA_DIR / "dev.parquet"
    for p in (train_path, dev_path):
        if not p.exists():
            raise SystemExit(
                f"Missing {p.name}. Run `python data/download_data.py` first."
            )

    print("Loading data...")
    train = pd.read_parquet(train_path)
    dev = pd.read_parquet(dev_path)
    print(f"  train: {len(train):,} rows")
    print(f"  dev:   {len(dev):,} rows")

    print("Loading zone centroids...")
    zone_coords = load_zone_coords()
    print(f"  {len(zone_coords)} zones loaded")

    print("Building lookup tables...")
    zone_pair_lookup, hour_dow_lookup, global_mean = build_lookups(train)
    print(f"  {len(zone_pair_lookup):,} zone pairs  |  {len(hour_dow_lookup)} hour-dow combos  |  global mean: {global_mean:.1f}s")

    X_train = engineer_features(train, zone_pair_lookup, hour_dow_lookup, global_mean, zone_coords)
    y_train = train["duration_seconds"].to_numpy()
    X_dev = engineer_features(dev, zone_pair_lookup, hour_dow_lookup, global_mean, zone_coords)
    y_dev = dev["duration_seconds"].to_numpy()

    print("\nTraining XGBoost...")
    model = xgb.XGBRegressor(
        n_estimators=1000,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        n_jobs=-1,
        random_state=42,
        early_stopping_rounds=20,
        eval_metric="mae",
    )
    t0 = time.time()
    model.fit(X_train, y_train, eval_set=[(X_dev, y_dev)], verbose=50)
    print(f"  trained in {time.time() - t0:.0f}s  |  best iteration: {model.best_iteration}")

    preds = model.predict(X_dev)
    mae = float(np.mean(np.abs(preds - y_dev)))
    print(f"\nDev MAE: {mae:.1f} seconds")

    artifact = {
        "model": model,
        "zone_pair_lookup": {(int(k[0]), int(k[1])): float(v) for k, v in zone_pair_lookup.items()},
        "hour_dow_lookup": {(int(k[0]), int(k[1])): float(v) for k, v in hour_dow_lookup.items()},
        "global_mean": global_mean,
        "zone_coords": zone_coords,
    }
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(artifact, f)
    print(f"Saved model to {MODEL_PATH}")

    write_header = not EXPERIMENTS_PATH.exists() or EXPERIMENTS_PATH.stat().st_size == 0
    with open(EXPERIMENTS_PATH, "a", newline="") as f:
        if write_header:
            f.write("Experiment,MAE\n")
        f.write(f"{EXPERIMENT},{mae:.1f}\n")
    print(f"Logged to {EXPERIMENTS_PATH}")


if __name__ == "__main__":
    main()
