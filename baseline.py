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

EXPERIMENT = "zone-pair avg feature"

FEATURES = [
    "pickup_zone",
    "dropoff_zone",
    "hour",
    "dow",
    "month",
    "passenger_count",
    "zone_pair_avg_duration",
]

def build_zone_pair_lookup(df: pd.DataFrame) -> tuple[pd.Series, float]:
    """Compute mean duration per (pickup_zone, dropoff_zone) pair from training data.

    Returns the lookup Series and the global mean fallback value.
    """
    global_mean = float(df["duration_seconds"].mean())
    lookup = (
        df.groupby(["pickup_zone", "dropoff_zone"])["duration_seconds"]
        .mean()
    )
    return lookup, global_mean
    
def engineer_features(
    df: pd.DataFrame,
    zone_pair_lookup: pd.Series,
    global_mean: float,
) -> pd.DataFrame:
    """Turn raw request columns into model features."""
    ts = pd.to_datetime(df["requested_at"])

    # Vectorized merge instead of a Python loop — ~100x faster on large DataFrames
    lookup_df = zone_pair_lookup.rename("zone_pair_avg_duration").reset_index()
    zone_pair_avg = (
        df[["pickup_zone", "dropoff_zone"]]
        .merge(lookup_df, on=["pickup_zone", "dropoff_zone"], how="left")["zone_pair_avg_duration"]
        .fillna(global_mean)
        .astype("float32")
        .to_numpy()
    )

    return pd.DataFrame({
        "pickup_zone":            df["pickup_zone"].astype("int32"),
        "dropoff_zone":           df["dropoff_zone"].astype("int32"),
        "hour":                   ts.dt.hour.astype("int8"),
        "dow":                    ts.dt.dayofweek.astype("int8"),
        "month":                  ts.dt.month.astype("int8"),
        "passenger_count":        df["passenger_count"].astype("int8"),
        "zone_pair_avg_duration": zone_pair_avg,
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

    print("Building zone-pair lookup table...")
    zone_pair_lookup, global_mean = build_zone_pair_lookup(train)
    print(f"  {len(zone_pair_lookup):,} unique zone pairs  |  global mean: {global_mean:.1f}s")

    X_train = engineer_features(train, zone_pair_lookup, global_mean)
    y_train = train["duration_seconds"].to_numpy()
    X_dev = engineer_features(dev, zone_pair_lookup, global_mean)
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

    # Bundle the lookup table with the model so predict.py can use it
    artifact = {
        "model": model,
        "zone_pair_lookup": zone_pair_lookup,
        "global_mean": global_mean,
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
