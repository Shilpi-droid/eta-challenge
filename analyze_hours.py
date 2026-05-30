#!/usr/bin/env python
"""Histogram of trip counts and median duration by hour of day."""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent / "data"


def main() -> None:
    print("Loading train data...")
    df = pd.read_parquet(DATA_DIR / "train.parquet", columns=["requested_at", "duration_seconds"])
    df["hour"] = pd.to_datetime(df["requested_at"]).dt.hour

    trips_by_hour = df.groupby("hour").size()
    median_duration = df.groupby("hour")["duration_seconds"].median() / 60  # minutes

    hours = np.arange(24)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle("NYC Taxi Trips by Hour of Day (2023 Training Set)", fontsize=14, fontweight="bold")

    # Top: trip volume
    bars = ax1.bar(hours, trips_by_hour.reindex(hours, fill_value=0) / 1_000_000,
                   color="steelblue", edgecolor="white", linewidth=0.5)
    ax1.set_ylabel("Trips (millions)")
    ax1.set_title("Trip Volume")
    ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1fM"))
    ax1.grid(axis="y", alpha=0.3)

    # Annotate peak hour
    peak_hour = int(trips_by_hour.idxmax())
    ax1.bar(peak_hour, trips_by_hour[peak_hour] / 1_000_000, color="tomato", edgecolor="white", linewidth=0.5, label=f"Peak: {peak_hour}:00")
    ax1.legend()

    # Bottom: median trip duration
    ax2.bar(hours, median_duration.reindex(hours, fill_value=0),
            color="mediumseagreen", edgecolor="white", linewidth=0.5)
    ax2.set_ylabel("Median Duration (min)")
    ax2.set_title("Median Trip Duration")
    ax2.set_xlabel("Hour of Day")
    ax2.set_xticks(hours)
    ax2.set_xticklabels([f"{h:02d}:00" for h in hours], rotation=45, ha="right")
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    out_path = Path(__file__).parent / "hour_analysis.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_path}")

    # Print summary table
    print("\nHour  |  Trips     |  Median Duration")
    print("------|------------|------------------")
    for h in hours:
        count = trips_by_hour.get(h, 0)
        dur = median_duration.get(h, 0)
        peak = " ← peak" if h == peak_hour else ""
        print(f"  {h:02d}  |  {count:>9,}  |  {dur:>6.1f} min{peak}")


if __name__ == "__main__":
    main()
