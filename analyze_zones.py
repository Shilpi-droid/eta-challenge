#!/usr/bin/env python
"""Analyse the NYC taxi zone shapefile and export a zone_centroids.csv for use in baseline.py.

Prerequisites — run once from the eta-challenge-starter directory:

    Download (PowerShell):
        Invoke-WebRequest https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip -OutFile taxi_zones.zip

    Download (curl / bash):
        curl -L https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip -o taxi_zones.zip

    Unzip (PowerShell):
        Expand-Archive taxi_zones.zip -DestinationPath taxi_zones

    Unzip (bash):
        unzip taxi_zones.zip -d taxi_zones

Then run:
    python analyze_zones.py
"""

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd

SHAPEFILE = Path(__file__).parent / "taxi_zones" / "taxi_zones" / "taxi_zones.shp"
OUT_CSV = Path(__file__).parent / "data" / "zone_centroids.csv"


def main() -> None:
    print("Loading shapefile...")
    zones = gpd.read_file(SHAPEFILE)

    print("\n--- Columns ---")
    print(zones.columns.tolist())

    print("\n--- Sample rows ---")
    print(zones.drop(columns="geometry").head(10).to_string(index=False))

    print(f"\n--- Shape: {zones.shape[0]} zones ---")
    print(f"CRS: {zones.crs}")

    print("\n--- Borough breakdown ---")
    print(zones["borough"].value_counts().to_string())

    print("\n--- LocationID range ---")
    print(f"  min: {zones['LocationID'].min()}  max: {zones['LocationID'].max()}")
    missing = set(range(1, 266)) - set(zones["LocationID"])
    if missing:
        print(f"  Missing IDs: {sorted(missing)}")
    else:
        print("  All 265 IDs present")

    # Compute centroids in lat/lon
    # Project to NY State Plane (EPSG:2263, feet) for accurate centroid geometry,
    # then reproject to WGS84 for lat/lon values.
    zones_proj = zones.to_crs("EPSG:2263")
    zones_proj["centroid"] = zones_proj.geometry.centroid
    centroids_wgs84 = zones_proj["centroid"].to_crs("EPSG:4326")
    zones["lon"] = centroids_wgs84.x
    zones["lat"] = centroids_wgs84.y

    print("\n--- Centroid sample ---")
    print(zones[["LocationID", "zone", "borough", "lon", "lat"]].head(10).to_string(index=False))

    # Save centroid CSV for use in baseline.py
    centroids = zones[["LocationID", "zone", "borough", "lon", "lat"]].sort_values("LocationID")
    centroids.to_csv(OUT_CSV, index=False)
    print(f"\nSaved centroids → {OUT_CSV}")

    # Plot: zones coloured by borough
    fig, ax = plt.subplots(figsize=(10, 10))
    zones.to_crs("EPSG:4326").plot(
        column="borough",
        categorical=True,
        legend=True,
        ax=ax,
        edgecolor="white",
        linewidth=0.3,
        alpha=0.8,
    )
    ax.scatter(zones["lon"], zones["lat"], s=4, c="black", zorder=5, label="Centroids")
    ax.set_title("NYC Taxi Zones with Centroids", fontsize=14, fontweight="bold")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend(loc="upper right")
    ax.axis("equal")

    out_png = Path(__file__).parent / "zone_map.png"
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"Saved map → {out_png}")


if __name__ == "__main__":
    main()
