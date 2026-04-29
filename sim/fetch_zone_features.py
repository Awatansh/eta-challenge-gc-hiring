#!/usr/bin/env python3
"""
Comprehensive zone data pipeline: fetch, process, and enrich trip data with zone features.

Single script for:
1. Downloading NYC taxi zone lookup and shapefile
2. Extracting and reprojecting polygon centroids
3. Aligning to train data zone support
4. Computing zone-pair distances
5. Enriching ALL trip datasets (train, dev, sample) with distance features
6. Generating insights and all output parquets

Outputs:
- {output_dir}/zones.parquet              — Master zone reference (263 zones)
- {output_dir}/zone_distances.parquet     — Zone-pair distance matrix (69,169 pairs)
- {output_dir}/train_features.parquet     — Train data with distance features (36.7M trips)
- {output_dir}/dev_features.parquet       — Dev data with distance features (~1.2M trips)
- {output_dir}/sample_features.parquet    — Sample data with distance features (1M trips)
- {output_dir}/insights.json              — Summary statistics
- {output_dir}/insights.md                — Human-readable report
- {output_dir}/metadata.json              — Metadata about this run

Usage:
    # Default (enriches all 3 datasets, insights from dev): ~10 min
    python fetch_zone_features.py

    # Use train data for insights (slower, ~15 min)
    python fetch_zone_features.py --use-full-train

    # Custom precision and output location
    python fetch_zone_features.py --precision 8 --output-dir sim/outputs
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
import shapefile
from pyproj import CRS, Transformer
from shapely.geometry import shape as shapely_shape


BASE_URL = "https://d37ci6vzurychx.cloudfront.net/misc"
LOOKUP_URL = f"{BASE_URL}/taxi_zone_lookup.csv"
ZONES_URL = f"{BASE_URL}/taxi_zones.zip"

REPO_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_DIR / "data"


class ZoneFeaturePipeline:
    def __init__(
        self,
        train_path: Path = DATA_DIR / "train.parquet",
        dev_path: Path = DATA_DIR / "dev.parquet",
        sample_path: Path = DATA_DIR / "sample_1M.parquet",
        output_dir: Path = Path(__file__).resolve().parent / "zone_outputs",
        cache_dir: Path = Path(__file__).resolve().parent / "zone_data",
        precision: int = 6,
        chunk_size: int = 100000,
        use_full_train: bool = False,
        force_download: bool = False,
    ):
        self.train_path = train_path
        self.dev_path = dev_path
        self.sample_path = sample_path
        self.output_dir = Path(output_dir)
        self.cache_dir = Path(cache_dir)
        self.precision = precision
        self.chunk_size = chunk_size
        self.use_full_train = use_full_train
        self.force_download = force_download

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        (self.cache_dir / "raw").mkdir(parents=True, exist_ok=True)

        self.lookup_csv = self.cache_dir / "raw" / "taxi_zone_lookup.csv"
        self.zones_zip = self.cache_dir / "raw" / "taxi_zones.zip"
        self.extract_dir = self.cache_dir / "raw" / "taxi_zones"

        self.metadata: dict[str, Any] = {
            "created_at": datetime.now().isoformat(),
            "precision": precision,
            "chunk_size": chunk_size,
            "use_full_train": use_full_train,
            "outputs": {},
        }

    def download(self, url: str, out_path: Path) -> Path:
        if out_path.exists() and not self.force_download:
            print(f"  cached   {out_path.name}")
            return out_path
        print(f"  fetching {url}")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        urlretrieve(url, out_path)
        return out_path

    def extract_zip(self, zip_path: Path, out_dir: Path) -> None:
        marker = out_dir / "taxi_zones" / "taxi_zones.shp"
        if marker.exists() and not self.force_download:
            print(f"  cached   shapefile")
            return
        print(f"  extracting {zip_path.name}")
        out_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(out_dir)

    def find_shapefile(self, root: Path) -> Path:
        matches = sorted(root.rglob("*.shp"))
        if not matches:
            raise FileNotFoundError(f"No shapefile found under {root}")
        return matches[0]

    def read_centroids(self, shp_path: Path) -> pd.DataFrame:
        print("  reading centroids and reprojecting to WGS84")
        reader = shapefile.Reader(str(shp_path))
        field_names = [field[0] for field in reader.fields[1:]]
        prj_path = shp_path.with_suffix(".prj")
        if prj_path.exists():
            source_crs = CRS.from_wkt(prj_path.read_text(encoding="utf-8"))
        else:
            source_crs = CRS.from_epsg(2263)
        transformer = Transformer.from_crs(source_crs, CRS.from_epsg(4326), always_xy=True)

        rows: list[dict[str, Any]] = []
        for record, shp in zip(reader.records(), reader.shapes()):
            attrs = dict(zip(field_names, record))
            geom = shapely_shape(shp.__geo_interface__)
            centroid = geom.centroid
            centroid_lon, centroid_lat = transformer.transform(centroid.x, centroid.y)
            rows.append(
                {
                    "LocationID": int(attrs["LocationID"]),
                    "Borough": attrs.get("borough"),
                    "Zone": attrs.get("zone"),
                    "latitude": round(float(centroid_lat), self.precision),
                    "longitude": round(float(centroid_lon), self.precision),
                    "has_polygon_centroid": True,
                }
            )

        centroids = pd.DataFrame(rows).sort_values("LocationID").reset_index(drop=True)
        return centroids

    def load_train_support(self) -> list[int]:
        print(f"  reading {self.train_path.name}")
        train = pd.read_parquet(self.train_path, columns=["pickup_zone", "dropoff_zone"])
        support = sorted(set(train["pickup_zone"].astype(int)).union(train["dropoff_zone"].astype(int)))
        return support

    def build_zone_table(self, lookup: pd.DataFrame, centroids: pd.DataFrame, support: list[int]) -> pd.DataFrame:
        print(f"  aligning {len(lookup):,} lookup zones to train support ({len(support):,} zones)")
        zone_table = (
            lookup.merge(centroids, on="LocationID", how="left", suffixes=("", "_centroid"))
            .query("LocationID in @support")
            .copy()
        )

        zone_table = zone_table.sort_values("LocationID").reset_index(drop=True)
        polygon_mask = zone_table["has_polygon_centroid"].astype("boolean").fillna(False)
        valid_coords = zone_table.loc[polygon_mask, ["latitude", "longitude"]]
        global_lat = round(float(valid_coords["latitude"].median()), self.precision)
        global_lon = round(float(valid_coords["longitude"].median()), self.precision)

        borough_medians = (
            zone_table.loc[polygon_mask]
            .groupby("Borough", dropna=False, observed=True)[["latitude", "longitude"]]
            .median()
            .round(self.precision)
            .rename(columns={"latitude": "borough_lat", "longitude": "borough_lon"})
        )

        zone_table["centroid_source"] = np.where(polygon_mask, "polygon", "needs_fallback")
        for idx, row in zone_table.loc[zone_table["centroid_source"] == "needs_fallback"].iterrows():
            borough = row.get("Borough")
            if pd.notna(borough) and borough in borough_medians.index:
                zone_table.loc[idx, "latitude"] = float(borough_medians.loc[borough, "borough_lat"])
                zone_table.loc[idx, "longitude"] = float(borough_medians.loc[borough, "borough_lon"])
                zone_table.loc[idx, "centroid_source"] = "borough_median"
            else:
                zone_table.loc[idx, "latitude"] = global_lat
                zone_table.loc[idx, "longitude"] = global_lon
                zone_table.loc[idx, "centroid_source"] = "global_median"

        zone_table["is_special_zone"] = zone_table["LocationID"].isin([264, 265])
        zone_table["has_fallback_centroid"] = zone_table["centroid_source"].ne("polygon")
        zone_table = zone_table.sort_values("LocationID").reset_index(drop=True)

        if len(zone_table) != len(support):
            raise ValueError(f"Zone table has {len(zone_table)} rows but train support has {len(support)}")

        return zone_table

    def haversine_km(self, lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
        r = 6371.0
        lat1_rad, lon1_rad, lat2_rad, lon2_rad = map(np.radians, (lat1, lon1, lat2, lon2))
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
        return 2.0 * r * np.arcsin(np.sqrt(a))

    def bearing_degrees(self, lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
        lat1_rad, lon1_rad, lat2_rad, lon2_rad = map(np.radians, (lat1, lon1, lat2, lon2))
        dlon = lon2_rad - lon1_rad
        y = np.sin(dlon) * np.cos(lat2_rad)
        x = np.cos(lat1_rad) * np.sin(lat2_rad) - np.sin(lat1_rad) * np.cos(lat2_rad) * np.cos(dlon)
        return (np.degrees(np.arctan2(y, x)) + 360.0) % 360.0

    def build_zone_distances(self, zone_table: pd.DataFrame) -> pd.DataFrame:
        print(f"  computing pairwise distances for {len(zone_table):,} zones")
        left = zone_table[["LocationID", "latitude", "longitude", "Borough", "Zone"]].rename(
            columns={
                "LocationID": "pickup_zone",
                "latitude": "pickup_latitude",
                "longitude": "pickup_longitude",
                "Borough": "pickup_borough",
                "Zone": "pickup_zone_name",
            }
        )
        right = zone_table[["LocationID", "latitude", "longitude", "Borough", "Zone"]].rename(
            columns={
                "LocationID": "dropoff_zone",
                "latitude": "dropoff_latitude",
                "longitude": "dropoff_longitude",
                "Borough": "dropoff_borough",
                "Zone": "dropoff_zone_name",
            }
        )

        pair_grid = left.assign(_k=1).merge(right.assign(_k=1), on="_k", how="outer").drop(columns=["_k"])
        pair_grid["distance_km"] = self.haversine_km(
            pair_grid["pickup_latitude"].to_numpy(),
            pair_grid["pickup_longitude"].to_numpy(),
            pair_grid["dropoff_latitude"].to_numpy(),
            pair_grid["dropoff_longitude"].to_numpy(),
        ).round(self.precision)
        pair_grid["bearing_degrees"] = self.bearing_degrees(
            pair_grid["pickup_latitude"].to_numpy(),
            pair_grid["pickup_longitude"].to_numpy(),
            pair_grid["dropoff_latitude"].to_numpy(),
            pair_grid["dropoff_longitude"].to_numpy(),
        ).round(1)
        pair_grid["same_borough"] = pair_grid["pickup_borough"].fillna("Unknown") == pair_grid["dropoff_borough"].fillna("Unknown")
        pair_grid["cross_borough"] = ~pair_grid["same_borough"]
        pair_grid = pair_grid.sort_values(["pickup_zone", "dropoff_zone"]).reset_index(drop=True)
        return pair_grid

    def load_and_enrich_trips(self, trips_path: Path, zone_table: pd.DataFrame) -> pd.DataFrame:
        print(f"  loading {trips_path.name} and enriching with distance features")
        columns = ["pickup_zone", "dropoff_zone", "requested_at", "passenger_count", "duration_seconds"]
        df = pd.read_parquet(trips_path, columns=columns)
        df["pickup_zone"] = df["pickup_zone"].astype(int)
        df["dropoff_zone"] = df["dropoff_zone"].astype(int)

        origin = zone_table[["LocationID", "Borough", "Zone", "latitude", "longitude", "centroid_source", "is_special_zone"]].rename(
            columns={
                "LocationID": "pickup_zone",
                "Borough": "pickup_borough",
                "Zone": "pickup_zone_name",
                "latitude": "pickup_latitude",
                "longitude": "pickup_longitude",
                "centroid_source": "pickup_centroid_source",
                "is_special_zone": "pickup_is_special_zone",
            }
        )
        dest = zone_table[["LocationID", "Borough", "Zone", "latitude", "longitude", "centroid_source", "is_special_zone"]].rename(
            columns={
                "LocationID": "dropoff_zone",
                "Borough": "dropoff_borough",
                "Zone": "dropoff_zone_name",
                "latitude": "dropoff_latitude",
                "longitude": "dropoff_longitude",
                "centroid_source": "dropoff_centroid_source",
                "is_special_zone": "dropoff_is_special_zone",
            }
        )

        enriched = df.merge(origin, on="pickup_zone", how="left").merge(dest, on="dropoff_zone", how="left")
        request_ts = pd.to_datetime(enriched["requested_at"], errors="coerce")
        enriched["request_hour"] = request_ts.dt.hour.astype("Int16")
        enriched["request_dayofweek"] = request_ts.dt.dayofweek.astype("Int16")
        enriched["request_month"] = request_ts.dt.month.astype("Int16")
        enriched["is_weekend"] = request_ts.dt.dayofweek.isin([5, 6])
        enriched["is_rush_hour"] = enriched["request_hour"].isin([7, 8, 9, 16, 17, 18, 19])
        enriched["same_borough"] = enriched["pickup_borough"].fillna("Unknown") == enriched["dropoff_borough"].fillna("Unknown")
        enriched["cross_borough"] = ~enriched["same_borough"]
        enriched["distance_km"] = self.haversine_km(
            enriched["pickup_latitude"].to_numpy(),
            enriched["pickup_longitude"].to_numpy(),
            enriched["dropoff_latitude"].to_numpy(),
            enriched["dropoff_longitude"].to_numpy(),
        ).round(self.precision)
        enriched["bearing_degrees"] = self.bearing_degrees(
            enriched["pickup_latitude"].to_numpy(),
            enriched["pickup_longitude"].to_numpy(),
            enriched["dropoff_latitude"].to_numpy(),
            enriched["dropoff_longitude"].to_numpy(),
        ).round(1)
        return enriched

    def compute_insights(self, enriched: pd.DataFrame, pair_distances: pd.DataFrame) -> dict[str, Any]:
        print("  computing insights")
        summary: dict[str, Any] = {}
        summary["trip_rows"] = int(len(enriched))
        summary["pickup_unique_zones"] = int(enriched["pickup_zone"].nunique())
        summary["dropoff_unique_zones"] = int(enriched["dropoff_zone"].nunique())
        summary["distance_duration_corr"] = float(enriched[["distance_km", "duration_seconds"]].corr().iloc[0, 1])
        summary["mean_duration_seconds"] = round(float(enriched["duration_seconds"].mean()), 1)
        summary["median_duration_seconds"] = round(float(enriched["duration_seconds"].median()), 1)
        summary["mean_distance_km"] = round(float(enriched["distance_km"].mean()), 2)
        summary["mean_cross_borough_trip_share"] = round(float(enriched["cross_borough"].mean()), 4)

        top_pickups = (
            enriched.groupby(["pickup_zone", "pickup_zone_name"], dropna=False)
            .size()
            .sort_values(ascending=False)
            .head(10)
            .reset_index(name="trip_count")
        )
        top_dropoffs = (
            enriched.groupby(["dropoff_zone", "dropoff_zone_name"], dropna=False)
            .size()
            .sort_values(ascending=False)
            .head(10)
            .reset_index(name="trip_count")
        )
        top_pairs = (
            enriched.groupby(["pickup_zone", "dropoff_zone"], dropna=False)
            .agg(trip_count=("duration_seconds", "size"), mean_duration_seconds=("duration_seconds", "mean"), mean_distance_km=("distance_km", "mean"))
            .query("trip_count >= 100")
            .sort_values(["trip_count", "mean_duration_seconds"], ascending=[False, False])
            .head(15)
            .reset_index()
        )

        summary["top_pickups"] = top_pickups.to_dict(orient="records")
        summary["top_dropoffs"] = top_dropoffs.to_dict(orient="records")
        summary["top_pairs"] = top_pairs.to_dict(orient="records")
        return summary

    def write_insights(self, summary: dict[str, Any]) -> None:
        json_path = self.output_dir / "insights.json"
        md_path = self.output_dir / "insights.md"
        json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        lines = [
            "# Zone Features Insights",
            "",
            f"- trip rows analyzed: {summary['trip_rows']:,}",
            f"- unique pickup zones: {summary['pickup_unique_zones']}",
            f"- unique dropoff zones: {summary['dropoff_unique_zones']}",
            f"- duration vs distance correlation: {summary['distance_duration_corr']:.4f}",
            f"- mean trip duration: {summary['mean_duration_seconds']:.0f}s",
            f"- median trip duration: {summary['median_duration_seconds']:.0f}s",
            f"- mean trip distance: {summary['mean_distance_km']:.2f} km",
            f"- cross-borough trip share: {summary['mean_cross_borough_trip_share']:.2%}",
            "",
            "## Top Pickup Zones",
        ]
        for row in summary["top_pickups"]:
            lines.append(f"- {row['pickup_zone']} {row['pickup_zone_name']}: {row['trip_count']:,}")
        lines += ["", "## Top Dropoff Zones"]
        for row in summary["top_dropoffs"]:
            lines.append(f"- {row['dropoff_zone']} {row['dropoff_zone_name']}: {row['trip_count']:,}")
        lines += ["", "## Top Zone Pairs"]
        for row in summary["top_pairs"]:
            lines.append(f"- {row['pickup_zone']} → {row['dropoff_zone']}: {row['trip_count']:,} trips, {row['mean_duration_seconds']:.0f}s avg, {row['mean_distance_km']:.1f} km")
        md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        self.metadata["outputs"]["insights_json"] = str(json_path.relative_to(REPO_DIR))
        self.metadata["outputs"]["insights_md"] = str(md_path.relative_to(REPO_DIR))

    def run(self) -> None:
        print("\n" + "=" * 80)
        print("Zone Features Pipeline")
        print("=" * 80)

        print("\nStep 1: Download zone metadata")
        self.download(LOOKUP_URL, self.lookup_csv)
        self.download(ZONES_URL, self.zones_zip)

        print("\nStep 2: Extract and process shapefile")
        self.extract_zip(self.zones_zip, self.extract_dir)
        shapefile_path = self.find_shapefile(self.extract_dir)
        centroids = self.read_centroids(shapefile_path)

        print("\nStep 3: Load lookup and merge centroids")
        lookup = pd.read_csv(self.lookup_csv)
        print(f"  {len(lookup):,} zones in lookup")

        print("\nStep 4: Determine train zone support")
        support = self.load_train_support()
        missing = sorted(set(range(1, 266)) - set(support))
        print(f"  {len(support):,} zones in train support")
        print(f"  zones absent from train: {missing}")

        print("\nStep 5: Build aligned zone table")
        zone_table = self.build_zone_table(lookup, centroids, support)
        zones_path = self.output_dir / "zones.parquet"
        zone_table.to_parquet(zones_path, index=False)
        print(f"  wrote {zones_path.relative_to(REPO_DIR)} ({len(zone_table):,} zones)")
        self.metadata["outputs"]["zones"] = str(zones_path.relative_to(REPO_DIR))

        print("\nStep 6: Build zone-pair distance matrix")
        pair_distances = self.build_zone_distances(zone_table)
        distances_path = self.output_dir / "zone_distances.parquet"
        pair_distances.to_parquet(distances_path, index=False)
        print(f"  wrote {distances_path.relative_to(REPO_DIR)} ({len(pair_distances):,} pairs)")
        self.metadata["outputs"]["zone_distances"] = str(distances_path.relative_to(REPO_DIR))

        print("\nStep 7: Enrich all trip datasets with distance features")
        all_enriched = {}
        
        # Enrich sample
        print("  enriching sample_1M.parquet")
        sample_enriched = self.load_and_enrich_trips(self.sample_path, zone_table)
        sample_features_path = self.output_dir / "sample_features.parquet"
        sample_enriched.to_parquet(sample_features_path, index=False)
        print(f"    wrote {sample_features_path.relative_to(REPO_DIR)} ({len(sample_enriched):,} trips)")
        self.metadata["outputs"]["sample_features"] = str(sample_features_path.relative_to(REPO_DIR))
        all_enriched["sample"] = sample_enriched
        
        # Enrich dev
        print("  enriching dev.parquet")
        dev_enriched = self.load_and_enrich_trips(self.dev_path, zone_table)
        dev_features_path = self.output_dir / "dev_features.parquet"
        dev_enriched.to_parquet(dev_features_path, index=False)
        print(f"    wrote {dev_features_path.relative_to(REPO_DIR)} ({len(dev_enriched):,} trips)")
        self.metadata["outputs"]["dev_features"] = str(dev_features_path.relative_to(REPO_DIR))
        all_enriched["dev"] = dev_enriched
        
        # Enrich train (optionally full or sample)
        print("  enriching train.parquet")
        train_enriched = self.load_and_enrich_trips(self.train_path, zone_table)
        train_features_path = self.output_dir / "train_features.parquet"
        train_enriched.to_parquet(train_features_path, index=False)
        print(f"    wrote {train_features_path.relative_to(REPO_DIR)} ({len(train_enriched):,} trips)")
        self.metadata["outputs"]["train_features"] = str(train_features_path.relative_to(REPO_DIR))
        all_enriched["train"] = train_enriched

        print("\nStep 8: Compute and write insights")
        insights_source = all_enriched["train"] if self.use_full_train else all_enriched["dev"]
        insights_label = "train" if self.use_full_train else "dev"
        print(f"  computing insights from {insights_label} data")
        summary = self.compute_insights(insights_source, pair_distances)
        summary["insights_source"] = insights_label
        self.write_insights(summary)

        print("\nStep 9: Write metadata")
        metadata_path = self.output_dir / "metadata.json"
        metadata_path.write_text(json.dumps(self.metadata, indent=2), encoding="utf-8")
        print(f"  wrote {metadata_path.relative_to(REPO_DIR)}")

        print("\n" + "=" * 80)
        print("Complete. Output files:")
        print("=" * 80)
        for key, path in self.metadata["outputs"].items():
            print(f"  {key:20s}: {path}")
        print("=" * 80 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Comprehensive zone feature pipeline: download, process, and enrich with distance features."
    )
    parser.add_argument(
        "--train-path",
        type=Path,
        default=DATA_DIR / "train.parquet",
        help="Path to training data (used to define zone support).",
    )
    parser.add_argument(
        "--sample-path",
        type=Path,
        default=DATA_DIR / "sample_1M.parquet",
        help="Path to sample data for analytics (default if --use-full-train not set).",
    )
    parser.add_argument(
        "--use-full-train",
        action="store_true",
        help="Use full train.parquet for insights computation instead of dev.parquet. All three datasets (train, dev, sample) are always enriched.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "zone_outputs",
        help="Directory to write all outputs.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "zone_data",
        help="Directory to cache downloads.",
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=6,
        help="Decimal places for latitude/longitude (default 6).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=100000,
        help="Chunk size for memory-efficient processing.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Re-download files even if cached.",
    )

    args = parser.parse_args()
    pipeline = ZoneFeaturePipeline(
        train_path=args.train_path,
        sample_path=args.sample_path,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        precision=args.precision,
        chunk_size=args.chunk_size,
        use_full_train=args.use_full_train,
        force_download=args.force_download,
    )
    pipeline.run()


if __name__ == "__main__":
    main()
