#!/usr/bin/env python3
"""Train the submission model and save `solution/model.pkl`.

Architecture:
- Prefer `sim/zone_outputs/train_features.parquet` and `sim/zone_outputs/dev_features.parquet`.
- Fall back to raw parquet files in `data/` plus `solution/zone_reference.json`.
- Read parquet in batches, transform each batch, and write the result into disk-backed
  NumPy memmaps so the full dataset can be used without loading everything into RAM.
- Train LightGBM from the memmaps.
- Save the trained model to `solution/model.pkl` and the feature metadata to
  `solution/model_metadata.json` so inference can use the same category encoding.
"""

from __future__ import annotations

import argparse
import json
import pickle
import gc
import subprocess
import tempfile
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
ZONE_OUTPUT_DIR = ROOT / "sim" / "zone_outputs"
DEFAULT_TRAIN_FEATURES = ZONE_OUTPUT_DIR / "train_features.parquet"
DEFAULT_DEV_FEATURES = ZONE_OUTPUT_DIR / "dev_features.parquet"
DEFAULT_TRAIN_RAW = DATA_DIR / "train.parquet"
DEFAULT_DEV_RAW = DATA_DIR / "dev.parquet"
DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "model.pkl"
DEFAULT_METADATA_PATH = Path(__file__).resolve().parent / "model_metadata.json"
DEFAULT_ZONE_REFERENCE = Path(__file__).resolve().parent / "zone_reference.json"

TARGET = "duration_seconds"
FEATURE_COLUMNS = [
    "pickup_zone",
    "dropoff_zone",
    "passenger_count",
    "pickup_borough",
    "pickup_latitude",
    "pickup_longitude",
    "pickup_is_special_zone",
    "dropoff_borough",
    "dropoff_latitude",
    "dropoff_longitude",
    "dropoff_is_special_zone",
    "request_hour",
    "request_dayofweek",
    "request_month",
    "is_weekend",
    "is_rush_hour",
    "same_borough",
    "cross_borough",
    "distance_km",
    "bearing_degrees",
]
CAT_COLS = ["pickup_borough", "dropoff_borough"]
RAW_INPUT_COLS = ["pickup_zone", "dropoff_zone", "requested_at", "passenger_count", TARGET]
ENRICHED_INPUT_COLS = FEATURE_COLUMNS + [TARGET]
DEFAULT_BATCH_SIZE = 100_000


def load_zone_reference(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"Missing zone reference: {path}")
    records = json.loads(path.read_text(encoding="utf-8"))
    zone_table = pd.DataFrame.from_records(records)
    expected = {"LocationID", "Borough", "Zone", "latitude", "longitude", "centroid_source", "is_special_zone"}
    missing = expected.difference(zone_table.columns)
    if missing:
        raise SystemExit(f"Zone reference is missing columns: {sorted(missing)}")
    return zone_table


def gpu_available() -> bool:
    try:
        result = subprocess.run(["nvidia-smi"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        return result.returncode == 0
    except FileNotFoundError:
        return False


def borough_categories(zone_table: pd.DataFrame) -> list[str]:
    categories = sorted({str(value) for value in zone_table["Borough"].fillna("Unknown").astype(str).tolist()})
    if "Unknown" not in categories:
        categories.append("Unknown")
    return categories


def borough_code_map(categories: list[str]) -> dict[str, int]:
    return {value: index for index, value in enumerate(categories)}


def read_batches(path: Path, columns: list[str], batch_size: int, max_rows: int | None):
    parquet_file = pq.ParquetFile(path)
    rows_read = 0
    for batch in parquet_file.iter_batches(batch_size=batch_size, columns=columns):
        frame = batch.to_pandas()
        if max_rows is not None:
            remaining = max_rows - rows_read
            if remaining <= 0:
                break
            if len(frame) > remaining:
                frame = frame.iloc[:remaining].reset_index(drop=True)
        rows_read += len(frame)
        yield frame
        if max_rows is not None and rows_read >= max_rows:
            break


def haversine_km(lat1: pd.Series, lon1: pd.Series, lat2: pd.Series, lon2: pd.Series) -> pd.Series:
    earth_radius_km = 6371.0
    lat1_rad = np.radians(lat1.to_numpy(dtype=np.float64))
    lon1_rad = np.radians(lon1.to_numpy(dtype=np.float64))
    lat2_rad = np.radians(lat2.to_numpy(dtype=np.float64))
    lon2_rad = np.radians(lon2.to_numpy(dtype=np.float64))
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
    return pd.Series(2.0 * earth_radius_km * np.arcsin(np.sqrt(a)))


def bearing_degrees(lat1: pd.Series, lon1: pd.Series, lat2: pd.Series, lon2: pd.Series) -> pd.Series:
    lat1_rad = np.radians(lat1.to_numpy(dtype=np.float64))
    lon1_rad = np.radians(lon1.to_numpy(dtype=np.float64))
    lat2_rad = np.radians(lat2.to_numpy(dtype=np.float64))
    lon2_rad = np.radians(lon2.to_numpy(dtype=np.float64))
    dlon = lon2_rad - lon1_rad
    y = np.sin(dlon) * np.cos(lat2_rad)
    x = np.cos(lat1_rad) * np.sin(lat2_rad) - np.sin(lat1_rad) * np.cos(lat2_rad) * np.cos(dlon)
    return pd.Series((np.degrees(np.arctan2(y, x)) + 360.0) % 360.0)


def enrich_raw_frame(df: pd.DataFrame, zone_table: pd.DataFrame) -> pd.DataFrame:
    enriched = df.copy()
    enriched["pickup_zone"] = enriched["pickup_zone"].astype("int32")
    enriched["dropoff_zone"] = enriched["dropoff_zone"].astype("int32")

    pickup = zone_table.rename(
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
    dropoff = zone_table.rename(
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

    enriched = enriched.merge(
        pickup[["pickup_zone", "pickup_borough", "pickup_zone_name", "pickup_latitude", "pickup_longitude", "pickup_centroid_source", "pickup_is_special_zone"]],
        on="pickup_zone",
        how="left",
    ).merge(
        dropoff[["dropoff_zone", "dropoff_borough", "dropoff_zone_name", "dropoff_latitude", "dropoff_longitude", "dropoff_centroid_source", "dropoff_is_special_zone"]],
        on="dropoff_zone",
        how="left",
    )

    request_ts = pd.to_datetime(enriched["requested_at"], errors="coerce")
    enriched["request_hour"] = request_ts.dt.hour.astype("Int16")
    enriched["request_dayofweek"] = request_ts.dt.dayofweek.astype("Int16")
    enriched["request_month"] = request_ts.dt.month.astype("Int16")
    enriched["is_weekend"] = request_ts.dt.dayofweek.isin([5, 6])
    enriched["is_rush_hour"] = enriched["request_hour"].isin([7, 8, 9, 16, 17, 18, 19])
    enriched["same_borough"] = enriched["pickup_borough"].fillna("Unknown") == enriched["dropoff_borough"].fillna("Unknown")
    enriched["cross_borough"] = ~enriched["same_borough"]
    enriched["distance_km"] = haversine_km(enriched["pickup_latitude"], enriched["pickup_longitude"], enriched["dropoff_latitude"], enriched["dropoff_longitude"])
    enriched["distance_km"] = enriched["distance_km"].round(6)
    enriched["bearing_degrees"] = bearing_degrees(enriched["pickup_latitude"], enriched["pickup_longitude"], enriched["dropoff_latitude"], enriched["dropoff_longitude"])
    enriched["bearing_degrees"] = enriched["bearing_degrees"].round(1)
    return enriched[[*FEATURE_COLUMNS, TARGET]]


def normalize_batch(batch: pd.DataFrame, zone_table: pd.DataFrame | None, enriched: bool) -> pd.DataFrame:
    if enriched:
        return batch[[*FEATURE_COLUMNS, TARGET]].copy()
    if zone_table is None:
        raise SystemExit("Raw parquet requires a zone reference for enrichment.")
    return enrich_raw_frame(batch, zone_table)


def batch_to_matrix(frame: pd.DataFrame, borough_map: dict[str, int]) -> tuple[np.ndarray, np.ndarray]:
    rows = len(frame)
    features = np.empty((rows, len(FEATURE_COLUMNS)), dtype=np.float32)
    for column_index, column in enumerate(FEATURE_COLUMNS):
        series = frame[column]
        if column in CAT_COLS:
            values = series.fillna("Unknown").astype(str).map(lambda value: borough_map.get(value, borough_map["Unknown"]))
            features[:, column_index] = values.to_numpy(dtype=np.float32)
        elif series.dtype == bool or str(series.dtype).startswith("boolean"):
            features[:, column_index] = series.fillna(False).astype(np.int8).to_numpy(dtype=np.float32)
        else:
            features[:, column_index] = pd.to_numeric(series, errors="coerce").fillna(0).to_numpy(dtype=np.float32)
    target = pd.to_numeric(frame[TARGET], errors="coerce").fillna(0).to_numpy(dtype=np.float32)
    return features, target


def count_rows(path: Path, max_rows: int | None) -> int:
    total_rows = pq.ParquetFile(path).metadata.num_rows
    return min(total_rows, max_rows) if max_rows is not None else total_rows


def build_memmaps(
    source_path: Path,
    zone_table: pd.DataFrame | None,
    enriched: bool,
    max_rows: int | None,
    batch_size: int,
    borough_map: dict[str, int],
    temp_dir: Path,
    prefix: str,
) -> tuple[np.memmap, np.memmap, int]:
    rows = count_rows(source_path, max_rows)
    feature_path = temp_dir / f"{prefix}_features.dat"
    target_path = temp_dir / f"{prefix}_target.dat"
    feature_memmap = np.memmap(feature_path, dtype="float32", mode="w+", shape=(rows, len(FEATURE_COLUMNS)))
    target_memmap = np.memmap(target_path, dtype="float32", mode="w+", shape=(rows,))

    offset = 0
    for batch in read_batches(source_path, ENRICHED_INPUT_COLS if enriched else RAW_INPUT_COLS, batch_size=batch_size, max_rows=max_rows):
        if batch.empty:
            continue
        frame = normalize_batch(batch, zone_table, enriched)
        batch_features, batch_target = batch_to_matrix(frame, borough_map)
        end = offset + len(frame)
        feature_memmap[offset:end, :] = batch_features
        target_memmap[offset:end] = batch_target
        offset = end
        if max_rows is not None and offset >= max_rows:
            break

    feature_memmap.flush()
    target_memmap.flush()
    return feature_memmap, target_memmap, offset


def train_lightgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_dev: np.ndarray,
    y_dev: np.ndarray,
    device: str,
) -> lgb.Booster:
    params = {
        "objective": "regression",
        "metric": "mae",
        "device": device,
        "max_bin": 255,
        "learning_rate": 0.05,
        "num_leaves": 256,
        "max_depth": -1,
        "min_data_in_leaf": 100,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbosity": -1,
        "seed": 42,
    }
    if device == "gpu":
        params["gpu_use_dp"] = False

    train_data = lgb.Dataset(
        X_train,
        label=y_train,
        feature_name=FEATURE_COLUMNS,
        categorical_feature=CAT_COLS,
        free_raw_data=True,
    )
    valid_data = lgb.Dataset(
        X_dev,
        label=y_dev,
        feature_name=FEATURE_COLUMNS,
        categorical_feature=CAT_COLS,
        reference=train_data,
        free_raw_data=True,
    )

    return lgb.train(
        params,
        train_data,
        num_boost_round=3000,
        valid_sets=[valid_data],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-parquet", type=Path, default=DEFAULT_TRAIN_FEATURES)
    parser.add_argument("--dev-parquet", type=Path, default=DEFAULT_DEV_FEATURES)
    parser.add_argument("--raw-train-parquet", type=Path, default=DEFAULT_TRAIN_RAW)
    parser.add_argument("--raw-dev-parquet", type=Path, default=DEFAULT_DEV_RAW)
    parser.add_argument("--zone-reference", type=Path, default=DEFAULT_ZONE_REFERENCE)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--metadata-path", type=Path, default=DEFAULT_METADATA_PATH)
    parser.add_argument("--max-rows", type=int, default=None, help="Optional row cap for faster local experiments")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Rows per parquet batch")
    parser.add_argument("--device", choices=["auto", "gpu", "cpu"], default="auto")
    parser.add_argument("--fallback-device", choices=["cpu", "gpu"], default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    zone_table = load_zone_reference(args.zone_reference)

    train_source = args.train_parquet if args.train_parquet.exists() else args.raw_train_parquet
    dev_source = args.dev_parquet if args.dev_parquet.exists() else args.raw_dev_parquet
    train_is_enriched = train_source.name.endswith("features.parquet")
    dev_is_enriched = dev_source.name.endswith("features.parquet")

    if not train_source.exists():
        raise SystemExit(f"Missing train data: {train_source}")
    if not dev_source.exists():
        raise SystemExit(f"Missing dev data: {dev_source}")

    print(f"Loading TRAIN from {train_source}...")
    print(f"Loading DEV from {dev_source}...")
    if train_is_enriched and dev_is_enriched:
        print("Using enriched parquet directly.")
    else:
        print("Using raw parquet with batch enrichment.")

    categories = borough_categories(zone_table)
    borough_map = borough_code_map(categories)

    device = args.device
    if device == "auto":
        device = "gpu" if gpu_available() else "cpu"
    print(f"Training LightGBM on {device.upper()}...")

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        try:
            train_X, train_y, train_rows = build_memmaps(
                train_source,
                zone_table if not train_is_enriched else None,
                train_is_enriched,
                args.max_rows,
                args.batch_size,
                borough_map,
                tmp_path,
                "train",
            )
            dev_X, dev_y, dev_rows = build_memmaps(
                dev_source,
                zone_table if not dev_is_enriched else None,
                dev_is_enriched,
                args.max_rows,
                args.batch_size,
                borough_map,
                tmp_path,
                "dev",
            )
            print(f"  batched train rows written: {train_rows:,}")
            print(f"  batched dev rows written:   {dev_rows:,}")
            model = train_lightgbm(train_X, train_y, dev_X, dev_y, device=device)
        except Exception as exc:
            if device != args.fallback_device:
                print(f"Primary device failed ({exc}); falling back to {args.fallback_device.upper()}.")
                train_X, train_y, _ = build_memmaps(
                    train_source,
                    zone_table if not train_is_enriched else None,
                    train_is_enriched,
                    args.max_rows,
                    args.batch_size,
                    borough_map,
                    tmp_path,
                    "train_retry",
                )
                dev_X, dev_y, _ = build_memmaps(
                    dev_source,
                    zone_table if not dev_is_enriched else None,
                    dev_is_enriched,
                    args.max_rows,
                    args.batch_size,
                    borough_map,
                    tmp_path,
                    "dev_retry",
                )
                model = train_lightgbm(train_X, train_y, dev_X, dev_y, device=args.fallback_device)
            else:
                raise

        # Release memmaps before leaving the temporary directory on Windows.
        del train_X, train_y, dev_X, dev_y
        gc.collect()

    best_mae = None
    if getattr(model, "best_score", None):
        best_mae = model.best_score.get("valid_0", {}).get("l1")
    if best_mae is not None:
        print(f"\nDev MAE: {float(best_mae):.4f} seconds")

    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    with open(args.model_path, "wb") as handle:
        pickle.dump(model, handle)
    print(f"Saved model to {args.model_path}")

    metadata_payload = {
        "feature_names": FEATURE_COLUMNS,
        "pickup_borough_categories": categories,
        "dropoff_borough_categories": categories,
        "pickup_unknown_idx": borough_map.get("Unknown", 0),
        "dropoff_unknown_idx": borough_map.get("Unknown", 0),
    }
    args.metadata_path.write_text(json.dumps(metadata_payload, indent=2), encoding="utf-8")
    print(f"Saved metadata to {args.metadata_path}")


if __name__ == "__main__":
    main()
