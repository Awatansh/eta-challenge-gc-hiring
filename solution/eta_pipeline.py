"""Fast inference pipeline for ETA predictions."""

from __future__ import annotations

import json
import math
import pickle
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parent
MODEL_PATH = REPO_ROOT / "model.pkl"
MODEL_METADATA_PATH = REPO_ROOT / "model_metadata.json"
ZONE_REFERENCE_PATH = PACKAGE_DIR / "zone_reference.json"
KNOWN_CATEGORICAL_FEATURES = {
    "pickup_zone",
    "dropoff_zone",
    "pickup_borough",
    "dropoff_borough",
}


def _parse_requested_at(value: str) -> datetime:
    ts = value.strip()
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    return datetime.fromisoformat(ts)


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    earth_radius_km = 6371.0
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = math.sin(dlat / 2.0) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2.0) ** 2
    return 2.0 * earth_radius_km * math.asin(math.sqrt(a))


def _bearing_degrees(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)
    dlon = lon2_rad - lon1_rad
    y = math.sin(dlon) * math.cos(lat2_rad)
    x = math.cos(lat1_rad) * math.sin(lat2_rad) - math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(dlon)
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


@lru_cache(maxsize=1)
def _load_zone_map() -> dict[int, dict[str, Any]]:
    if not ZONE_REFERENCE_PATH.exists():
        raise FileNotFoundError(f"Missing {ZONE_REFERENCE_PATH.name}.")
    records = json.loads(ZONE_REFERENCE_PATH.read_text(encoding="utf-8"))
    return {int(record["LocationID"]): record for record in records}


@lru_cache(maxsize=1)
def _load_model() -> Any:
    if MODEL_PATH.exists():
        with open(MODEL_PATH, "rb") as handle:
            return pickle.load(handle)

    legacy_path = PACKAGE_DIR / "model.pkl"
    if legacy_path.exists():
        with open(legacy_path, "rb") as handle:
            return pickle.load(handle)

    raise FileNotFoundError(f"Missing {MODEL_PATH.name}.")


@lru_cache(maxsize=1)
def _load_feature_metadata() -> dict[str, Any] | None:
    if MODEL_METADATA_PATH.exists():
        return json.loads(MODEL_METADATA_PATH.read_text(encoding="utf-8"))

    legacy_path = PACKAGE_DIR / "model_metadata.json"
    if legacy_path.exists():
        return json.loads(legacy_path.read_text(encoding="utf-8"))

    return None


@lru_cache(maxsize=1)
def _load_runtime_config() -> dict[str, Any]:
    model = _load_model()
    feature_metadata = _load_feature_metadata()

    if feature_metadata is not None:
        feature_names = list(feature_metadata["feature_names"])
        categorical_maps = {
            "pickup_borough": {value: idx for idx, value in enumerate(feature_metadata["pickup_borough_categories"])},
            "dropoff_borough": {value: idx for idx, value in enumerate(feature_metadata["dropoff_borough_categories"])},
        }
    else:
        feature_names = list(model.feature_name())
        pandas_categorical = list(getattr(model, "pandas_categorical", []))
        categorical_feature_names = [name for name in feature_names if name in KNOWN_CATEGORICAL_FEATURES]
        if len(pandas_categorical) != len(categorical_feature_names):
            raise RuntimeError(
                "Expected categorical metadata for pickup_zone, dropoff_zone, pickup_borough, and dropoff_borough."
            )
        categorical_maps = {
            name: {value: idx for idx, value in enumerate(categories)}
            for name, categories in zip(categorical_feature_names, pandas_categorical)
        }

    return {
        "model": model,
        "feature_names": feature_names,
        "categorical_maps": categorical_maps,
    }


def build_zone_features(request: dict[str, Any]) -> dict[str, Any]:
    pickup_zone = int(request["pickup_zone"])
    dropoff_zone = int(request["dropoff_zone"])
    passenger_count = int(request["passenger_count"])
    requested_at = _parse_requested_at(str(request["requested_at"]))

    zone_map = _load_zone_map()
    if pickup_zone not in zone_map:
        raise KeyError(f"Unknown taxi zone: {pickup_zone}")
    if dropoff_zone not in zone_map:
        raise KeyError(f"Unknown taxi zone: {dropoff_zone}")

    pickup = zone_map[pickup_zone]
    dropoff = zone_map[dropoff_zone]

    pickup_latitude = float(pickup["latitude"])
    pickup_longitude = float(pickup["longitude"])
    dropoff_latitude = float(dropoff["latitude"])
    dropoff_longitude = float(dropoff["longitude"])

    pickup_borough = str(pickup.get("Borough", "Unknown"))
    dropoff_borough = str(dropoff.get("Borough", "Unknown"))
    same_borough = pickup_borough == dropoff_borough

    return {
        "pickup_zone": pickup_zone,
        "dropoff_zone": dropoff_zone,
        "passenger_count": passenger_count,
        "pickup_borough": pickup_borough,
        "pickup_latitude": pickup_latitude,
        "pickup_longitude": pickup_longitude,
        "pickup_is_special_zone": bool(pickup.get("is_special_zone")),
        "dropoff_borough": dropoff_borough,
        "dropoff_latitude": dropoff_latitude,
        "dropoff_longitude": dropoff_longitude,
        "dropoff_is_special_zone": bool(dropoff.get("is_special_zone")),
        "request_hour": requested_at.hour,
        "request_dayofweek": requested_at.weekday(),
        "request_month": requested_at.month,
        "is_weekend": requested_at.weekday() in (5, 6),
        "is_rush_hour": requested_at.hour in (7, 8, 9, 16, 17, 18, 19),
        "same_borough": same_borough,
        "cross_borough": not same_borough,
        "distance_km": round(_haversine_km(pickup_latitude, pickup_longitude, dropoff_latitude, dropoff_longitude), 6),
        "bearing_degrees": round(_bearing_degrees(pickup_latitude, pickup_longitude, dropoff_latitude, dropoff_longitude), 1),
    }


def _encode_features(features: dict[str, Any]) -> np.ndarray:
    config = _load_runtime_config()
    feature_names = config["feature_names"]
    categorical_maps = config["categorical_maps"]

    values: list[float] = []
    for name in feature_names:
        value = features[name]
        if name in categorical_maps:
            category_map = categorical_maps[name]
            unknown_idx = category_map.get("Unknown", 0)
            values.append(float(category_map.get(str(value), unknown_idx)))
        elif isinstance(value, bool):
            values.append(float(int(value)))
        else:
            values.append(float(value))
    return np.array([values], dtype=np.float32)


def predict(request: dict[str, Any]) -> float:
    config = _load_runtime_config()
    model = config["model"]
    features = build_zone_features(request)
    encoded = _encode_features(features)

    # Align encoded vector to the model's expected feature ordering and
    # dimensionality. This avoids needing any hard-coded prediction tweaks
    # while keeping training and inference feature shapes consistent.
    model_feature_names: list[str] = []
    try:
        model_feature_names = list(model.feature_name())
    except Exception:
        # Some model objects may not expose feature_name(); fall back to
        # runtime feature list in that case.
        model_feature_names = list(_load_runtime_config()["feature_names"])  # type: ignore[index]

    if len(model_feature_names) != encoded.shape[1]:
        aligned = np.zeros((1, len(model_feature_names)), dtype=np.float32)
        runtime_names = _load_runtime_config()["feature_names"]  # type: ignore[index]
        for i, fname in enumerate(model_feature_names):
            if fname in runtime_names:
                idx = runtime_names.index(fname)
                aligned[0, i] = encoded[0, idx]
            else:
                # Missing feature at inference time -> leave zero (safe default)
                aligned[0, i] = 0.0
        pred = float(model.predict(aligned)[0])
    else:
        pred = float(model.predict(encoded)[0])

    return max(1.0, pred)
