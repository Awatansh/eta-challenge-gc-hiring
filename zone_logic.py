"""Standalone zone feature serving utilities.

This module exposes two layers:
- `build_zone_features(request)` enriches a single request with zone metadata
  and geospatial features using the generated zone reference.
- `predict(request)` keeps the current submission contract and returns a
  duration prediction from `model.pkl`.

It also works as a script:
- `python zone_logic.py --request '{...}'` prints enriched features as JSON
- `python zone_logic.py --request '{...}' --predict` prints a prediction
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

REPO_DIR = Path(__file__).resolve().parent
MODEL_PATH = REPO_DIR / "model.pkl"
ZONE_REFERENCE_PATH = REPO_DIR / "zone_reference.json"


@lru_cache(maxsize=1)
def load_zone_reference() -> dict[int, dict[str, Any]]:
    if not ZONE_REFERENCE_PATH.exists():
        raise FileNotFoundError(
            f"Missing {ZONE_REFERENCE_PATH.name}. Run `python sim/fetch_zone_features.py` first."
        )

    records = json.loads(ZONE_REFERENCE_PATH.read_text(encoding="utf-8"))
    zone_map: dict[int, dict[str, Any]] = {}
    for record in records:
        zone_id = int(record["LocationID"])
        zone_map[zone_id] = record
    return zone_map


@lru_cache(maxsize=1)
def load_model() -> Any:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing {MODEL_PATH.name}. Run `python baseline.py` first.")

    with open(MODEL_PATH, "rb") as handle:
        model = pickle.load(handle)
    if hasattr(model, "get_booster"):
        model.get_booster().feature_names = None
    return model


def parse_requested_at(requested_at: str) -> datetime:
    value = requested_at.strip()
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    return datetime.fromisoformat(value)


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    earth_radius_km = 6371.0
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = math.sin(dlat / 2.0) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2.0) ** 2
    return 2.0 * earth_radius_km * math.asin(math.sqrt(a))


def bearing_degrees(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)
    dlon = lon2_rad - lon1_rad
    y = math.sin(dlon) * math.cos(lat2_rad)
    x = math.cos(lat1_rad) * math.sin(lat2_rad) - math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(dlon)
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


def get_zone_record(zone_id: int) -> dict[str, Any]:
    zone_map = load_zone_reference()
    try:
        return zone_map[int(zone_id)]
    except KeyError as exc:
        raise KeyError(f"Unknown taxi zone: {zone_id}") from exc


def build_zone_features(request: dict[str, Any]) -> dict[str, Any]:
    pickup_zone = int(request["pickup_zone"])
    dropoff_zone = int(request["dropoff_zone"])
    passenger_count = int(request["passenger_count"])
    requested_at = parse_requested_at(str(request["requested_at"]))

    pickup = get_zone_record(pickup_zone)
    dropoff = get_zone_record(dropoff_zone)

    pickup_latitude = float(pickup["latitude"])
    pickup_longitude = float(pickup["longitude"])
    dropoff_latitude = float(dropoff["latitude"])
    dropoff_longitude = float(dropoff["longitude"])

    distance_km = round(haversine_km(pickup_latitude, pickup_longitude, dropoff_latitude, dropoff_longitude), 6)
    bearing = round(bearing_degrees(pickup_latitude, pickup_longitude, dropoff_latitude, dropoff_longitude), 1)
    same_borough = str(pickup.get("Borough", "Unknown")) == str(dropoff.get("Borough", "Unknown"))

    return {
        "pickup_zone": pickup_zone,
        "dropoff_zone": dropoff_zone,
        "requested_at": requested_at.isoformat(),
        "passenger_count": passenger_count,
        "request_hour": requested_at.hour,
        "request_dayofweek": requested_at.weekday(),
        "request_month": requested_at.month,
        "is_weekend": requested_at.weekday() in (5, 6),
        "is_rush_hour": requested_at.hour in (7, 8, 9, 16, 17, 18, 19),
        "pickup_latitude": pickup_latitude,
        "pickup_longitude": pickup_longitude,
        "pickup_borough": pickup.get("Borough"),
        "pickup_zone_name": pickup.get("Zone"),
        "pickup_centroid_source": pickup.get("centroid_source"),
        "pickup_is_special_zone": bool(pickup.get("is_special_zone")),
        "dropoff_latitude": dropoff_latitude,
        "dropoff_longitude": dropoff_longitude,
        "dropoff_borough": dropoff.get("Borough"),
        "dropoff_zone_name": dropoff.get("Zone"),
        "dropoff_centroid_source": dropoff.get("centroid_source"),
        "dropoff_is_special_zone": bool(dropoff.get("is_special_zone")),
        "distance_km": distance_km,
        "bearing_degrees": bearing,
        "same_borough": same_borough,
        "cross_borough": not same_borough,
    }


def build_model_features(request: dict[str, Any]) -> np.ndarray:
    requested_at = parse_requested_at(str(request["requested_at"]))
    return np.array(
        [[
            int(request["pickup_zone"]),
            int(request["dropoff_zone"]),
            requested_at.hour,
            requested_at.weekday(),
            requested_at.month,
            int(request["passenger_count"]),
        ]],
        dtype=np.int32,
    )


def predict(request: dict[str, Any]) -> float:
    model = load_model()
    features = build_model_features(request)
    return float(model.predict(features)[0])


def _load_request_from_args(args: argparse.Namespace) -> dict[str, Any]:
    if args.request_file:
        return json.loads(Path(args.request_file).read_text(encoding="utf-8"))
    if args.request_json:
        return json.loads(args.request_json)
    payload = sys.stdin.read().strip()
    if not payload:
        raise SystemExit("Provide --request-json, --request-file, or pipe a JSON request on stdin.")
    return json.loads(payload)


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve or inspect NYC taxi zone features for a single request.")
    parser.add_argument("--request-json", help="Request JSON string.")
    parser.add_argument("--request-file", help="Path to a JSON file containing the request.")
    parser.add_argument("--predict", action="store_true", help="Return a duration prediction instead of enriched features.")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print enriched features JSON.")
    args = parser.parse_args()

    request = _load_request_from_args(args)
    if args.predict:
        print(predict(request))
        return

    features = build_zone_features(request)
    if args.pretty:
        print(json.dumps(features, indent=2, sort_keys=True))
    else:
        print(json.dumps(features))


if __name__ == "__main__":
    main()
