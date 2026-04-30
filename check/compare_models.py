#!/usr/bin/env python3
"""Compare baseline and solution models on Dev parquet.

Usage:
  python check/compare_models.py [--sample N]

This script loads or trains the baseline `base.pkl` (if missing), loads
the `solution/model.pkl`, scores both on `data/dev.parquet` (or a
random sample) and prints MAE and latency numbers.
"""

from __future__ import annotations

import argparse
import pickle
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DEV_PARQUET = DATA_DIR / "dev.parquet"
BASELINE_MODEL = ROOT / "base.pkl"
SOLUTION_MODEL = ROOT / "solution" / "model.pkl"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def ensure_baseline_model():
    if BASELINE_MODEL.exists():
        print("Baseline model found: base.pkl")
        with open(BASELINE_MODEL, "rb") as f:
            model = pickle.load(f)
        return model

    # Train baseline if missing
    print("Baseline model missing; training baseline.py (this may take minutes)...")
    import baseline as bmod

    bmod.main()
    with open(BASELINE_MODEL, "rb") as f:
        model = pickle.load(f)
    return model


def load_solution_predictor():
    sol_predict = None
    if not SOLUTION_MODEL.exists():
        print("Solution model missing; training solution/train_model.py (this may take minutes)...")
        trainer = ROOT / "solution" / "train_model.py"
        if not trainer.exists():
            raise SystemExit(f"Missing trainer: {trainer}")
        subprocess.run([sys.executable, str(trainer)], check=True, cwd=str(ROOT))
    try:
        # prefer the package entrypoint
        from solution.predict import predict as sol_predict
    except Exception:
        try:
            from predict import predict as sol_predict
        except Exception as exc:
            raise SystemExit(f"Could not import solution predict(): {exc}")
    if not SOLUTION_MODEL.exists():
        raise SystemExit(f"Missing {SOLUTION_MODEL} — ensure your solution/ folder contains model.pkl")
    return sol_predict


def engineer_baseline_features(df: pd.DataFrame) -> pd.DataFrame:
    # lightweight reimplementation matching baseline.py's engineer_features
    ts = pd.to_datetime(df["requested_at"])
    return pd.DataFrame({
        "pickup_zone": df["pickup_zone"].astype("int32"),
        "dropoff_zone": df["dropoff_zone"].astype("int32"),
        "hour": ts.dt.hour.astype("int8"),
        "dow": ts.dt.dayofweek.astype("int8"),
        "month": ts.dt.month.astype("int8"),
        "passenger_count": df["passenger_count"].astype("int8"),
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=50_000, help="Number of Dev rows to sample for quick comparison")
    args = parser.parse_args()

    if not DEV_PARQUET.exists():
        raise SystemExit(f"Missing {DEV_PARQUET}. Run `python data/download_data.py` first.")

    print("Loading Dev parquet (columns)...")
    cols = ["pickup_zone", "dropoff_zone", "requested_at", "passenger_count", "duration_seconds"]
    df = pd.read_parquet(DEV_PARQUET, columns=cols)

    n = len(df)
    sample_n = min(args.sample, n)
    if sample_n < n:
        df = df.sample(n=sample_n, random_state=42).reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    print(f"Scoring {len(df):,} rows on Dev (sample_n={sample_n})")

    # Baseline model
    baseline = ensure_baseline_model()
    X_base = engineer_baseline_features(df)
    try:
        preds_base = baseline.predict(X_base)
    except Exception:
        # fallback if model is a Booster or similar
        preds_base = baseline.predict(X_base.values)

    # Solution model predictions (uses request dict -> predict API)
    sol_predict = load_solution_predictor()
    records = df[["pickup_zone", "dropoff_zone", "requested_at", "passenger_count"]].to_dict("records")

    # measure latency on the sample
    t0 = time.perf_counter()
    preds_sol = [float(sol_predict(r)) for r in records]
    elapsed_ms = (time.perf_counter() - t0) / len(records) * 1000

    truth = df["duration_seconds"].to_numpy(dtype=np.float64)
    preds_base = np.asarray(preds_base, dtype=np.float64)
    preds_sol = np.asarray(preds_sol, dtype=np.float64)

    mae_base = float(np.mean(np.abs(preds_base - truth)))
    mae_sol = float(np.mean(np.abs(preds_sol - truth)))

    print("\nResults:")
    print(f"  Baseline MAE: {mae_base:.3f} seconds")
    print(f"  Solution MAE: {mae_sol:.3f} seconds")
    print(f"  Solution avg latency: {elapsed_ms:.3f} ms/request")
    print(f"  Improvement (baseline - solution): {mae_base - mae_sol:.3f} seconds")


if __name__ == "__main__":
    main()
