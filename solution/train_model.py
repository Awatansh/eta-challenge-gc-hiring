#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
import subprocess
import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import lightgbm as lgb
import pandas as pd
from sklearn.metrics import mean_absolute_error


# ================================
# CONFIG
# ================================
TARGET = "duration_seconds"

DROP_COLS = {
    TARGET,
    "requested_at",
    "pickup_zone_name",
    "dropoff_zone_name",
    "pickup_centroid_source",
    "dropoff_centroid_source",
}

CAT_COLS = [
    "pickup_borough",
    "dropoff_borough",
    "pickup_zone",
    "dropoff_zone",
]

# ================================
# AUTO PATH RESOLUTION
# ================================
def get_repo_root() -> Path:
    """Find repo root dynamically."""
    try:
        return Path(__file__).resolve().parents[1]
    except NameError:
        return Path.cwd()


REPO_ROOT = get_repo_root()

DEFAULT_TRAIN_PATH = REPO_ROOT / "sim/zone_outputs/train_features.parquet"
DEFAULT_DEV_PATH   = REPO_ROOT / "sim/zone_outputs/dev_features.parquet"

DEFAULT_MODEL_PATH = REPO_ROOT / "model.pkl"
DEFAULT_METADATA_PATH = REPO_ROOT / "model_metadata.json"


# ================================
# UTILITIES
# ================================
def gpu_available() -> bool:
    try:
        return subprocess.run(
            ["nvidia-smi"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode == 0
    except Exception:
        return False


def get_device(device: str = "auto") -> str:
    return "gpu" if device == "auto" and gpu_available() else device


def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    float_cols = df.select_dtypes(include=["float64"]).columns
    int_cols = df.select_dtypes(include=["int64"]).columns

    df[float_cols] = df[float_cols].astype("float32")
    df[int_cols] = df[int_cols].astype("int32")

    return df


# ================================
# DATA PREP
# ================================
def prepare_data(
    df_train: pd.DataFrame,
    df_dev: pd.DataFrame,
    max_rows: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:

    if max_rows:
        df_train = df_train.head(max_rows)
        df_dev = df_dev.head(max_rows)

    features = [c for c in df_train.columns if c not in DROP_COLS]

    X_train = df_train[features].copy()
    y_train = df_train[TARGET].copy()

    # safe alignment
    X_dev = df_dev.reindex(columns=features).copy()
    y_dev = df_dev[TARGET].copy()

    # categorical handling
    for col in CAT_COLS:
        if col in X_train.columns:
            X_train[col] = X_train[col].astype("string").fillna("Unknown")
            X_dev[col] = X_dev[col].astype("string").fillna("Unknown")

            X_train[col] = pd.Categorical(X_train[col])
            X_dev[col] = pd.Categorical(
                X_dev[col],
                categories=X_train[col].cat.categories,
            )

    X_train = optimize_dtypes(X_train)
    X_dev = optimize_dtypes(X_dev)

    return X_train, y_train, X_dev, y_dev


# ================================
# MODEL
# ================================
def build_params(device: str):
    params = {
        "objective": "regression",
        "metric": "mae",
        "device": device,
        "learning_rate": 0.05,
        "num_leaves": 256,
        "min_data_in_leaf": 100,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "max_bin": 255,
        "verbosity": -1,  # 🔥 suppress logs
        "seed": 42,
    }

    if device == "gpu":
        params["gpu_use_dp"] = False

    return params


def train_model(X_train, y_train, X_dev, y_dev, device):
    params = build_params(device)

    train_data = lgb.Dataset(
        X_train,
        label=y_train,
        categorical_feature=[c for c in CAT_COLS if c in X_train.columns],
    )

    valid_data = lgb.Dataset(X_dev, label=y_dev)

    model = lgb.train(
        params,
        train_data,
        num_boost_round=3000,
        valid_sets=[valid_data],
        callbacks=[
            lgb.early_stopping(100),
            lgb.log_evaluation(100),
        ],
    )

    return model


# ================================
# TRAIN
# ================================
def train(df_train, df_dev, device="auto"):
    # ✅ ALWAYS resolve device properly
    if device == "auto":
        device = "gpu" if gpu_available() else "cpu"

    print("Preparing data...")
    X_train, y_train, X_dev, y_dev = prepare_data(df_train, df_dev)

    print(f"Train: {X_train.shape}")
    print(f"Dev:   {X_dev.shape}")

    print(f"Training on {device.upper()}...")

    # 🔥 better suppression (must be before training)
    import os
    os.environ["LIGHTGBM_VERBOSE"] = "-1"

    try:
        model = train_model(X_train, y_train, X_dev, y_dev, device)

    except Exception as e:
        # 🔥 fallback (VERY IMPORTANT)
        print(f"⚠️ {device.upper()} failed → falling back to CPU")
        print("Error:", e)

        model = train_model(X_train, y_train, X_dev, y_dev, "cpu")

    print("Evaluating...")
    preds = model.predict(X_dev)
    mae = mean_absolute_error(y_dev, preds)
    print(f"🔥 MAE: {mae:.5f}")

    with open(DEFAULT_MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    print(f"✅ Model saved at: {DEFAULT_MODEL_PATH}")

    return model


# ================================
# CLI
# ================================
def cli_main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train-parquet", type=Path)
    parser.add_argument("--dev-parquet", type=Path)
    parser.add_argument("--device", default="auto")

    args, _ = parser.parse_known_args()

    # 🔥 AUTO PATH FALLBACK
    train_path = args.train_parquet or DEFAULT_TRAIN_PATH
    dev_path   = args.dev_parquet or DEFAULT_DEV_PATH

    print(f"Using TRAIN: {train_path}")
    print(f"Using DEV:   {dev_path}")

    if not train_path.exists():
        raise FileNotFoundError(f"Train not found: {train_path}")
    if not dev_path.exists():
        raise FileNotFoundError(f"Dev not found: {dev_path}")

    df_train = pd.read_parquet(train_path)
    df_dev   = pd.read_parquet(dev_path)

    train(df_train, df_dev, device=args.device)


import sys

def is_notebook():
    try:
        from IPython import get_ipython
        return get_ipython() is not None
    except:
        return False


if __name__ == "__main__" and not is_notebook():
    cli_main()

##FOR KAGGLE RUN
# import pandas as pd

# df_train = pd.read_parquet("/kaggle/input/datasets/awatanshsingh/eta-zoned/train_features.parquet")
# df_dev   = pd.read_parquet("/kaggle/input/datasets/awatanshsingh/eta-zoned/dev_features.parquet")

# model = train(df_train, df_dev, device="gpu")