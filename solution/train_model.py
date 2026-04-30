#!/usr/bin/env python3
"""
Simplified modular trainer for enriched parquet data.

Local usage:
    python solution/train_model.py --train-parquet sim/zone_outputs/train_features.parquet --dev-parquet sim/zone_outputs/dev_features.parquet

Kaggle drop-in usage (paste this into a notebook cell):
    import pandas as pd
    import json, pickle, subprocess
    import lightgbm as lgb
    from sklearn.metrics import mean_absolute_error
    
    TARGET = 'duration_seconds'
    CAT_COLS = ['pickup_borough', 'dropoff_borough']
    
    df_train = pd.read_parquet('/kaggle/input/.../train_features.parquet')
    df_dev = pd.read_parquet('/kaggle/input/.../dev_features.parquet')
    
    # Train and save (see train() function below)
    train(df_train, df_dev)
"""

from __future__ import annotations

import argparse
import json
import pickle
import subprocess
from pathlib import Path

import lightgbm as lgb
import pandas as pd

# =============================================================================
# CONSTANTS
# =============================================================================
TARGET = "duration_seconds"
CAT_COLS = ["pickup_borough", "dropoff_borough"]

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "model.pkl"
DEFAULT_METADATA_PATH = Path(__file__).resolve().parent / "model_metadata.json"


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def gpu_available() -> bool:
    """Check if GPU is available."""
    try:
        return subprocess.run(["nvidia-smi"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
    except Exception:
        return False


def prepare_data(
    df_train: pd.DataFrame, df_dev: pd.DataFrame, max_rows: int | None = None
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Extract features and target, align categories, downcast dtypes.
    Only includes numeric and categorical features; drops unused string columns.
    """
    if max_rows is not None:
        df_train = df_train.head(max_rows).reset_index(drop=True)
        df_dev = df_dev.head(max_rows).reset_index(drop=True)

    # Exclude target and non-useful string columns
    drop_set = {TARGET, "requested_at", "pickup_zone_name", "dropoff_zone_name", "pickup_centroid_source", "dropoff_centroid_source"}
    features = [c for c in df_train.columns if c not in drop_set]
    
    X_train = df_train[features].copy()
    y_train = df_train[TARGET].copy()
    X_dev = df_dev[features].copy()
    y_dev = df_dev[TARGET].copy()

    # Categorical alignment (critical for LightGBM)
    for col in CAT_COLS:
        if col in X_train.columns:
            X_train[col] = pd.Categorical(X_train[col])
            X_dev[col] = pd.Categorical(X_dev[col], categories=X_train[col].cat.categories)

    # Downcast for memory efficiency
    for col in X_train.select_dtypes(include=["float64"]).columns:
        X_train[col] = X_train[col].astype("float32")
    for col in X_train.select_dtypes(include=["int64"]).columns:
        X_train[col] = X_train[col].astype("int32")

    for col in X_dev.select_dtypes(include=["float64"]).columns:
        X_dev[col] = X_dev[col].astype("float32")
    for col in X_dev.select_dtypes(include=["int64"]).columns:
        X_dev[col] = X_dev[col].astype("int32")

    return X_train, y_train, X_dev, y_dev


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_dev: pd.DataFrame,
    y_dev: pd.Series,
    device: str = "auto",
) -> lgb.Booster:
    """Train LightGBM model."""
    if device == "auto":
        device = "gpu" if gpu_available() else "cpu"

    params = {
        "objective": "regression",
        "metric": "mae",
        "device": device,
        "max_bin": 255,
        "learning_rate": 0.05,
        "num_leaves": 256,
        "min_data_in_leaf": 100,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbosity": -1,
        "seed": 42,
    }

    train_data = lgb.Dataset(
        X_train, label=y_train, categorical_feature=CAT_COLS, free_raw_data=True
    )
    valid_data = lgb.Dataset(X_dev, label=y_dev, reference=train_data, free_raw_data=True)

    model = lgb.train(
        params,
        train_data,
        num_boost_round=3000,
        valid_sets=[valid_data],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)],
    )
    return model


def save_artifacts(
    model: lgb.Booster,
    X_train: pd.DataFrame,
    model_path: Path | str = DEFAULT_MODEL_PATH,
    metadata_path: Path | str = DEFAULT_METADATA_PATH,
) -> None:
    """Save model and metadata."""
    model_path = Path(model_path)
    metadata_path = Path(metadata_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    # Metadata for inference alignment
    feature_names = list(X_train.columns)
    pickup_cats = list(X_train["pickup_borough"].cat.categories) if "pickup_borough" in X_train.columns else []
    dropoff_cats = list(X_train["dropoff_borough"].cat.categories) if "dropoff_borough" in X_train.columns else []

    meta = {
        "feature_names": feature_names,
        "pickup_borough_categories": pickup_cats,
        "dropoff_borough_categories": dropoff_cats,
        "pickup_unknown_idx": pickup_cats.index("Unknown") if "Unknown" in pickup_cats else 0,
        "dropoff_unknown_idx": dropoff_cats.index("Unknown") if "Unknown" in dropoff_cats else 0,
    }
    metadata_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"✅ Saved model to {model_path}")
    print(f"✅ Saved metadata to {metadata_path}")


# =============================================================================
# MAIN TRAINING FUNCTION
# =============================================================================
def train(
    df_train: pd.DataFrame,
    df_dev: pd.DataFrame,
    device: str = "auto",
    model_path: Path | str = DEFAULT_MODEL_PATH,
    metadata_path: Path | str = DEFAULT_METADATA_PATH,
    max_rows: int | None = None,
) -> lgb.Booster:
    """
    End-to-end training: prepare, train, save artifacts.
    
    Args:
        df_train: Training DataFrame (enriched with zone features)
        df_dev: Dev DataFrame (enriched with zone features)
        device: 'auto' (default), 'gpu', or 'cpu'
        model_path: Where to save model.pkl
        metadata_path: Where to save model_metadata.json
        max_rows: Optional row cap for quick tests
    
    Returns:
        Trained LightGBM booster
    """
    print(f"Preparing data...")
    X_train, y_train, X_dev, y_dev = prepare_data(df_train, df_dev, max_rows=max_rows)
    print(f"  Train: {len(X_train):,} rows × {len(X_train.columns)} features")
    print(f"  Dev:   {len(X_dev):,} rows × {len(X_dev.columns)} features")

    print(f"\nTraining on {(device if device != 'auto' else 'auto').upper()}...")
    model = train_model(X_train, y_train, X_dev, y_dev, device=device)

    print(f"\nSaving artifacts...")
    save_artifacts(model, X_train, model_path, metadata_path)

    return model


# =============================================================================
# CLI FOR LOCAL RUNS
# =============================================================================
def cli_main() -> None:
    """Local runner with argparse."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-parquet", type=Path, default=Path("sim/zone_outputs/train_features.parquet"))
    parser.add_argument("--dev-parquet", type=Path, default=Path("sim/zone_outputs/dev_features.parquet"))
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--metadata-path", type=Path, default=DEFAULT_METADATA_PATH)
    parser.add_argument("--device", choices=["auto", "gpu", "cpu"], default="auto")
    parser.add_argument("--max-rows", type=int, default=None)
    args = parser.parse_args()

    if not args.train_parquet.exists() or not args.dev_parquet.exists():
        raise SystemExit(f"Parquet files not found: {args.train_parquet} or {args.dev_parquet}")

    print(f"Loading {args.train_parquet}...")
    df_train = pd.read_parquet(args.train_parquet)
    print(f"Loading {args.dev_parquet}...")
    df_dev = pd.read_parquet(args.dev_parquet)

    train(
        df_train,
        df_dev,
        device=args.device,
        model_path=args.model_path,
        metadata_path=args.metadata_path,
        max_rows=args.max_rows,
    )


if __name__ == "__main__":
    cli_main()
