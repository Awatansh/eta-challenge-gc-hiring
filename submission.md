# Your Submission: Writeup

---

## Your final score

- Dev MAE: **280.53171 s**

## Your approach, in one paragraph

- I started from the parquet files produced by `data/download_data.py`, which turn the public NYC TLC 2023 yellow-taxi trips into `train.parquet` and `dev.parquet`.
- The starter baseline only used six raw features from that data: pickup zone, dropoff zone, hour, day of week, month, and passenger count.
- To build richer zone features, I used the `sim/fetch_zone_features.py` pipeline and the bundled taxi-zone reference data to derive boroughs, centroid coordinates, and special-zone flags from the public NYC zone lookup/shapefile.
- For inference, `predict` loads `solution/model.pkl` and `solution/zone_reference.json`, enriches each request with those geographic and time-based features, and returns a clipped positive duration prediction.
- I also added `check/compare_models.py` so I can compare baseline vs. my model on Dev data and see the improvement quickly.

- Final feature schema:
  `pickup_borough`, `pickup_zone_name`, `pickup_latitude`, `pickup_longitude`, `pickup_centroid_source`, `pickup_is_special_zone`, `dropoff_borough`, `dropoff_zone_name`, `dropoff_latitude`, `dropoff_longitude`, `dropoff_centroid_source`, `dropoff_is_special_zone`, `request_hour`, `request_dayofweek`, `request_month`, `is_weekend`, `is_rush_hour`, `same_borough`, `cross_borough`, `distance_km`, and `bearing_degrees`.

These features come from the public NYC taxi zone lookup/shapefile pipeline in `sim/fetch_zone_features.py`, which turns zone IDs into boroughs, centroid coordinates, special-zone flags, and pairwise distance/bearing features.

## What you tried that didn't work

- Pushing model complexity by itself did not help much; the biggest lift came from better feature engineering and stronger trip signals.
- Working directly with the parquet data and the packaged zone reference was more efficient than trying to do some extra preprocessing.
- I also hit some packaging/runtime issues while reorganizing the submission surface, which is why I consolidated the final artifacts under `solution/` and kept the root entrypoints as shims.

## Where AI tooling sped you up most

- Copilot helped most with the mechanical parts: wiring the submission layout, restoring the root entrypoints, and adding a comparison script for Dev MAE checks.
- It also saved time on debugging path/import issues, especially around making the packaged model load cleanly in both local and Docker runs.
- Where it fell short was judgment: I still had to make the model-selection calls, decide what to keep from EDA, confirm the model selection, and run the real timing/MAE checks myself, basically making concious decisions for improvements.

## Next experiments

- If I kept going, I would try a stronger model, make better use of GPU training, and experiment with clustering-based features for months so the model can capture seasonal patterns more cleanly.
- I would also spend more time on data preprocessing, because cleaner inputs and better feature cleanup often matter more than simply making the model deeper or more complex.
- Increasing model complexity could help, but it would also raise training time and compute cost a lot on data this size, so I would have to balance accuracy gains against runtime and resource usage.

## How to reproduce

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 1) Create the model artifact
python data/download_data.py
python sim/fetch_zone_features.py
python solution/train_model.py

# 2) Run the local benchmark and call predictions through the API
python check/compare_models.py --sample 2000000
python grade.py

# 3) Run predictions via Docker (Batch or API mode)
docker build -t eta-challenge-starter .
# Batch Mode:
MSYS_NO_PATHCONV=1 docker run --rm -v //c/Users/awatansh/Documents/Dev/arena/eta-challenge-starter/data:/data eta-challenge-starter /data/dev.parquet /data/preds.csv
# API Mode:
docker run --rm -p 8000:8000 eta-challenge-starter api
```

Sample record `curl -X POST localhost:8000/predict -H "Content-Type: application/json" -d '{"pickup_zone": 161, "dropoff_zone": 236, "requested_at": "2024-05-01 12:00:00", "passenger_count": 1}'`

If you want the full Dev comparison, use a sample number of at least 1,230,911.

### 1. Creation of `model.pkl`

```bash
python data/download_data.py
python sim/fetch_zone_features.py
python solution/train_model.py
```

This creates the root-level `model.pkl` artifact that the submission loads at inference time.

### 2. Running the benchmark and calling predictions through the API

```bash
python check/compare_models.py --sample 2000000
python grade.py
```

`check/compare_models.py` reports baseline vs. solution MAE on Dev, and `grade.py` exercises the same `predict(request)` API used by the grader.

### 3. Calling predictions through a Dockerized container (Batch & API)

```bash
docker build -t eta-challenge-starter .

# Option A: Batch Prediction Mode
MSYS_NO_PATHCONV=1 docker run --rm -v //c/Users/awatansh/Documents/Dev/arena/eta-challenge-starter/data:/data eta-challenge-starter /data/dev.parquet /data/preds.csv

# Option B: HTTP API Mode (FastAPI)
docker run --rm -p 8000:8000 eta-challenge-starter api
```

The Docker container now exposes a dual-purpose CLI wrapper (`solution/predict_cli.py`).

- **Batch mode** writes predictions directly to a CSV file and is used by the grader.
- **API mode** spins up a FastAPI server on port 8000 for 1-by-1 inference. You can test it via:
  `curl -X POST localhost:8000/predict -H "Content-Type: application/json" -d '{"pickup_zone": 161, "dropoff_zone": 236, "requested_at": "2024-05-01 12:00:00", "passenger_count": 1}'`

- Total time spent on this challenge: _about 6-7 hours._
