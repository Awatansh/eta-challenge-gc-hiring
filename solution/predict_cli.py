import sys
import numpy as np
import pandas as pd
from solution.eta_pipeline import predict

REQUEST_FIELDS = ["pickup_zone", "dropoff_zone", "requested_at", "passenger_count"]

def run_batch(input_path: str, output_path: str):
    print(f"Loading: {input_path}")
    df = pd.read_parquet(input_path)

    print("Predicting...")
    preds = np.empty(len(df), dtype=np.float64)
    # Ensure all request fields are present
    missing_cols = [col for col in REQUEST_FIELDS if col not in df.columns]
    if missing_cols:
        print(f"Error: Missing columns in input data: {missing_cols}")
        sys.exit(1)

    records = df[REQUEST_FIELDS].to_dict("records")
    for i, req in enumerate(records):
        preds[i] = predict(req)

    if "row_idx" in df.columns:
        row_idx = df["row_idx"].to_numpy()
    else:
        row_idx = np.arange(len(df), dtype=np.int64)

    df_out = pd.DataFrame({"row_idx": row_idx, "prediction": preds})
    df_out.to_csv(output_path, index=False)

    print(f"Saved: {output_path}")

def run_api():
    import uvicorn
    from fastapi import FastAPI
    from pydantic import BaseModel

    app = FastAPI(title="ETA Prediction API")

    class PredictRequest(BaseModel):
        pickup_zone: int
        dropoff_zone: int
        requested_at: str
        passenger_count: int

    @app.post("/predict")
    def predict_endpoint(req: PredictRequest):
        prediction = predict(req.model_dump())
        return {"prediction": prediction}

    print("Starting API on port 8000...")
    uvicorn.run(app, host="0.0.0.0", port=8000)

def main():
    if len(sys.argv) == 2 and sys.argv[1] == "api":
        run_api()
    elif len(sys.argv) == 3:
        run_batch(sys.argv[1], sys.argv[2])
    else:
        print("Usage:")
        print("  Batch mode: python predict_cli.py <input.parquet> <output.csv>")
        print("  API mode:   python predict_cli.py api")
        sys.exit(1)

if __name__ == "__main__":
    main()
