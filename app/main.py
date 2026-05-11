"""
DelayPredict — FastAPI Inference Service
----------------------------------------
Milestone 01: Load model artifact, separate training from inference
Milestone 02: Define input/output contract with Pydantic schemas
Learning Loop: Log predictions, detect drift, retrain, reload model

Start with:
    uvicorn app.main:app --host 0.0.0.0 --port 8000

Test with:
    curl http://localhost:8000/health
    curl -X POST http://localhost:8000/predict \
         -H "Content-Type: application/json" \
         -d '{"Airline":"WN","AirportFrom":"LAX","AirportTo":"SFO","DayOfWeek":3,"Length":60,"DepartureHour":8}'
"""

import sys
sys.path.append("/app/src")

import csv
import subprocess
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from predict import load_model, build_input, predict as run_predict

# ── Constants ─────────────────────────────────────────────────────────────────
MODEL_PATH    = "models/xgb_model.pkl"
PRED_LOG_PATH = Path("data/processed/predictions.csv")
DRIFT_THRESHOLD = 0.60   # retrain if accuracy drops below this

# ── Load model once at startup ────────────────────────────────────────────────
model = load_model()

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="DelayPredict",
    description="Binary flight delay prediction — AIOps SoSe 2026",
    version="1.0.0",
)

# ── Input schema ──────────────────────────────────────────────────────────────
class FlightInput(BaseModel):
    Airline: str = Field(..., description="IATA airline code (e.g. 'WN', 'AA', 'DL')", examples=["WN"])
    AirportFrom: str = Field(..., description="IATA departure airport code (e.g. 'LAX')", examples=["LAX"])
    AirportTo: str = Field(..., description="IATA arrival airport code (e.g. 'SFO')", examples=["SFO"])
    DayOfWeek: int = Field(..., ge=1, le=7, description="Day of week — 1=Monday, 7=Sunday", examples=[3])
    Length: int = Field(..., gt=0, description="Scheduled flight duration in minutes", examples=[60])
    DepartureHour: int = Field(..., ge=0, le=23, description="Scheduled departure hour (0–23)", examples=[8])

    @field_validator("Airline", "AirportFrom", "AirportTo")
    @classmethod
    def must_be_uppercase(cls, v: str) -> str:
        if not v.isupper():
            raise ValueError("Must be uppercase IATA code (e.g. 'WN', 'LAX')")
        return v

# ── Output schema ─────────────────────────────────────────────────────────────
class PredictionOutput(BaseModel):
    delay_predicted: bool = Field(..., description="True if the flight is predicted to be delayed")
    delay_probability: float = Field(..., description="Model confidence that the flight will be delayed (0.0–1.0)")
    model: str = Field(..., description="Model used for inference")

# ── Feedback schema ───────────────────────────────────────────────────────────
class FeedbackInput(BaseModel):
    prediction_id: str = Field(..., description="ID from the prediction log (timestamp)")
    actual_delay: int  = Field(..., ge=0, le=1, description="Actual outcome — 0=on time, 1=delayed")

# ── Helper: log prediction to CSV ─────────────────────────────────────────────
def log_prediction(flight: FlightInput, result: dict, prediction_id: str):
    """Append a prediction row to the predictions CSV."""
    PRED_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_header = not PRED_LOG_PATH.exists()

    with open(PRED_LOG_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "prediction_id", "timestamp", "Airline", "AirportFrom", "AirportTo",
            "DayOfWeek", "Length", "DepartureHour",
            "delay_predicted", "delay_probability", "actual_delay"
        ])
        if write_header:
            writer.writeheader()
        writer.writerow({
            "prediction_id"    : prediction_id,
            "timestamp"        : datetime.utcnow().isoformat(),
            "Airline"          : flight.Airline,
            "AirportFrom"      : flight.AirportFrom,
            "AirportTo"        : flight.AirportTo,
            "DayOfWeek"        : flight.DayOfWeek,
            "Length"           : flight.Length,
            "DepartureHour"    : flight.DepartureHour,
            "delay_predicted"  : int(result["delay_predicted"]),
            "delay_probability": result["delay_probability"],
            "actual_delay"     : "",   # filled in later via /feedback
        })

# ── Helper: check drift ────────────────────────────────────────────────────────
def check_drift() -> dict:
    """
    Compare delay_predicted vs actual_delay for rows where feedback exists.
    Returns drift status and current accuracy.
    """
    if not PRED_LOG_PATH.exists():
        return {"drift_detected": False, "reason": "No predictions logged yet"}

    import pandas as pd
    df = pd.read_csv(PRED_LOG_PATH)
    df_feedback = df[df["actual_delay"].notna()]

    if len(df_feedback) < 10:
        return {"drift_detected": False, "reason": f"Not enough feedback yet ({len(df_feedback)}/10)"}

    df_feedback = df_feedback.copy()
    df_feedback["actual_delay"]    = df_feedback["actual_delay"].astype(int)
    df_feedback["delay_predicted"] = df_feedback["delay_predicted"].astype(int)

    accuracy = float((df_feedback["delay_predicted"] == df_feedback["actual_delay"]).mean())
    drift    = bool(accuracy < DRIFT_THRESHOLD)

    return {
        "drift_detected"  : drift,
        "current_accuracy": round(accuracy, 4),
        "threshold"       : DRIFT_THRESHOLD,
        "feedback_count"  : int(len(df_feedback)),
    }

# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    """Check that the service and model are loaded correctly."""
    return {"status": "ok", "model": MODEL_PATH}


@app.post("/predict", response_model=PredictionOutput)
def predict_endpoint(flight: FlightInput):
    """Predict whether a flight will be delayed and log the prediction."""
    try:
        input_df = build_input(
            airline       =flight.Airline,
            airport_from  =flight.AirportFrom,
            airport_to    =flight.AirportTo,
            day_of_week   =flight.DayOfWeek,
            length        =flight.Length,
            departure_hour=flight.DepartureHour,
        )
        result = run_predict(model, input_df)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    # Log prediction
    prediction_id = datetime.utcnow().strftime("%Y%m%d%H%M%S%f")
    log_prediction(flight, result, prediction_id)

    return PredictionOutput(
        delay_predicted  =result["delay_predicted"],
        delay_probability=result["delay_probability"],
        model            =MODEL_PATH,
    )


@app.post("/feedback")
def feedback(fb: FeedbackInput):
    """Submit the actual outcome for a previous prediction."""
    if not PRED_LOG_PATH.exists():
        raise HTTPException(status_code=404, detail="No predictions logged yet.")

    import pandas as pd
    df = pd.read_csv(PRED_LOG_PATH, dtype=str)

    if fb.prediction_id not in df["prediction_id"].values:
        raise HTTPException(status_code=404, detail=f"Prediction ID '{fb.prediction_id}' not found.")

    df.loc[df["prediction_id"] == fb.prediction_id, "actual_delay"] = str(fb.actual_delay)
    df.to_csv(PRED_LOG_PATH, index=False)

    return {"status": "feedback recorded", "prediction_id": fb.prediction_id, "actual_delay": fb.actual_delay}


@app.get("/drift")
def drift():
    """Check if model performance has drifted below the threshold."""
    return check_drift()


@app.post("/retrain")
def retrain():
    """Trigger model retraining via src/train.py."""
    try:
        result = subprocess.run(
            ["python", "src/train.py"],
            capture_output=True, text=True, cwd="/app"
        )
        if result.returncode != 0:
            raise HTTPException(status_code=500, detail=result.stderr)
        return {"status": "retraining complete", "output": result.stdout}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reload-model")
def reload_model():
    """Reload the model from disk — call this after retraining."""
    global model
    model = load_model()
    return {"status": "model reloaded", "model": MODEL_PATH}