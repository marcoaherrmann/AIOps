"""
DelayPredict — FastAPI Inference Service
----------------------------------------
Milestone 01: Load model artifact, separate training from inference
Milestone 02: Define input/output contract with Pydantic schemas
Learning Loop: Stream new data, detect PSI drift, retrain, reload model

Start with:
    uvicorn app.main:app --host 0.0.0.0 --port 8000

Test with:
    curl http://localhost:8000/health
    curl http://localhost:8000/dashboard
"""

import sys
sys.path.append("/app/src")

import subprocess
import threading
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field, field_validator

from predict import load_model, build_input, predict as run_predict
from drift import check_drift_psi
from data_stream import stream_next_chunk, reset_stream, get_current_index, STREAM_POOL_PATH
from database import log_prediction, log_training_run, log_drift_scores

# ── Constants ─────────────────────────────────────────────────────────────────
MODEL_PATH   = "models/xgb_model.pkl"
BACKUP_PATH  = "models/xgb_model_backup.pkl"
STREAM_CHUNK = 53938  # ~10% of dataset per chunk (538383 * 0.1)
PRED_TRIGGER      = 10     # stream new chunk every N predictions
RETRAIN_LOG_PATH          = Path("data/processed/retrain_history.json")
INCREMENTAL_HISTORY_PATH  = Path("data/processed/incremental_history.json")

# ── State ─────────────────────────────────────────────────────────────────────
model           = load_model()
prediction_count = 0
incremental_training_running = False
retrain_running = False
is_rolled_back  = False   # True after /rollback, reset to False when a new retrain starts

# Load retrain history from disk if exists
import json
if Path("data/processed/retrain_history.json").exists():
    try:
        retrain_history = json.loads(Path("data/processed/retrain_history.json").read_text())
    except:
        retrain_history = []
else:
    retrain_history = []

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="DelayPredict",
    description="Binary flight delay prediction with PSI drift detection — AIOps SoSe 2026",
    version="2.0.0",
)

# ── Input schema ──────────────────────────────────────────────────────────────
class FlightInput(BaseModel):
    Airline: str = Field(..., description="IATA airline code (e.g. 'WN')", examples=["WN"])
    AirportFrom: str = Field(..., description="IATA departure airport", examples=["LAX"])
    AirportTo: str = Field(..., description="IATA arrival airport", examples=["SFO"])
    DayOfWeek: int = Field(..., ge=1, le=7, description="1=Monday, 7=Sunday", examples=[3])
    Length: int = Field(..., gt=0, description="Flight duration in minutes", examples=[60])
    DepartureHour: int = Field(..., ge=0, le=23, description="Departure hour (0-23)", examples=[8])

    @field_validator("Airline", "AirportFrom", "AirportTo")
    @classmethod
    def must_be_uppercase(cls, v: str) -> str:
        if not v.isupper():
            raise ValueError("Must be uppercase IATA code (e.g. 'WN', 'LAX')")
        return v

# ── Output schema ─────────────────────────────────────────────────────────────
class PredictionOutput(BaseModel):
    delay_predicted: bool
    delay_probability: float
    model: str

# ── Auto loop ─────────────────────────────────────────────────────────────────
def auto_loop():
    """
    Every PRED_TRIGGER predictions:
    1. Stream next chunk of data
    2. Check PSI drift
    3. If drift detected → retrain + reload
    """
    global model, prediction_count

    if prediction_count % PRED_TRIGGER != 0:
        return

    print(f"[Auto Loop] {prediction_count} predictions — streaming next chunk...")

    # ── Stream next chunk ──────────────────────────────────────────────────────
    result = stream_next_chunk(chunk_size=STREAM_CHUNK)

    if result.get("status") == "stream_complete":
        print("[Auto Loop] Stream complete — all data consumed.")
        return

    if "error" in result:
        print(f"[Auto Loop] Stream error: {result['error']}")
        return

    drift = result.get("drift", {})
    print(f"[Auto Loop] PSI drift check: {drift}")

    try:
        log_drift_scores(drift.get("psi_scores", {}))
    except Exception:
        pass

    # ── Retrain if drift detected ──────────────────────────────────────────────
    current_stream_pos = int(get_current_index())
    if drift.get("drift_detected"):
        print(f"[Auto Loop] Drift detected (PSI={drift.get('max_psi')}) — retraining (pos={current_stream_pos})...")
        try:
            proc = subprocess.run(
                ["python", "src/train.py", "--all-data"],
                capture_output=True, text=True, cwd="/app"
            )
            if proc.returncode == 0:
                model = load_model()
                # Evaluate new model and store metrics
                from evaluate import compute_metrics
                from data_preprocessing import load_data, get_splits, FEATURES, TARGET
                import pandas as pd
                from data_stream import STREAM_POOL_PATH
                df_pool = pd.read_csv(STREAM_POOL_PATH)
                X_test = df_pool[FEATURES]
                y_test = df_pool[TARGET]
                metrics = compute_metrics(model, X_test, y_test)

                retrain_history.append({
                    "timestamp"    : datetime.utcnow().isoformat(),
                    "total_streamed": result.get("total_streamed"),
                    "train_size"   : 269691 + current_stream_pos,
                    "max_psi"      : drift.get("max_psi"),
                    "worst_feature": drift.get("worst_feature"),
                    "psi_scores"   : drift.get("psi_scores"),
                    "roc_auc"      : round(metrics["ROC-AUC"], 4),
                    "f1"           : round(metrics["F1"], 4),
                    "accuracy"     : round(metrics["Accuracy"], 4),
                })
                print("[Auto Loop] Retrain complete — model reloaded.")
                # Save history to disk
                Path("data/processed/retrain_history.json").write_text(json.dumps(retrain_history))
                try:
                    log_training_run(
                        run_type     ="auto",
                        train_size   =269691 + current_stream_pos,
                        roc_auc      =metrics["ROC-AUC"],
                        f1           =metrics["F1"],
                        accuracy     =metrics["Accuracy"],
                        precision    =metrics["Precision"],
                        recall       =metrics["Recall"],
                        max_psi      =drift.get("max_psi"),
                        worst_feature=drift.get("worst_feature"),
                    )
                except Exception:
                    pass
            else:
                print(f"[Auto Loop] Retrain failed: {proc.stderr}")
        except Exception as e:
            print(f"[Auto Loop] Retrain error: {e}")
    else:
        print(f"[Auto Loop] No drift (PSI={drift.get('max_psi')}) — model stable.")

# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_PATH}


@app.post("/predict", response_model=PredictionOutput)
def predict_endpoint(flight: FlightInput):
    global prediction_count

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

    prediction_count += 1
    try:
        log_prediction(
            airline          =flight.Airline,
            airport_from     =flight.AirportFrom,
            airport_to       =flight.AirportTo,
            day_of_week      =flight.DayOfWeek,
            departure_hour   =flight.DepartureHour,
            length           =flight.Length,
            delay_predicted  =result["delay_predicted"],
            delay_probability=result["delay_probability"],
        )
    except Exception:
        pass
    auto_loop()

    return PredictionOutput(
        delay_predicted  =result["delay_predicted"],
        delay_probability=result["delay_probability"],
        model            =MODEL_PATH,
    )


@app.get("/drift")
def drift():
    """Check current PSI drift status against training reference."""
    from data_stream import STREAM_DATA_PATH
    import pandas as pd

    if not STREAM_DATA_PATH.exists():
        return {"drift_detected": False, "reason": "No stream data yet."}

    df_new = pd.read_csv(STREAM_DATA_PATH)
    return check_drift_psi(df_new)


def _run_progressive_retrain(rounds: int = 5):
    """Train in `rounds` steps on increasing fractions of available data.
    Each completed round is saved to retrain_history so the dashboard chart
    updates live, one point at a time. Each round is also logged to MLflow."""
    global model, retrain_history, retrain_running
    retrain_running = True
    try:
        import joblib
        import mlflow
        from train import build_pipeline
        from data_preprocessing import load_data, FEATURES, TARGET
        from evaluate import compute_metrics
        import pandas as pd

        df_full = load_data()
        df_base = df_full.sample(frac=0.5, random_state=42)
        df_test = df_full.drop(df_base.index)

        from drift import check_drift_psi

        stream_path = Path("data/processed/stream_data.csv")
        if stream_path.exists():
            df_stream = pd.read_csv(stream_path)
            df_pool   = pd.concat([df_base, df_stream], ignore_index=True)
        else:
            df_stream = None
            df_pool   = df_base

        X_test = df_test[FEATURES]
        y_test = df_test[TARGET]

        # Compute PSI once against the training reference
        drift_snap   = check_drift_psi(df_stream) if df_stream is not None else {}
        snap_psi     = drift_snap.get("psi_scores", {})
        snap_max_psi = drift_snap.get("max_psi")
        snap_worst   = drift_snap.get("worst_feature", "—")

        # Backup current model before first round
        if Path(MODEL_PATH).exists():
            import shutil
            shutil.copy(MODEL_PATH, BACKUP_PATH)

        # MLflow setup — one run per round under the DelayPredict experiment
        mlflow.set_tracking_uri("file:///app/notebooks/mlruns")
        mlflow.set_experiment("DelayPredict")

        for i in range(1, rounds + 1):
            df_train = df_pool.sample(frac=i / rounds, random_state=42)
            pipeline  = build_pipeline()
            pipeline.fit(df_train[FEATURES], df_train[TARGET])
            metrics   = compute_metrics(pipeline, X_test, y_test)

            retrain_history.append({
                "timestamp"    : datetime.utcnow().isoformat(),
                "total_streamed": len(df_pool) - len(df_base),
                "train_size"   : len(df_train),
                "max_psi"      : snap_max_psi,
                "worst_feature": snap_worst,
                "psi_scores"   : snap_psi,
                "roc_auc"      : round(metrics["ROC-AUC"], 4),
                "f1"           : round(metrics["F1"], 4),
                "accuracy"     : round(metrics["Accuracy"], 4),
            })
            Path("data/processed/retrain_history.json").write_text(json.dumps(retrain_history))
            print(f"[Retrain] Round {i}/{rounds} done — train_size={len(df_train):,}")
            try:
                log_training_run(
                    run_type     ="progressive",
                    round        =i,
                    train_size   =len(df_train),
                    roc_auc      =metrics["ROC-AUC"],
                    f1           =metrics["F1"],
                    accuracy     =metrics["Accuracy"],
                    precision    =metrics["Precision"],
                    recall       =metrics["Recall"],
                    max_psi      =snap_max_psi,
                    worst_feature=snap_worst,
                )
            except Exception:
                pass

            # Log each round as its own MLflow run
            with mlflow.start_run(run_name=f"XGBoost_progressive_r{i}of{rounds}_{len(df_train)}rows"):
                mlflow.log_params({
                    "round"        : i,
                    "total_rounds" : rounds,
                    "train_size"   : len(df_train),
                    "stream_rows"  : len(df_pool) - len(df_base),
                    "max_psi"      : snap_max_psi,
                    "worst_feature": snap_worst,
                })
                mlflow.log_metrics({
                    "accuracy" : metrics["Accuracy"],
                    "precision": metrics["Precision"],
                    "recall"   : metrics["Recall"],
                    "f1"       : metrics["F1"],
                    "roc_auc"  : metrics["ROC-AUC"],
                })

            if i == rounds:
                joblib.dump(pipeline, MODEL_PATH)
                model = load_model()

    except Exception as e:
        print(f"[Retrain] Error: {e}")
    finally:
        retrain_running = False


@app.post("/retrain")
def retrain():
    """Progressive retrain: 5 rounds on increasing data fractions, live chart updates."""
    global retrain_running, retrain_history, is_rolled_back
    if retrain_running:
        return {"status": "already_running"}
    retrain_history.clear()
    Path("data/processed/retrain_history.json").write_text(json.dumps([]))
    is_rolled_back = False   # new model replaces the rolled-back one
    threading.Thread(target=_run_progressive_retrain, daemon=True).start()
    return {"status": "started"}


@app.post("/reload-model")
def reload_model():
    """Reload model from disk."""
    global model
    model = load_model()
    return {"status": "model reloaded", "model": MODEL_PATH}


@app.post("/rollback")
def rollback():
    """Rollback to previous model version."""
    global model, is_rolled_back
    import shutil
    if not Path(BACKUP_PATH).exists():
        raise HTTPException(status_code=404, detail="No backup model found.")
    shutil.copy(BACKUP_PATH, MODEL_PATH)
    model = load_model()
    is_rolled_back = True   # show indicator in dashboard until next retrain
    return {"status": "rolled back to previous model"}


@app.post("/stream/next")
def stream_next(chunk_size: int = STREAM_CHUNK):
    """Manually trigger next data stream chunk."""
    result = stream_next_chunk(chunk_size=chunk_size)
    return result


@app.post("/stream/reset")
def stream_reset():
    """Reset stream to beginning — useful for demo."""
    reset_stream()
    return {"status": "stream reset"}


@app.get("/status")
def status():
    """Full system status for dashboard."""
    import pandas as pd

    stream_index = get_current_index()
    pool_size = len(pd.read_csv(STREAM_POOL_PATH)) if STREAM_POOL_PATH.exists() else 0

    drift_result = {}
    from data_stream import STREAM_DATA_PATH
    if STREAM_DATA_PATH.exists():
        df_new = pd.read_csv(STREAM_DATA_PATH)
        drift_result = check_drift_psi(df_new)

    return {
        "predictions_made"  : prediction_count,
        "stream_consumed"   : stream_index,
        "stream_remaining"  : max(0, pool_size - stream_index),
        "stream_total"      : pool_size,
        "drift"             : drift_result,
        "retrain_count"     : len(retrain_history),
        "retrain_history"   : retrain_history[-5:],
        "retrain_running"   : retrain_running,
        "is_rolled_back"    : is_rolled_back,
        "model"             : MODEL_PATH,
        "backup_available"  : Path(BACKUP_PATH).exists(),
    }


@app.post("/incremental-training")
def start_incremental_training():
    """Start 10-round incremental training in the background."""
    global incremental_training_running
    if incremental_training_running:
        rounds_done = 0
        if INCREMENTAL_HISTORY_PATH.exists():
            try:
                rounds_done = len(json.loads(INCREMENTAL_HISTORY_PATH.read_text()))
            except Exception:
                pass
        return {"status": "already_running", "rounds_done": rounds_done}

    def run():
        global incremental_training_running
        incremental_training_running = True
        try:
            from train import train_incremental
            train_incremental()
        except Exception as e:
            print(f"[Incremental] Error: {e}")
        finally:
            incremental_training_running = False

    threading.Thread(target=run, daemon=True).start()
    return {"status": "started"}


@app.get("/incremental-status")
def incremental_status():
    """Return current incremental training state and completed round results."""
    history = []
    if INCREMENTAL_HISTORY_PATH.exists():
        try:
            history = json.loads(INCREMENTAL_HISTORY_PATH.read_text())
        except Exception:
            pass
    return {
        "running"     : incremental_training_running,
        "rounds_done" : len(history),
        "total_rounds": 10,
        "history"     : history,
    }


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard():
    """Live dashboard showing loop status, drift, and model metrics."""
    return """
<!DOCTYPE html>
<html>
<head>
    <title>DelayPredict - Learning Loop Dashboard</title>
    <meta charset="utf-8">
    <link rel="icon" href="data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 100 100%22><text y=%22.9em%22 font-size=%2290%22>✈️</text></svg>">
    
    <style>
        body { font-family: Arial, sans-serif; background: #0f172a; color: #e2e8f0; margin: 0; padding: 24px; }
        h1 { color: #38bdf8; margin-bottom: 4px; }
        .subtitle { color: #94a3b8; margin-bottom: 32px; font-size: 14px; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; margin-bottom: 24px; }
        .card { background: #1e293b; border-radius: 12px; padding: 20px; border: 1px solid #334155; }
        .card h3 { margin: 0 0 16px 0; color: #94a3b8; font-size: 13px; text-transform: uppercase; letter-spacing: 1px; }
        .metric { font-size: 32px; font-weight: bold; color: #38bdf8; }
        .label { font-size: 13px; color: #64748b; margin-top: 4px; }
        .badge { display: inline-block; padding: 4px 12px; border-radius: 20px; font-size: 13px; font-weight: bold; }
        .badge-green { background: #064e3b; color: #34d399; }
        .badge-red { background: #450a0a; color: #f87171; }
        .badge-yellow { background: #451a03; color: #fbbf24; }
        .badge-orange { background: #431407; color: #fb923c; }
        .psi-row { display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid #334155; }
        .psi-bar { height: 8px; background: #334155; border-radius: 4px; margin-top: 4px; }
        .psi-fill { height: 8px; border-radius: 4px; transition: width 0.5s; }
        table { width: 100%; border-collapse: collapse; font-size: 13px; }
        th { text-align: left; padding: 8px; color: #64748b; border-bottom: 1px solid #334155; }
        td { padding: 8px; border-bottom: 1px solid #1e293b; }
        .footer { color: #475569; font-size: 12px; margin-top: 16px; }
    </style>
</head>
<body>
    <h1>✈️ DelayPredict — Learning Loop Dashboard</h1>
    <p class="subtitle">Auto-refreshes every 5 seconds</p>

    <div class="card" style="margin-bottom:24px">
        <h3>Try a Prediction</h3>
        <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:12px;margin-bottom:16px">
            <div>
                <label class="form-label">Airline</label>
                <input id="p-airline" class="form-input" type="text" placeholder="e.g. WN" maxlength="3">
            </div>
            <div>
                <label class="form-label">Airport From</label>
                <input id="p-from" class="form-input" type="text" placeholder="e.g. LAX" maxlength="3">
            </div>
            <div>
                <label class="form-label">Airport To</label>
                <input id="p-to" class="form-input" type="text" placeholder="e.g. JFK" maxlength="3">
            </div>
            <div>
                <label class="form-label">Day of Week</label>
                <select id="p-dow" class="form-input">
                    <option value="1">Monday</option>
                    <option value="2">Tuesday</option>
                    <option value="3">Wednesday</option>
                    <option value="4">Thursday</option>
                    <option value="5" selected>Friday</option>
                    <option value="6">Saturday</option>
                    <option value="7">Sunday</option>
                </select>
            </div>
            <div>
                <label class="form-label">Departure Hour</label>
                <input id="p-hour" class="form-input" type="number" min="0" max="23" value="8">
            </div>
            <div>
                <label class="form-label">Duration (min)</label>
                <input id="p-length" class="form-input" type="number" min="1" value="120">
            </div>
        </div>
        <div style="display:flex;align-items:center;gap:16px;flex-wrap:wrap">
            <button id="pred-btn" class="btn-primary" onclick="submitPrediction()">Predict Delay</button>
            <div id="pred-result"></div>
        </div>
    </div>

    <div id="content">Loading...</div>

    <style>
        .btn-primary { background:#1d4ed8;color:#fff;border:none;padding:10px 20px;border-radius:8px;cursor:pointer;font-size:14px;font-weight:bold; }
        .btn-primary:disabled { background:#334155;color:#64748b;cursor:not-allowed; }
        .btn-primary:hover:not(:disabled) { background:#2563eb; }
        .form-input { background:#0f172a;border:1px solid #334155;color:#e2e8f0;padding:8px 12px;border-radius:6px;width:100%;font-size:14px;box-sizing:border-box; }
        .form-input:focus { outline:none;border-color:#38bdf8; }
        .form-label { font-size:12px;color:#64748b;display:block;margin-bottom:4px; }
    </style>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script>
    async function submitPrediction() {
        const btn = document.getElementById('pred-btn');
        btn.disabled = true;
        btn.textContent = 'Predicting...';
        document.getElementById('pred-result').innerHTML = '';

        const body = {
            Airline      : document.getElementById('p-airline').value.toUpperCase().trim(),
            AirportFrom  : document.getElementById('p-from').value.toUpperCase().trim(),
            AirportTo    : document.getElementById('p-to').value.toUpperCase().trim(),
            DayOfWeek    : parseInt(document.getElementById('p-dow').value),
            DepartureHour: parseInt(document.getElementById('p-hour').value),
            Length       : parseInt(document.getElementById('p-length').value),
        };

        try {
            const res  = await fetch('/predict', {
                method : 'POST',
                headers: {'Content-Type': 'application/json'},
                body   : JSON.stringify(body),
            });
            const data = await res.json();
            if (res.ok) {
                const delayed = data.delay_predicted;
                const prob    = (data.delay_probability * 100).toFixed(1);
                const color   = delayed ? '#f87171' : '#34d399';
                const bg      = delayed ? '#450a0a' : '#064e3b';
                const icon    = delayed ? '⚠️' : '✅';
                const label   = delayed ? 'Delay likely' : 'On time likely';
                document.getElementById('pred-result').innerHTML =
                    `<div style="background:${bg};border:1px solid ${color};color:${color};padding:10px 16px;border-radius:8px;font-weight:bold;font-size:15px">
                        ${icon} ${label} &nbsp;—&nbsp; ${prob}% probability
                    </div>`;
            } else {
                document.getElementById('pred-result').innerHTML =
                    `<div style="color:#f87171">Error: ${data.detail || 'Unknown error'}</div>`;
            }
        } catch(e) {
            document.getElementById('pred-result').innerHTML =
                `<div style="color:#f87171">Request failed</div>`;
        }

        btn.disabled = false;
        btn.textContent = 'Predict Delay';
    }

    async function startIncremental() {
        const btn = document.getElementById('inc-btn');
        btn.disabled = true;
        btn.textContent = 'Starting...';
        await fetch('/incremental-training', {method: 'POST'});
    }

    window._rollingBack  = window._rollingBack  || false;
    window._notification = window._notification || null;

    function triggerRetrain() {
        fetch('/retrain', {method: 'POST'});   // fire-and-forget; d.retrain_running tracks state
    }

    async function triggerRollback() {
        if (window._rollingBack) return;
        window._rollingBack = true;
        await fetch('/rollback', {method: 'POST'});
        window._rollingBack = false;
        window._notification = { msg: '↩ Rolled back to previous model', color: '#38bdf8', bg: '#0c2a4a', time: Date.now() };
    }

    async function load() {
        const [res, incRes] = await Promise.all([fetch('/status'), fetch('/incremental-status')]);
        const d   = await res.json();
        const inc = await incRes.json();
        const drift = d.drift || {};
        const psi = drift.psi_scores || {};

        const driftBadge = drift.drift_detected
            ? '<span class="badge badge-red">⚠ DRIFT DETECTED</span>'
            : '<span class="badge badge-green">✓ STABLE</span>';

        const streamPct = d.stream_total > 0
            ? Math.round((d.stream_consumed / d.stream_total) * 100)
            : 0;

        const RETRAIN_ROUNDS    = 5;
        const TOTAL_ROWS        = 539383;
        const lastRound         = d.retrain_history.length > 0 ? d.retrain_history[d.retrain_history.length - 1] : null;
        const isManualRetrain   = d.retrain_running || (lastRound && lastRound.worst_feature !== 'manual' ? false : !!lastRound);
        const currentTrainSize  = lastRound ? lastRound.train_size : 0;
        const displayPct        = isManualRetrain
            ? Math.round((currentTrainSize / TOTAL_ROWS) * 100)
            : streamPct;
        const displayTitle      = isManualRetrain ? 'Training Progress' : 'Data Stream';
        const roundLabel        = d.retrain_running
            ? ` — Round ${d.retrain_count}/${RETRAIN_ROUNDS}`
            : '';
        const displayLabel      = isManualRetrain
            ? `${currentTrainSize.toLocaleString()} / ${TOTAL_ROWS.toLocaleString()} rows trained${roundLabel}`
            : `${d.stream_consumed.toLocaleString()} / ${d.stream_total.toLocaleString()} rows consumed`;

        const incStatusBadge = inc.running
            ? `<span class="badge badge-yellow">⏳ Running... Round ${inc.rounds_done}/${inc.total_rounds}</span>`
            : inc.rounds_done === 10
                ? `<span class="badge badge-green">✓ Complete — all 10 rounds done</span>`
                : inc.rounds_done > 0
                    ? `<span class="badge badge-yellow">${inc.rounds_done}/10 rounds available</span>`
                    : '<span style="color:#64748b">Not started yet</span>';

        const n = window._notification;
        const notif = n && (Date.now() - n.time) < 6000
            ? `<div style="background:${n.bg};border:1px solid ${n.color};color:${n.color};padding:10px 18px;border-radius:8px;margin-bottom:16px;font-weight:bold;font-size:14px">${n.msg}</div>`
            : '';

        // Persistent rollback banner — stays visible until next retrain
        const rollbackBanner = d.is_rolled_back
            ? `<div style="background:#431407;border:1px solid #fb923c;color:#fb923c;padding:12px 18px;border-radius:8px;margin-bottom:16px;font-size:14px">
                   <strong>↩ Rollback active</strong> — the model was rolled back to the previous version. Run a new retrain to replace it.
               </div>`
            : '';

        const lastRetrain = d.retrain_history.length > 0 ? d.retrain_history[d.retrain_history.length - 1] : null;
        const currentRocAuc = lastRetrain ? lastRetrain.roc_auc.toFixed(4) : 'N/A';

        let psiRows = '';
        for (const [feat, val] of Object.entries(psi)) {
            const color = val > 0.2 ? '#f87171' : val > 0.1 ? '#fbbf24' : '#34d399';
            const pct = Math.min(Math.round((val / 0.3) * 100), 100);
            psiRows += `
                <div class="psi-row">
                    <div>
                        <div>${feat}</div>
                        <div class="psi-bar"><div class="psi-fill" style="width:${pct}%;background:${color}"></div></div>
                    </div>
                    <div style="color:${color};font-weight:bold">${val}</div>
                </div>`;
        }

        let retrainRows = '';
        for (const r of (d.retrain_history || []).reverse()) {
            const psiColor = r.max_psi > 0.2 ? '#f87171' : r.max_psi > 0.1 ? '#fbbf24' : '#34d399';
            retrainRows += `<tr>
                <td>${r.timestamp.substring(0,19)}</td>
                <td>${r.train_size?.toLocaleString() ?? r.total_streamed?.toLocaleString() ?? '-'}</td>
                <td>${r.worst_feature ?? '-'}</td>
                <td style="color:${r.max_psi ? psiColor : '#64748b'}">${r.max_psi ?? '-'}</td>
            </tr>`;
        }

        document.getElementById('content').innerHTML = `
        ${notif}
        ${rollbackBanner}
        <div class="grid">
            <div class="card">
                <h3>Model ROC-AUC</h3>
                <div class="metric">${currentRocAuc}</div>
                <div class="label">${d.is_rolled_back ? '⚠ from last retrain — rollback active' : (lastRetrain ? 'current model performance (after last retrain)' : 'current model performance — no retrain yet')}</div>
            </div>
            <div class="card">
                <h3>${displayTitle}</h3>
                <div class="metric">${displayPct}%</div>
                <div class="label">${displayLabel}</div>
            </div>
            <div class="card">
                <h3>Drift Status</h3>
                <div style="margin-bottom:12px">${driftBadge}</div>
                <div class="label">Max PSI: <strong>${drift.max_psi ?? 'N/A'}</strong> (threshold: ${drift.psi_threshold ?? 0.2})</div>
                <div class="label">Worst feature: <strong>${drift.worst_feature ?? 'N/A'}</strong></div>
            </div>
            <div class="card">
                <h3>Model</h3>
                <div class="metric">${d.retrain_count}</div>
                <div class="label">times retrained</div>
                <div style="margin-top:12px">
                    ${d.is_rolled_back
                        ? '<span class="badge badge-orange" style="margin-bottom:10px;display:inline-block">↩ ROLLBACK ACTIVE</span>'
                        : ''}
                    ${d.backup_available
                        ? `<div><button id="rollback-btn" class="btn-primary" onclick="triggerRollback()" style="font-size:12px;padding:6px 14px" ${window._rollingBack ? 'disabled' : ''}>
                               ${window._rollingBack ? 'Rolling back...' : '↩ Rollback to Previous'}
                           </button></div>`
                        : '<span class="badge badge-yellow">No backup yet</span>'}
                </div>
            </div>
        </div>

        <div class="grid">
            <div class="card">
                <h3>PSI per Feature</h3>
                ${psiRows || '<div style="color:#64748b">No stream data yet</div>'}
                <div style="margin-top:12px;font-size:12px;color:#64748b">
                    🟢 &lt; 0.1 stable &nbsp; 🟡 0.1-0.2 monitor &nbsp; 🔴 &gt; 0.2 retrain
                </div>
            </div>
            <div class="card">
                <h3>Retrain History</h3>
                ${retrainRows ? `<table>
                    <tr><th>Time</th><th>Train Rows</th><th>Feature</th><th>PSI</th></tr>
                    ${retrainRows}
                </table>` : '<div style="color:#64748b">No retrains yet</div>'}
            </div>
        </div>

        <div class="card" style="margin-bottom:24px">
            <h3>Model Performance over Training Size — Shows how model quality improves as more training data is added</h3>
            <div style="margin-bottom:16px">
                <button id="retrain-btn" class="btn-primary" onclick="triggerRetrain()" ${d.retrain_running ? 'disabled' : ''}>
                    ${d.retrain_running ? `Retraining... (${d.retrain_count}/5)` : 'Retrain on Stream Data'}
                </button>
            </div>
            <canvas id="metricsChart" height="80"></canvas>
            ${d.retrain_history.length === 0 ? '<div style="color:#64748b;margin-top:12px">No retrains yet — stream data first, then click Retrain</div>' : ''}
        </div>
        <div class="card" style="margin-bottom:24px">
            <h3>Incremental Training — Learning Curve (90/10 Split, 10 Rounds)</h3>
            <div style="display:flex;align-items:center;gap:16px;margin-bottom:16px;flex-wrap:wrap">
                <button id="inc-btn" class="btn-primary" onclick="startIncremental()" ${inc.running ? 'disabled' : ''}>
                    ${inc.running ? 'Training in progress...' : 'Start Incremental Training'}
                </button>
                <div>${incStatusBadge}</div>
            </div>
            ${inc.running && inc.rounds_done === 0
                ? '<div style="color:#fbbf24;padding:32px 0;text-align:center;font-size:15px">⏳ Starting training — waiting for Round 1 to complete...</div>'
                : `<canvas id="incChart" height="80"></canvas>
                   ${inc.history.length === 0 ? '<div style="color:#64748b;margin-top:12px">No data yet — click the button to run 10 cumulative training rounds and see the learning curve</div>' : ''}`}
        </div>
        <p class="footer">Model: ${d.model} &nbsp;|&nbsp; AIOps SoSe 2026 &nbsp;|&nbsp; <a href="http://localhost:8050" style="color:#38bdf8">📊 Data Analytics</a> &nbsp;|&nbsp; <a href="http://localhost:3000" style="color:#38bdf8">🗃️ Metabase</a> &nbsp;|&nbsp; <a href="http://localhost:5001" style="color:#38bdf8">MLflow</a></p>`;

    // ── Chart ─────────────────────────────────────────────────────────────────
    if (d.retrain_history && d.retrain_history.length > 0) {
        // Sort ascending by train_size so X-axis goes small → large
        const sorted = [...d.retrain_history].sort((a, b) => (a.train_size || 0) - (b.train_size || 0));
        const labels  = sorted.map(r => ((r.train_size || 0) / 1000).toFixed(0) + 'k rows');
        const rocData = sorted.map(r => r.roc_auc || 0);
        const f1Data  = sorted.map(r => r.f1 || 0);
        const accData = sorted.map(r => r.accuracy || 0);

        if (window._chart) window._chart.destroy();
        const ctx = document.getElementById('metricsChart').getContext('2d');
        window._chart = new Chart(ctx, {
            type: 'line',
            data: {
                labels,
                datasets: [
                    { label: 'ROC-AUC', data: rocData, borderColor: '#38bdf8', backgroundColor: 'rgba(56,189,248,0.1)', tension: 0.3, fill: true },
                    { label: 'F1',      data: f1Data,  borderColor: '#34d399', backgroundColor: 'rgba(52,211,153,0.1)', tension: 0.3, fill: true },
                    { label: 'Accuracy',data: accData, borderColor: '#fbbf24', backgroundColor: 'rgba(251,191,36,0.1)',  tension: 0.3, fill: true },
                ]
            },
            options: {
                plugins: { legend: { labels: { color: '#e2e8f0' } } },
                scales: {
                    x: { ticks: { color: '#94a3b8' }, grid: { color: '#334155' } },
                    y: { min: 0.5, max: 1.0, ticks: { color: '#94a3b8' }, grid: { color: '#334155' } }
                }
            }
        });
    }
    // ── Incremental Training Chart ─────────────────────────────────────────────
    if (inc.history && inc.history.length > 0 && !(inc.running && inc.rounds_done === 0)) {
        const labels  = inc.history.map(r => `Round ${r.round}\n(${(r.train_size/1000).toFixed(0)}k rows)`);
        const rocData = inc.history.map(r => r.roc_auc);
        const f1Data  = inc.history.map(r => r.f1);
        const accData = inc.history.map(r => r.accuracy);

        if (window._incChart) window._incChart.destroy();
        const ctx2 = document.getElementById('incChart').getContext('2d');
        window._incChart = new Chart(ctx2, {
            type: 'line',
            data: {
                labels,
                datasets: [
                    { label: 'ROC-AUC', data: rocData, borderColor: '#38bdf8', backgroundColor: 'rgba(56,189,248,0.1)', tension: 0.3, fill: true },
                    { label: 'F1',      data: f1Data,  borderColor: '#34d399', backgroundColor: 'rgba(52,211,153,0.1)', tension: 0.3, fill: true },
                    { label: 'Accuracy',data: accData, borderColor: '#fbbf24', backgroundColor: 'rgba(251,191,36,0.1)',  tension: 0.3, fill: true },
                ]
            },
            options: {
                plugins: { legend: { labels: { color: '#e2e8f0' } } },
                scales: {
                    x: { ticks: { color: '#94a3b8' }, grid: { color: '#334155' } },
                    y: { min: 0.5, max: 1.0, ticks: { color: '#94a3b8' }, grid: { color: '#334155' } }
                }
            }
        });
    }
    }

    load();
    setInterval(() => {
        const y = window.scrollY;
        load().then(() => window.scrollTo(0, y));
    }, 5000);
    </script>
</body>
</html>
"""