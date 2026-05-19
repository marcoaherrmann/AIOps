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
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field, field_validator

from predict import load_model, build_input, predict as run_predict
from drift import check_drift_psi
from data_stream import stream_next_chunk, reset_stream, get_current_index, STREAM_POOL_PATH

# ── Constants ─────────────────────────────────────────────────────────────────
MODEL_PATH   = "models/xgb_model.pkl"
BACKUP_PATH  = "models/xgb_model_backup.pkl"
STREAM_CHUNK = 53938  # ~10% of dataset per chunk (538383 * 0.1)
PRED_TRIGGER      = 10     # stream new chunk every N predictions
RETRAIN_LOG_PATH  = Path("data/processed/retrain_history.json")

# ── State ─────────────────────────────────────────────────────────────────────
model           = load_model()
prediction_count = 0

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


@app.post("/retrain")
def retrain():
    """Manually trigger retraining with all accumulated stream data."""
    global model
    try:
        proc = subprocess.run(
            ["python", "src/train.py", "--all-data"],
            capture_output=True, text=True, cwd="/app"
        )
        if proc.returncode != 0:
            raise HTTPException(status_code=500, detail=proc.stderr)
        model = load_model()
        return {"status": "retraining complete", "output": proc.stdout}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reload-model")
def reload_model():
    """Reload model from disk."""
    global model
    model = load_model()
    return {"status": "model reloaded", "model": MODEL_PATH}


@app.post("/rollback")
def rollback():
    """Rollback to previous model version."""
    global model
    import shutil
    if not Path(BACKUP_PATH).exists():
        raise HTTPException(status_code=404, detail="No backup model found.")
    shutil.copy(BACKUP_PATH, MODEL_PATH)
    model = load_model()
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
        "retrain_history"   : retrain_history[-5:],   # last 5 retrains
        "model"             : MODEL_PATH,
        "backup_available"  : Path(BACKUP_PATH).exists(),
    }


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard():
    """Live dashboard showing loop status, drift, and model metrics."""
    return """
<!DOCTYPE html>
<html>
<head>
    <title>DelayPredict — Learning Loop Dashboard</title>
    <meta charset="utf-8">
    
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

    <div id="content">Loading...</div>

    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script>
    async function load() {
        const res = await fetch('/status');
        const d = await res.json();
        const drift = d.drift || {};
        const psi = drift.psi_scores || {};

        const driftBadge = drift.drift_detected
            ? '<span class="badge badge-red">⚠ DRIFT DETECTED</span>'
            : '<span class="badge badge-green">✓ STABLE</span>';

        const streamPct = d.stream_total > 0
            ? Math.round((d.stream_consumed / d.stream_total) * 100)
            : 0;

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
            retrainRows += `<tr>
                <td>${r.timestamp.substring(0,19)}</td>
                <td>${r.total_streamed?.toLocaleString() ?? '-'}</td>
                <td>${r.worst_feature ?? '-'}</td>
                <td style="color:#f87171">${r.max_psi ?? '-'}</td>
            </tr>`;
        }

        document.getElementById('content').innerHTML = `
        <div class="grid">
            <div class="card">
                <h3>Predictions</h3>
                <div class="metric">${d.predictions_made.toLocaleString()}</div>
                <div class="label">total predictions made</div>
            </div>
            <div class="card">
                <h3>Data Stream</h3>
                <div class="metric">${streamPct}%</div>
                <div class="label">${d.stream_consumed.toLocaleString()} / ${d.stream_total.toLocaleString()} rows consumed</div>
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
                    ${d.backup_available ? '<span class="badge badge-green">✓ Rollback available</span>' : '<span class="badge badge-yellow">No backup yet</span>'}
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
                    <tr><th>Time</th><th>Rows</th><th>Feature</th><th>PSI</th></tr>
                    ${retrainRows}
                </table>` : '<div style="color:#64748b">No retrains yet</div>'}
            </div>
        </div>

        <div class="card" style="margin-bottom:24px">
            <h3>Model Performance over Training Size</h3>
            <canvas id="metricsChart" height="80"></canvas>
            ${d.retrain_history.length === 0 ? '<div style="color:#64748b;margin-top:12px">No retrains yet — run predictions to trigger the loop</div>' : ''}
        </div>
        <p class="footer">Model: ${d.model} &nbsp;|&nbsp; AIOps SoSe 2026</p>`;

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
    }

    load();
    setInterval(() => {
        const y = window.scrollY;
        load().then(() => window.scrollTo(0, y));
    }, 30000);
    </script>
</body>
</html>
"""