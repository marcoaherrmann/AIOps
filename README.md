# DelayPredict – Flight Delay Prediction

## Project Overview

Millions of travelers book flights daily using platforms like Skyscanner, Google Flights, or Kayak. While these platforms provide information about price, time, and availability, they lack one critical factor: **flight reliability**.

DelayPredict solves this by providing a machine learning-based prediction of whether a flight will be delayed. The system includes a full **AIOps learning loop** that automatically detects data drift and retrains the model without manual intervention.

---

## Objective

> **Will a flight arrive with more than 15 minutes delay? (Yes/No)**

Binary classification using XGBoost, with automatic drift detection and retraining.

---

## Dataset

**Airline Delay Dataset (US Department of Transportation)** — 539,383 real US flights.

| Feature | Type | Description |
|---------|------|-------------|
| Airline | Categorical | IATA code (e.g. WN, DL, AA) |
| AirportFrom | Categorical | Departure airport IATA code |
| AirportTo | Categorical | Arrival airport IATA code |
| Route | Categorical | Feature-engineered: AirportFrom + "-" + AirportTo |
| DayOfWeek | Categorical | 1 = Monday, 7 = Sunday |
| DepartureHour | Numeric | Feature-engineered: Time // 60 (0–23) |
| Length | Numeric | Flight duration in minutes |

Target: `Delay` (0 = on time, 1 = delayed > 15 min)  **~44% delayed**

---

## Model

XGBoost v2 binary classifier wrapped in a scikit-learn Pipeline.

| Model | ROC-AUC | Accuracy | F1 |
|-------|---------|----------|----|
| Dummy Classifier | 0.500 | 55.5% | 0.000 |
| Logistic Regression | 0.692 | 64.6% | 0.551 |
| Decision Tree | 0.703 | 65.5% | 0.571 |
| Random Forest | 0.720 | 66.7% | 0.572 |
| **XGBoost (chosen)** | **0.724** | **67.1%** | **0.586** |

**XGBoost Hyperparameters:**
```python
n_estimators=500, learning_rate=0.1, max_depth=6,
subsample=0.8, colsample_bytree=0.8, min_child_weight=100
```

Pipeline: `OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)` for 5 categorical features + passthrough for 2 numeric features.

---

## 50/50 Split - Core Mechanism

- **50% (~270k rows)** → Training base (`random_state=42`)
- **50% (~270k rows)** → `stream_pool.csv` (simulates incoming real-time data)
- Stream pool is split into **5 chunks of ~54k rows**

The auto-loop processes one chunk every 10 predictions. The **Retrain button** always uses the full `stream_pool.csv` (all 270k rows) across 5 progressive rounds.

### Progressive Retrain Results

| Round | Train Rows | ROC-AUC |
|-------|-----------|---------|
| 0 (baseline) | 269,692 | 0.7153 |
| 1 | 323,630 | 0.7201 |
| 2 | 377,568 | 0.7240 |
| 3 | 431,507 | 0.7269 |
| 4 | 485,445 | 0.7297 |
| 5 (full) | 539,383 | 0.7322 |

### Incremental Training (10 Rounds, 90/10 Split)

10 cumulative rounds on a fixed validation set (10% of training pool):
- Round 1 (10% = 48k rows) → ROC-AUC 0.7011
- Round 10 (100% = 485k rows) → ROC-AUC 0.7208

Does not replace the live model - for learning curve analysis only.

---

## Project Structure

```
AIOps/
├── data/
│   ├── raw/airlines_delay.csv          # Original dataset (539k rows)
│   ├── processed/
│   │   ├── stream_pool.csv             # 50% held-out (created once by train.py)
│   │   ├── stream_data.csv             # Accumulated stream data (grows per chunk)
│   │   ├── stream_index.txt            # Current stream position
│   │   ├── train_reference.csv         # PSI reference distribution
│   │   ├── retrain_history.json        # Progressive retrain history
│   │   └── incremental_history.json    # Incremental training results
│   ├── delaypredict.db                 # SQLite: predictions, training_runs, drift_scores
│   ├── mlflow.db                       # MLflow backend DB
│   └── metabase.db/                    # Metabase internal config (H2)
├── models/
│   ├── xgb_model.pkl                   # Active model
│   └── xgb_model_backup.pkl            # Backup for rollback
├── app/
│   └── main.py                         # FastAPI app + auto-loop + progressive retrain
├── src/
│   ├── train.py                        # Training (initial, retrain, incremental)
│   ├── predict.py                      # Inference
│   ├── evaluate.py                     # Metrics (ROC-AUC, F1, Accuracy, Precision, Recall)
│   ├── drift.py                        # PSI drift detection
│   ├── data_preprocessing.py           # Feature engineering, FEATURES/TARGET constants
│   ├── data_stream.py                  # Stream pool management
│   ├── database.py                     # SQLAlchemy persistence
│   ├── seed_predictions.py             # Seed demo data for Metabase
│   └── demo.py                         # Full loop demo script
├── assets/
│   └── dash.css                        # Dark theme + tooltip styles for Plotly Dash
├── streamlit_app.py                    # Streamlit prediction UI (Port 8501)
├── dash_app.py                         # Plotly Dash analytics & controls (Port 8050)
├── docker-compose.yml                  # 6 services
└── requirements.txt
```

---

## Setup

### 1. Clone repository

```bash
git clone https://github.com/marcoaherrmann/AIOps.git
cd AIOps
```

### 2. Start with Docker

```bash
docker compose up -d
```

The API container automatically trains the model on startup (~45 seconds). Watch progress:

```bash
docker logs -f aiops-api-1
```

### 3. Seed demo data (optional)

Populates Metabase with realistic prediction history (stratified sampling, timestamps spread over last 7 days):

```bash
docker exec aiops-api-1 python src/seed_predictions.py --rows 2000
```

---

## URLs

| Service | URL | Purpose |
|---------|-----|---------|
| Streamlit UI | http://localhost:8501 | End-user prediction form |
| Metabase Dashboard | http://localhost:3000 | BI monitoring (15 pre-built cards) |
| Plotly Dash | http://localhost:8050 | Dataset analytics + training controls |
| MLflow | http://localhost:5001 | Experiment tracking |
| API Docs | http://localhost:8000/docs | Swagger UI |
| Health Check | http://localhost:8000/health | — |
| System Status | http://localhost:8000/status | Full system state incl. rollback status |
| Jupyter | http://localhost:8888 | EDA notebooks |

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Status check |
| `/predict` | POST | Make a delay prediction + triggers auto-loop |
| `/status` | GET | Full system status (incl. `is_rolled_back`) |
| `/drift` | GET | PSI drift check on accumulated stream data |
| `/retrain` | POST | Progressive retrain on full stream pool (5 rounds, async) |
| `/rollback` | POST | Roll back to backup model |
| `/reload-model` | POST | Reload model from disk |
| `/stream/next` | POST | Stream next data chunk manually |
| `/stream/reset` | POST | Reset stream to beginning |
| `/incremental-training` | POST | Start 10-round incremental training (async) |
| `/incremental-status` | GET | Current incremental training progress |

---

## Frontend

### Streamlit - User Interface (Port 8501)

For end-users and demos. Navigation links to Dash, Metabase, and MLflow in the header.

- Input form: Airline, Airport From/To, Day of Week, Departure Hour, Duration
- Result: ✅ On time / ⚠️ Delay likely with probability bar

### Plotly Dash — Dataset Analytics & Training Controls (Port 8050)

For data analysis and presentations.

- **Navigation bar** — links to Metabase, Streamlit, MLflow, API Docs
- **Dataset KPIs** — total flights (539k), delay rate (~44%), airlines, routes (from CSV)
- **Training Controls** — three buttons with `?` tooltips:
  - `🔄 Retrain on Stream Data` — progressive retrain using full stream pool (5 rounds, 270k → 539k training rows)
  - `📈 Start Incremental Training` — 10-round learning curve analysis
  - `↩ Rollback to Previous` — restore backup model
- **Rollback Banner** — orange notice when model has been rolled back; disappears after next retrain
- **Static charts** from raw CSV: Delay Rate by Airline, Day of Week, Departure Hour

### Metabase — BI Monitoring Dashboard (Port 3000)

For ML engineers and stakeholders. 15 pre-configured cards over `delaypredict.db`:

| Card | Type | Content |
|------|------|---------|
| Current ROC-AUC | Scalar | Latest model metric |
| Total Predictions | Scalar | Count from predictions table |
| Last Training Run | Scalar | run_type with emoji (🔵 Progressive, 🤖 Auto, etc.) |
| Last Trained | Scalar | Timestamp of last training run |
| Model Performance over Training Size | Line chart | ROC-AUC, F1, Accuracy per progressive round |
| Incremental Learning Curve | Line chart | ROC-AUC, F1 across 10 rounds |
| PSI Drift per Feature | Bar chart | Latest PSI scores per feature |
| PSI Drift Timeline | Table | All drift checks over time |
| Delay Rate by Airline | Bar chart | From predictions table |
| Delay Rate by Day of Week | Bar chart | From predictions table |
| Delay Rate by Departure Hour | Line chart | From predictions table |
| Delay Probability Distribution | Bar chart | Bucketed histogram |
| Predictions per Day | Bar chart | Daily prediction volume |
| All Training Runs | Table | All run types, full history |
| Training Run Log | Table | Simplified run history |

**Important SQL constraints for Metabase:**
- `is_rolled_back` does NOT exist in `training_runs` — rollback is in-memory only
- `drift_detected` does NOT exist in `drift_scores` — use `psi_score > 0.00001` instead
- `accuracy` is stored as decimal (0.671, not 67.1%)

---

## Learning Loop

### Auto-Loop (every 10 predictions)

1. Stream next chunk (~54k rows) from `stream_pool.csv` into `stream_data.csv`
2. PSI drift check on all accumulated `stream_data.csv`
3. If drift detected (PSI > 0.00001) → retrain on all accumulated data + reload model
4. Log to `training_runs` (run_type = `auto`) and `drift_scores`

**Auto-loop is paused during progressive retrain** to avoid conflicting training runs.

### PSI Drift Detection

```
PSI_THRESHOLD    = 0.00001  # demo-low (production: 0.1–0.2)
MONITOR_FEATURES = ["Airline", "DayOfWeek", "DepartureHour"]
```

Reference distribution: `train_reference.csv` (training base, never changes).

### Rollback

- Sets `is_rolled_back = True` in memory (not persisted to DB)
- Orange banner shown in Plotly Dash until next retrain
- `/status` endpoint returns `is_rolled_back`

---

## Database (SQLite)

`data/delaypredict.db` via SQLAlchemy (`src/database.py`):

### `predictions`
```
id, timestamp, airline, airport_from, airport_to, day_of_week,
departure_hour, length, delay_predicted (bool), delay_probability (float)
```

### `training_runs`
```
id, timestamp, run_type, round (nullable), train_size,
roc_auc, f1, accuracy, precision, recall, max_psi (nullable), worst_feature (nullable)
```
`run_type` values: `initial` | `progressive` | `incremental` | `retrain` | `auto`

### `drift_scores`
```
id, timestamp, feature, psi_score
```

---

## Experiment Tracking (MLflow)

| Experiment | Content |
|------------|---------|
| `DelayPredict` | Initial training + progressive retrain rounds |
| `DelayPredict_Incremental` | 10 incremental training rounds |

Each run logs: hyperparameters, train size, ROC-AUC, F1, Accuracy, Precision, Recall.

> MLflow requires `MLFLOW_ALLOW_FILE_STORE=true` (set in `docker-compose.yml`).

---

## Demo Workflow

```bash
# 1. Start everything
docker compose up -d
# Wait ~45s for initial training

# 2. Seed Metabase with demo data
docker exec aiops-api-1 python src/seed_predictions.py --rows 2000

# 3. Show auto-loop: reset stream, send 10 predictions
curl -X POST http://localhost:8000/stream/reset

for i in $(seq 1 10); do
  curl -s -X POST http://localhost:8000/predict \
    -H "Content-Type: application/json" \
    -d '{"Airline":"WN","AirportFrom":"LAX","AirportTo":"JFK","DayOfWeek":1,"Length":300,"DepartureHour":8}'
done
# → Auto-loop triggers: loads chunk, detects drift, retrains automatically

# 4. Show progressive retrain: click "Retrain on Stream Data" in Dash (http://localhost:8050)
# → 5 rounds, training size grows from 269k to 539k, Metabase chart updates live

# 5. Show rollback: click "Rollback to Previous" → orange banner appears in Dash
# → Next retrain clears the banner

# 6. Full automated demo
docker exec aiops-api-1 python src/demo.py
```

---

## Reset & Retrain from Scratch

```bash
# Delete stream state
rm -f data/processed/stream_index.txt \
      data/processed/stream_data.csv \
      data/processed/retrain_history.json \
      data/processed/incremental_history.json

# Restart API (triggers automatic initial training)
docker compose restart api
```

---

## Example Request

```
POST http://localhost:8000/predict
Content-Type: application/json
```

```json
{
  "Airline": "DL",
  "AirportFrom": "LAX",
  "AirportTo": "JFK",
  "DayOfWeek": 1,
  "DepartureHour": 7,
  "Length": 240
}
```

```json
{
  "delay_predicted": true,
  "delay_probability": 0.73,
  "model": "models/xgb_model.pkl"
}
```

---

## Known Quirks

1. **Multiple `initial` runs in DB** — every container restart triggers `train.py` → new `initial` entry. Metabase charts use `GROUP BY train_size` to deduplicate.
2. **Rollback is in-memory only** — resets on container restart. Visible in Dash banner and `/status`, not in Metabase.
3. **PSI threshold 0.00001** — intentionally low for demo. Auto-loop always detects drift since stream data comes from the same source as training data.
4. **Metabase rate limiting** — too many failed login attempts → locked out. Fix: `docker compose restart metabase`.
5. **`stream_pool.csv` created once** — `train.py` skips creation if file exists. Delete manually if you need to regenerate.
6. **Accuracy in DB** — stored as decimal (0.671). Use `CAST(round(accuracy*100,1) AS TEXT) || ' %'` in Metabase for display.

---

## Limitations

- No weather data
- US flights only
- PSI threshold set low for demo (production standard: 0.1–0.2)
- Predictions are probabilistic estimates

---

## Authors

- Taylan Güler
- Barco
- Julian Macher
- Marco Vierkorn

---

## Course

AI Operations (AIOps) · Hochschule Heilbronn · SoSe 2026 · Dozent: Pranav Sharma
