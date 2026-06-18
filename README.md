# DelayPredict – Flight Delay Prediction

## Project Overview

Millions of travelers book flights daily using platforms like Skyscanner, Google Flights, or Kayak. While these platforms provide information about price, time, and availability, they lack one critical factor: **flight reliability**.

DelayPredict aims to solve this problem by providing a machine learning-based prediction of whether a flight will be delayed. The system includes a full **learning loop** that automatically detects data drift and retrains the model without manual intervention.

---

## Objective

The goal of this project is to build an end-to-end ML system that predicts:

> **Will a flight arrive with more than 15 minutes delay? (Yes/No)**

This is a **binary classification problem**.

---

## Dataset

We use the **Airline Delay Dataset (US Department of Transportation)** with 539,383 real US flights.

Features include:

* Airline
* Origin and destination airport
* Day of the week
* Departure time (extracted as departure hour)
* Flight duration

Target:

* `Delay` (0 = no delay, 1 = delay >15 min)

---

## Approach

### 1. Data Preparation
* Feature engineering (extract departure hour, build route string)
* 50/50 split: 50% training, 50% simulated data stream
* Stratified train/test split

### 2. Baseline Model
* Logistic Regression

### 3. Improved Model
* XGBoost with OrdinalEncoder pipeline

### 4. Evaluation
* Accuracy, Precision, Recall, F1 Score, ROC-AUC

### 5. Experiment Tracking
* MLflow tracks every run and retrain with parameters and metrics
* Two experiments: `DelayPredict` (progressive retrains) and `DelayPredict_Incremental` (learning curve rounds)

### 6. Deployment
* FastAPI endpoint (`/predict`) for inference

### 7. Learning Loop
* New data streams in chunks of ~54k rows
* PSI drift detection on Airline, DayOfWeek, DepartureHour
* Automatic retraining when drift is detected
* Rollback to previous model available at any time
* Rollback state is visible in the dashboard until the next retrain

### 8. Incremental Training
* 90/10 split: 10% fixed validation set, 90% training pool
* 10 cumulative rounds: 10%, 20%, ..., 100% of training pool
* Each round evaluated against the same validation set
* Learning curve shows model improvement with more data

---

## Project Structure

```
AIOps/
├── data/
│   ├── raw/                        # Original dataset
│   ├── processed/
│   │   ├── stream_pool.csv         # 50% held-out stream
│   │   ├── stream_data.csv         # Accumulated stream data
│   │   ├── stream_index.txt        # Current stream position
│   │   ├── train_reference.csv     # PSI reference distribution
│   │   ├── retrain_history.json    # Auto-loop & manual retrain history
│   │   └── incremental_history.json# Incremental training round results
│   ├── delaypredict.db             # SQLite database (predictions, training_runs, drift_scores)
│   └── metabase.db/                # Metabase internal config
├── models/
│   ├── xgb_model.pkl               # Current model
│   └── xgb_model_backup.pkl        # Backup for rollback
├── app/
│   └── main.py                     # FastAPI app + learning loop
├── src/
│   ├── train.py                    # Training (50/50 loop + 90/10 incremental)
│   ├── predict.py                  # Inference logic
│   ├── evaluate.py                 # Metrics
│   ├── drift.py                    # PSI drift detection
│   ├── data_preprocessing.py       # Feature engineering
│   ├── data_stream.py              # Stream simulation
│   ├── database.py                 # SQLAlchemy SQLite persistence
│   ├── seed_predictions.py         # Seed predictions table with demo data
│   └── demo.py                     # Full loop demo script
├── assets/
│   └── dash.css                    # Dark theme override for Plotly Dash
├── notebooks/mlruns/               # MLflow experiment logs
├── streamlit_app.py                # Streamlit user interface (Port 8501)
├── dash_app.py                     # Plotly Dash analytics & controls (Port 8050)
└── docker-compose.yml              # 6 services: api, streamlit, dash, mlflow, notebook, metabase
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

The API container automatically trains the model on startup (~45 seconds). The API is available once training completes. Watch progress with:

```bash
docker logs -f aiops-api-1
```

### 3. Seed demo data (optional)

To populate the Metabase dashboard with realistic prediction history:

```bash
docker exec aiops-api-1 python src/seed_predictions.py           # 1000 rows
docker exec aiops-api-1 python src/seed_predictions.py --rows 2000
```

Timestamps are distributed over the last 7 days for realistic charts.

---

## URLs

Once the system is running, the following pages are available:

| Page | URL | Audience |
|------|-----|----------|
| **Streamlit UI** | http://localhost:8501 | End users |
| **Metabase BI Dashboard** | http://localhost:3000 | ML engineers / stakeholders |
| **Data Analytics (Plotly Dash)** | http://localhost:8050 | Data analysis |
| **MLflow UI** | http://localhost:5001 | ML engineers |
| **API Docs (Swagger)** | http://localhost:8000/docs | Developers |
| **Health Check** | http://localhost:8000/health | — |
| **System Status** | http://localhost:8000/status | — |
| **Jupyter** | http://localhost:8888 | — |

> For access from other devices in the same network, replace `localhost` with your local IP address.

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Status check |
| `/predict` | POST | Make a delay prediction |
| `/status` | GET | Full system status (incl. `is_rolled_back`) |
| `/drift` | GET | PSI drift check |
| `/retrain` | POST | Progressive retrain (5 rounds, async) |
| `/rollback` | POST | Roll back to previous model |
| `/reload-model` | POST | Reload model from disk |
| `/stream/next` | POST | Stream next data chunk |
| `/stream/reset` | POST | Reset stream to beginning |
| `/incremental-training` | POST | Start 10-round incremental training |
| `/incremental-status` | GET | Current incremental training progress |

---

## Frontend

The project has three separate UIs for different audiences:

### Streamlit — User Interface (Port 8501)
**http://localhost:8501**

Designed for end-users and demos:
* Input form: Airline, Airport From/To, Day of Week, Departure Hour, Duration
* Prediction result with probability bar (✅ On time / ⚠️ Delay likely)
* Sidebar with links to the Metabase Dashboard and Data Analytics

### Metabase — BI Monitoring Dashboard (Port 3000)
**http://localhost:3000**

Designed for ML engineers and stakeholders. Connects directly to `delaypredict.db` and provides SQL-based dashboards over all three tables.

KPI tiles:
* Current ROC-AUC, Total Predictions, Last Training Run Type

Charts:
* **Model Performance over Training Size** — ROC-AUC, F1, Accuracy per progressive retrain round
* **Incremental Learning Curve** — metrics across all 10 incremental training rounds
* **PSI Drift per Feature** — latest PSI scores per feature
* **Run Type Breakdown** — count and average ROC-AUC per run type
* **Delay Rate by Airline / Day of Week / Departure Hour** — from the predictions table
* **Delay Probability Distribution** — bucketed histogram of model confidence

> **Setup (first run only):** Go to http://localhost:3000, complete the account setup, then add a SQLite database with filename `/app-data/data/delaypredict.db`.

### Plotly Dash — Dataset Analytics & Training Controls (Port 8050)
**http://localhost:8050**

Designed for data analysis and presentations. Shows static dataset statistics and provides buttons to trigger training operations.

* **Dataset KPIs** — total flights (539k), overall delay rate, number of airlines and routes (from CSV)
* **Training Controls** — Retrain, Incremental Training, and Rollback buttons with `?` tooltips and live status feedback
* **Rollback Banner** — orange notice when the model has been rolled back; disappears after the next retrain
* **Delay Rate by Airline** — top 12 airlines by volume, color-coded by delay rate
* **Delay Rate by Day of Week** — weekday comparison
* **Delay Rate by Departure Hour** — hourly delay pattern as area chart
* Link to Metabase for live model metrics

---

## Database (SQLite)

Every prediction, training run, and PSI drift check is persisted in `data/delaypredict.db` via SQLAlchemy (`src/database.py`).

Three tables:

| Table | Written by | Content |
|-------|-----------|---------|
| `predictions` | `/predict` endpoint | Airline, airports, day, hour, duration, delay result + probability |
| `training_runs` | `train.py`, `main.py` | run_type, round, train_size, ROC-AUC, F1, Accuracy, Precision, Recall, max_psi, worst_feature |
| `drift_scores` | Auto-loop (every stream chunk) | Feature name + PSI score per check |

`run_type` values: `initial`, `retrain`, `progressive`, `incremental`, `auto`

Each time a progressive retrain or incremental training starts, old entries of that type are cleared first. This lets Metabase show the learning curve being built up round by round in real time.

The database file is mounted into the Metabase container at `/app-data/data/delaypredict.db`.

---

## Experiment Tracking (MLflow)

MLflow logs every training run automatically. Access the UI at **http://localhost:5001**.

Two experiments are tracked:

| Experiment | Description |
|------------|-------------|
| `DelayPredict` | Initial training + all progressive retrain rounds |
| `DelayPredict_Incremental` | All 10 rounds of incremental training (one run per round) |

Each run stores: model parameters, train/validation size, ROC-AUC, F1, Accuracy, Precision, Recall, and the model artifact.

> **Note:** MLflow requires `MLFLOW_ALLOW_FILE_STORE=true` (set in `docker-compose.yml` for all services) since newer MLflow versions no longer support the file-based backend by default.

---

## Running the Learning Loop Demo

```bash
docker exec aiops-api-1 python src/demo.py
```

This automatically streams all data, detects drift, and retrains the model. Watch Metabase update live.

---

## Incremental Training

Run directly via terminal:

```bash
docker exec aiops-api-1 python src/train.py --incremental
```

Or use the **Start Incremental Training** button in the Plotly Dash analytics dashboard. Results appear in the Metabase learning curve chart after each round.

---

## Reset & Retrain from Scratch

**Mac/Linux:**
```bash
# 1. Delete old stream data
rm -f data/processed/stream_index.txt \
      data/processed/stream_data.csv \
      data/processed/retrain_history.json \
      data/processed/incremental_history.json

# 2. Restart API (triggers automatic retraining)
docker compose restart api
```

**Windows:**
```powershell
# 1. Delete old stream data
del data\processed\stream_index.txt
del data\processed\stream_data.csv
del data\processed\retrain_history.json
del data\processed\incremental_history.json

# 2. Restart API (triggers automatic retraining)
docker compose restart api
```

---

## Example Request

```
POST /predict
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

## Example Response

```json
{
  "delay_predicted": true,
  "delay_probability": 0.73,
  "model": "models/xgb_model.pkl"
}
```

---

## Limitations

* No weather data included
* Only US flight data
* PSI threshold set very low (0.00001) for demo purposes — in production this would be 0.1–0.2
* Model predictions are probabilistic, not guaranteed

---

## Authors

* Taylan Güler
* Marco Herrmann
* Julian Macher
* Marco Vierkorn

---

## Course

AI Operations (AIOps) – Hochschule Heilbronn
