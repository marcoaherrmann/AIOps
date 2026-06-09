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
│   └── processed/
│       ├── stream_pool.csv         # 50% held-out stream
│       ├── stream_data.csv         # Accumulated stream data
│       ├── stream_index.txt        # Current stream position
│       ├── train_reference.csv     # PSI reference distribution
│       ├── retrain_history.json    # Auto-loop & manual retrain history
│       └── incremental_history.json# Incremental training round results
├── models/
│   ├── xgb_model.pkl               # Current model
│   └── xgb_model_backup.pkl        # Backup for rollback
├── app/
│   └── main.py                     # FastAPI app + dashboard + learning loop
├── src/
│   ├── train.py                    # Training (50/50 loop + 90/10 incremental)
│   ├── predict.py                  # Inference logic
│   ├── evaluate.py                 # Metrics
│   ├── drift.py                    # PSI drift detection
│   ├── data_preprocessing.py       # Feature engineering
│   ├── data_stream.py              # Stream simulation
│   └── demo.py                     # Full loop demo script
├── assets/
│   └── dash.css                    # Dark theme override for Plotly Dash
├── notebooks/mlruns/               # MLflow experiment logs
├── streamlit_app.py                # Streamlit user interface (Port 8501)
├── dash_app.py                     # Plotly Dash analytics dashboard (Port 8050)
└── docker-compose.yml              # 5 services: api, streamlit, dash, mlflow, notebook
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

### 3. Initial training

```bash
docker exec aiops-api-1 python src/train.py
docker compose restart api
```

---

## URLs

Once the system is running, the following pages are available:

| Page | URL | Audience |
|------|-----|----------|
| **Streamlit UI** | http://localhost:8501 | End users |
| **Monitoring Dashboard** | http://localhost:8000/dashboard | ML engineers |
| **Data Analytics (Plotly Dash)** | http://localhost:8050 | Data analysis |
| **MLflow UI** | http://localhost:5001 | ML engineers |
| **API Docs (Swagger)** | http://localhost:8000/docs | Developers |
| **Health Check** | http://localhost:8000/health | — |
| **System Status** | http://localhost:8000/status | — |
| **Jupyter** | http://localhost:8888 | — |

> For access from other devices in the same network, replace `localhost` with your local IP address (e.g. `http://192.168.178.175:8000/dashboard`).

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Status check |
| `/predict` | POST | Make a delay prediction |
| `/status` | GET | Full system status (incl. `is_rolled_back`) |
| `/dashboard` | GET | Live monitoring dashboard |
| `/drift` | GET | PSI drift check |
| `/retrain` | POST | Progressive retrain (5 rounds, live chart) |
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
* Sidebar with buttons linking to the Monitoring Dashboard and Data Analytics

### Custom Dashboard — Monitoring (Port 8000)
**http://localhost:8000/dashboard**

Designed for ML engineers:
* **Try a Prediction** — embedded prediction form directly in the dashboard
* **Model ROC-AUC** — current model performance from the last retrain
* **Rollback Indicator** — orange banner and badge visible whenever the model has been rolled back; disappears after the next retrain
* **Training Progress / Data Stream** — dynamically shows training data fraction during retrain
* **Drift Status** — PSI per feature with color-coded bars (green / yellow / red)
* **Retrain History** — last 5 retrains with timestamp, train rows, worst feature, and PSI
* **Retrain on Stream Data** — 5-round progressive retrain; chart builds up live round by round
* **Rollback to Previous** — one-click rollback with confirmation banner
* **Model Performance over Training Size** — line chart (ROC-AUC, F1, Accuracy) building up live
* **Incremental Training — Learning Curve** — 10-round training via button; chart fills in round by round

Auto-refreshes every 5 seconds.

### Plotly Dash — Data Analytics (Port 8050)
**http://localhost:8050**

Designed for data analysis and presentations:
* **Dataset KPIs** — total flights, overall delay rate, number of airlines and routes
* **Live Model KPIs** — current ROC-AUC, F1, retrain count, incremental ROC (auto-refreshes every 10s)
* **Training Controls** — Retrain, Incremental Training, and Rollback buttons with live status feedback
* **Model Performance Chart** — ROC-AUC, F1, Accuracy over progressive retrain rounds
* **Incremental Learning Curve** — metrics across all 10 incremental training rounds
* **PSI Drift per Feature** — horizontal bar chart with monitor/retrain threshold lines
* **Delay Rate by Airline** — top 12 airlines by volume, color-coded by delay rate
* **Delay Rate by Day of Week** — weekday comparison
* **Delay Rate by Departure Hour** — hourly delay pattern as area chart

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

This automatically streams all data, detects drift, and retrains the model. Watch the dashboard update live.

---

## Incremental Training

Run directly via terminal:

```bash
docker exec aiops-api-1 python src/train.py --incremental
```

Or use the **Start Incremental Training** button in the Monitoring Dashboard or Data Analytics. Results are saved to `data/processed/incremental_history.json` and the learning curve chart updates after each round.

---

## Reset & Retrain from Scratch

**Mac/Linux:**
```bash
# 1. Delete old stream data
rm -f data/processed/stream_index.txt \
      data/processed/stream_data.csv \
      data/processed/retrain_history.json \
      data/processed/incremental_history.json

# 2. Retrain model
docker exec aiops-api-1 python src/train.py

# 3. Restart API
docker compose restart api
```

**Windows:**
```powershell
# 1. Delete old stream data
del data\processed\stream_index.txt
del data\processed\stream_data.csv
del data\processed\retrain_history.json
del data\processed\incremental_history.json

# 2. Retrain model
docker exec aiops-api-1 python src/train.py

# 3. Restart API
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
