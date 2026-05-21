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

### 6. Deployment
* FastAPI endpoint (`/predict`) for inference

### 7. Learning Loop
* New data streams in chunks of ~54k rows
* PSI drift detection on Airline, DayOfWeek, DepartureHour
* Automatic retraining when drift is detected
* Rollback to previous model available at any time

---

## Project Structure

```
airline-delay-prediction/
│
├── data/
│   ├── raw/                  # Original dataset
│   └── processed/            # Stream pool, drift reference, retrain history
├── models/                   # Trained model + backup
├── notebooks/                # Jupyter notebooks + mlruns
├── src/
│   ├── main.py               # FastAPI app + learning loop
│   ├── train.py              # Model training
│   ├── predict.py            # Inference logic
│   ├── evaluate.py           # Metrics
│   ├── drift.py              # PSI drift detection
│   ├── data_preprocessing.py # Feature engineering
│   ├── data_stream.py        # Stream simulation
│   └── demo.py               # Full loop demo script
└── mlruns/                   # MLflow experiment logs
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

| Page | URL |
|------|-----|
| **Dashboard** | http://localhost:8000/dashboard |
| **API Docs (Swagger)** | http://localhost:8000/docs |
| **Health Check** | http://localhost:8000/health |
| **System Status** | http://localhost:8000/status |
| **MLflow UI** | http://localhost:5001 |
| **Jupyter** | http://localhost:8888 |

> For access from other devices in the same network, replace `localhost` with your local IP address (e.g. `http://192.168.178.175:8000/dashboard`).

---

## Running the Learning Loop Demo

```bash
docker exec aiops-api-1 python src/demo.py
```

This automatically streams all data, detects drift, and retrains the model. Watch the dashboard update live.

---

## Reset & Retrain from Scratch

**Mac/Linux:**
```bash
# 1. Delete old stream data
rm -f data/processed/stream_index.txt data/processed/stream_data.csv data/processed/retrain_history.json

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
