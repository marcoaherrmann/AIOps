# DelayPredict – Flight Delay Prediction

## Project Overview

Millions of travelers book flights daily using platforms like Skyscanner, Google Flights, or Kayak. While these platforms provide information about price, time, and availability, they lack one critical factor: **flight reliability**.

DelayPredict aims to solve this problem by providing a machine learning-based prediction of whether a flight will be delayed.

---

## Objective

The goal of this project is to build an end-to-end ML system that predicts:

> **Will a flight arrive with more than 15 minutes delay? (Yes/No)**

This is a **binary classification problem**.

---

## Dataset

We use the **Airline Delay Dataset (US Department of Transportation)** containing millions of real flights.

Features include:

* Airline
* Origin and destination airport
* Day of the week
* Departure time
* Flight distance

Target:

* `Delay` (0 = no delay, 1 = delay)

---

## Approach

### 1. Data Preparation

* Clean dataset
* Feature engineering (e.g. extract departure hour)
* Train/test split

### 2. Baseline Model

* Logistic Regression

### 3. Improved Model

* Random Forest (or similar)

### 4. Evaluation

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC

### 5. Experiment Tracking

* MLflow for tracking parameters, metrics, and models

### 6. Deployment

* FastAPI endpoint (`/predict`) for inference

---

## Project Structure

```
airline-delay-prediction/
│
├── data/
├── notebooks/
├── src/
├── models/
├── app/
└── mlruns/
```

---

## Setup

### 1. Clone repository

```
git clone <your-repo-url>
cd airline-delay-prediction
```

### 2. Create virtual environment

```
python -m venv venv
venv\Scripts\activate   # Windows
# source venv/bin/activate   # Mac/Linux
```

### 3. Install dependencies

```
pip install -r requirements.txt
```

---

## Running the project

### Run notebooks

```
jupyter notebook
```

### Run FastAPI app

```
uvicorn app.main:app --reload
```

---

## Example Request

```
POST /predict
```

```json
{
  "airline": "DL",
  "origin": "LAX",
  "destination": "JFK",
  "day_of_week": 1,
  "departure_hour": 7,
  "length": 3983
}
```

---

## Example Response

```json
{
  "is_delayed": true,
  "probability": 0.73
}
```

---

## Limitations

* No weather data included
* Only US flight data
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
 