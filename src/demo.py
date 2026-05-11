"""
src/demo.py
------------
Demonstriert den kompletten Learning Loop automatisch.
Schickt 10 Predictions an die API und zeigt was passiert.

Usage:
    python src/demo.py
"""

import requests
import time

API = "http://localhost:8000"

# 10 verschiedene Flüge
FLIGHTS = [
    {"Airline": "WN", "AirportFrom": "LAX", "AirportTo": "SFO", "DayOfWeek": 1, "Length": 60,  "DepartureHour": 8},
    {"Airline": "AA", "AirportFrom": "JFK", "AirportTo": "MIA", "DayOfWeek": 2, "Length": 190, "DepartureHour": 6},
    {"Airline": "DL", "AirportFrom": "ATL", "AirportTo": "ORD", "DayOfWeek": 3, "Length": 130, "DepartureHour": 9},
    {"Airline": "UA", "AirportFrom": "SFO", "AirportTo": "DEN", "DayOfWeek": 4, "Length": 158, "DepartureHour": 7},
    {"Airline": "CO", "AirportFrom": "IAH", "AirportTo": "LAX", "DayOfWeek": 5, "Length": 210, "DepartureHour": 5},
    {"Airline": "OO", "AirportFrom": "DEN", "AirportTo": "SLC", "DayOfWeek": 6, "Length": 90,  "DepartureHour": 10},
    {"Airline": "US", "AirportFrom": "CLT", "AirportTo": "DCA", "DayOfWeek": 7, "Length": 85,  "DepartureHour": 11},
    {"Airline": "B6", "AirportFrom": "JFK", "AirportTo": "BOS", "DayOfWeek": 1, "Length": 60,  "DepartureHour": 14},
    {"Airline": "XE", "AirportFrom": "IAH", "AirportTo": "HRL", "DayOfWeek": 2, "Length": 70,  "DepartureHour": 12},
    {"Airline": "MQ", "AirportFrom": "ATL", "AirportTo": "LGA", "DayOfWeek": 3, "Length": 125, "DepartureHour": 13},
]

def run_demo():
    print("=" * 55)
    print("  DelayPredict — Learning Loop Demo")
    print("=" * 55)

    # ── Health check ───────────────────────────────────────────
    r = requests.get(f"{API}/health")
    print(f"\nAPI Status: {r.json()['status'].upper()}")
    print(f"Model     : {r.json()['model']}")

    # ── 10 Predictions ─────────────────────────────────────────
    print(f"\nSending 10 predictions...")
    print("-" * 55)

    for i, flight in enumerate(FLIGHTS, 1):
        r = requests.post(f"{API}/predict", json=flight)
        result = r.json()
        print(f"  [{i:2}] {flight['Airline']} {flight['AirportFrom']}→{flight['AirportTo']} "
              f"| delayed={result['delay_predicted']} "
              f"| prob={result['delay_probability']:.2f}")
        time.sleep(0.3)

    # ── Wait for auto loop ─────────────────────────────────────
    print("\nWaiting for Auto Loop to complete...")
    time.sleep(3)

    # ── Drift check ────────────────────────────────────────────
    r = requests.get(f"{API}/drift")
    drift = r.json()
    print("\n" + "=" * 55)
    print("  Auto Loop Result")
    print("=" * 55)
    print(f"  Drift detected  : {drift.get('drift_detected')}")
    print(f"  Accuracy        : {drift.get('current_accuracy', 'N/A')}")
    print(f"  Threshold       : {drift.get('threshold', 'N/A')}")
    print(f"  Feedback count  : {drift.get('feedback_count', 'N/A')}")

    if drift.get("drift_detected"):
        print("\n  Drift detected — retraining was triggered automatically")
        print("  Model has been reloaded")
    else:
        print("\n  No drift — model is stable")
    print("=" * 55)

if __name__ == "__main__":
    run_demo()