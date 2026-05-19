"""
src/demo.py
-----------
Runs the complete learning loop automatically.
Sends predictions until the entire data stream is consumed.

Usage:
    python src/demo.py
    python src/demo.py --predictions-per-chunk 10  # predictions to trigger each chunk
"""

import requests
import time
import argparse
import random

import os
API = os.environ.get("API_URL", "http://localhost:8000")

AIRLINES     = ["WN", "AA", "DL", "UA", "CO", "OO", "US", "B6", "XE", "MQ", "EV", "FL"]
AIRPORTS     = ["LAX", "SFO", "JFK", "MIA", "ATL", "ORD", "DEN", "SLC", "DCA", "IAH", "BOS"]
DAYS         = [1, 2, 3, 4, 5, 6, 7]
HOURS        = list(range(6, 23))
LENGTHS      = [45, 60, 90, 120, 150, 180, 210, 240, 270, 300]


def random_flight():
    return {
        "Airline"      : random.choice(AIRLINES),
        "AirportFrom"  : random.choice(AIRPORTS),
        "AirportTo"    : random.choice(AIRPORTS),
        "DayOfWeek"    : random.choice(DAYS),
        "Length"       : random.choice(LENGTHS),
        "DepartureHour": random.choice(HOURS),
    }


def run_demo(predictions_per_chunk: int = 10):
    print("=" * 60)
    print("  DelayPredict — Full Learning Loop Demo")
    print("=" * 60)

    # ── Health check ───────────────────────────────────────────────
    r = requests.get(f"{API}/health")
    print(f"\nAPI: {r.json()['status'].upper()} | Model: {r.json()['model']}")

    # ── Reset stream to beginning ──────────────────────────────────
    requests.post(f"{API}/stream/reset")
    print("Stream reset to beginning.\n")

    chunk = 0
    total_predictions = 0

    while True:
        chunk += 1
        print(f"\n{'='*60}")
        print(f"  Chunk {chunk} — sending {predictions_per_chunk} predictions...")
        print(f"{'='*60}")

        # ── Send predictions ───────────────────────────────────────
        for i in range(predictions_per_chunk):
            flight = random_flight()
            r = requests.post(f"{API}/predict", json=flight)
            result = r.json()
            total_predictions += 1
            print(f"  [{total_predictions:3}] {flight['Airline']} "
                  f"{flight['AirportFrom']}→{flight['AirportTo']} "
                  f"| delayed={result['delay_predicted']} "
                  f"| prob={result['delay_probability']:.2f}")
            time.sleep(0.1)

        # ── Wait for loop to complete ──────────────────────────────
        print("\nWaiting for auto loop...")
        time.sleep(5)

        # ── Check status ───────────────────────────────────────────
        status = requests.get(f"{API}/status").json()
        drift  = status.get("drift", {})
        stream_remaining = status.get("stream_remaining", 0)

        print(f"\n  Stream consumed : {status['stream_consumed']:,} / {status['stream_total']:,} rows")
        print(f"  Drift detected  : {drift.get('drift_detected', False)}")
        print(f"  Max PSI         : {drift.get('max_psi', 'N/A')}")
        print(f"  Worst feature   : {drift.get('worst_feature', 'N/A')}")
        print(f"  Retrains so far : {status['retrain_count']}")

        if status['retrain_count'] > 0:
            last = status['retrain_history'][-1]
            print(f"  Last retrain    : ROC-AUC={last.get('roc_auc','N/A')} | F1={last.get('f1','N/A')}")

        # ── Check if stream is complete ────────────────────────────
        if stream_remaining == 0:
            print(f"\n{'='*60}")
            print("  Stream complete! All data consumed.")
            print(f"  Total predictions : {total_predictions}")
            print(f"  Total retrains    : {status['retrain_count']}")
            print(f"\n  View dashboard: http://localhost:8000/dashboard")
            print(f"{'='*60}")
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions-per-chunk", type=int, default=10)
    args = parser.parse_args()
    run_demo(predictions_per_chunk=args.predictions_per_chunk)