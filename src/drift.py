"""
src/drift.py
------------
Population Stability Index (PSI) based drift detection.
Compares feature distributions between training data and new incoming data.

PSI < 0.0001 → No drift
PSI > 0.0001 → Drift detected → trigger retrain

Usage:
    from drift import compute_psi, check_drift_psi
"""

import numpy as np
import pandas as pd
from pathlib import Path

# ── Constants ─────────────────────────────────────────────────────────────────
PSI_THRESHOLD    = 0.00001  # low threshold to trigger retrain for demo
MONITOR_FEATURES = ["Airline", "DayOfWeek", "DepartureHour"]  # features to monitor
TRAIN_REF_PATH   = Path("data/processed/train_reference.csv")  # saved training distribution


def compute_psi(expected: pd.Series, actual: pd.Series, buckets: int = 10) -> float:
    """
    Compute PSI between expected (training) and actual (new) distributions.
    Works for both categorical and numeric features.
    """
    # ── Categorical features ───────────────────────────────────────────────────
    if expected.dtype == object or expected.nunique() <= buckets:
        all_cats = set(expected.unique()) | set(actual.unique())
        expected_pct = expected.value_counts(normalize=True).reindex(all_cats, fill_value=1e-6)
        actual_pct   = actual.value_counts(normalize=True).reindex(all_cats, fill_value=1e-6)
    else:
        # ── Numeric features — use quantile buckets ────────────────────────────
        breakpoints = np.quantile(expected, np.linspace(0, 1, buckets + 1))
        breakpoints = np.unique(breakpoints)
        expected_pct = pd.cut(expected, bins=breakpoints, include_lowest=True).value_counts(normalize=True).sort_index()
        actual_pct   = pd.cut(actual,   bins=breakpoints, include_lowest=True).value_counts(normalize=True).reindex(expected_pct.index, fill_value=1e-6)

    expected_pct = expected_pct.clip(lower=1e-6)
    actual_pct   = actual_pct.clip(lower=1e-6)

    psi = float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))
    return round(psi, 4)


def save_training_reference(df_train: pd.DataFrame):
    """Save training feature distributions as reference for future drift checks."""
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    df_train[MONITOR_FEATURES].to_csv(TRAIN_REF_PATH, index=False)
    print(f"Training reference saved: {TRAIN_REF_PATH}")


def check_drift_psi(df_new: pd.DataFrame) -> dict:
    """
    Compare new data against training reference using PSI.
    Returns drift status and PSI per feature.
    """
    if not TRAIN_REF_PATH.exists():
        return {
            "drift_detected": False,
            "reason": "No training reference found. Run train.py first.",
            "psi_scores": {}
        }

    df_ref = pd.read_csv(TRAIN_REF_PATH)
    psi_scores = {}

    for feature in MONITOR_FEATURES:
        if feature in df_ref.columns and feature in df_new.columns:
            psi = compute_psi(df_ref[feature], df_new[feature])
            psi_scores[feature] = psi

    max_psi     = max(psi_scores.values()) if psi_scores else 0.0
    drift       = bool(max_psi > PSI_THRESHOLD)
    worst_feat  = max(psi_scores, key=psi_scores.get) if psi_scores else None

    return {
        "drift_detected" : drift,
        "max_psi"        : round(max_psi, 4),
        "psi_threshold"  : PSI_THRESHOLD,
        "psi_scores"     : psi_scores,
        "worst_feature"  : worst_feat,
    }