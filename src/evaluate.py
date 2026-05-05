"""
src/evaluate.py
----------------
Evaluates a trained model on the test set and prints all metrics.
Can be run standalone after train.py has saved the model.

Usage:
    python src/evaluate.py
"""

import joblib
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
)

from data_preprocessing import load_data, get_splits

# ── Constants ─────────────────────────────────────────────────────────────────
MODEL_PATH      = Path("models/xgb_model.pkl")
PROCESSED_DIR   = Path("data/processed")


def compute_metrics(model, X_test, y_test) -> dict:
    """Compute all evaluation metrics for a fitted model."""
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Accuracy" : accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall"   : recall_score(y_test, y_pred, zero_division=0),
        "F1"       : f1_score(y_test, y_pred, zero_division=0),
        "ROC-AUC"  : roc_auc_score(y_test, y_proba),
    }
    return metrics


def print_metrics(metrics: dict, model_name: str = "Model"):
    """Print metrics in a readable format."""
    print(f"\n{model_name} metrics:")
    for k, v in metrics.items():
        print(f"  {k:<10}: {v:.4f}")


def save_metrics(metrics: dict, filename: str = "xgb_metrics.csv"):
    """Save metrics to data/processed/ as CSV."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    path = PROCESSED_DIR / filename
    df = pd.DataFrame([{"model": "XGBoost", **metrics}])
    df.to_csv(path, index=False)
    print(f"Metrics saved: {path}")


if __name__ == "__main__":
    # Load data and model
    df = load_data()
    _, X_test, _, y_test = get_splits(df)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}.\n"
            "Run: python src/train.py first."
        )

    model = joblib.load(MODEL_PATH)
    print(f"Model loaded: {MODEL_PATH}")

    # Evaluate
    metrics = compute_metrics(model, X_test, y_test)
    print_metrics(metrics, model_name="XGBoost")

    # Classification report
    y_pred = model.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["On Time", "Delayed"]))

    # Save metrics
    save_metrics(metrics)