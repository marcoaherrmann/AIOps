"""
src/train.py
-------------
Trains the XGBoost model and saves it to models/xgb_model.pkl.
Also logs params and metrics to MLflow under the DelayPredict experiment.

Usage:
    python src/train.py
"""

import joblib
import mlflow
import mlflow.sklearn
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier

from data_preprocessing import load_data, get_splits, CATEGORICAL, NUMERIC
from evaluate import compute_metrics, print_metrics, save_metrics

# ── Constants ─────────────────────────────────────────────────────────────────
MODEL_PATH   = Path("models/xgb_model.pkl")
MLFLOW_URI   = "file:///app/notebooks/mlruns"
RANDOM_STATE = 42

# ── Hyperparameters ───────────────────────────────────────────────────────────
PARAMS = {
    "n_estimators"    : 500,
    "learning_rate"   : 0.1,
    "max_depth"       : 6,
    "subsample"       : 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 100,
}


def build_pipeline() -> Pipeline:
    """Build the XGBoost pipeline with OrdinalEncoder preprocessing."""
    preprocessor = ColumnTransformer([
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), CATEGORICAL),
        ("num", "passthrough", NUMERIC),
    ])

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", XGBClassifier(
            **PARAMS,
            eval_metric ="logloss",
            n_jobs      =-1,
            random_state=RANDOM_STATE,
        )),
    ])
    return pipeline


def train():
    """Full training run: load data, train, evaluate, save, log to MLflow."""

    # ── Data ──────────────────────────────────────────────────────────────────
    df = load_data()
    X_train, X_test, y_train, y_test = get_splits(df)

    # ── Train ─────────────────────────────────────────────────────────────────
    print("\nTraining XGBoost...")
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    print("Training complete.")

    # ── Evaluate ──────────────────────────────────────────────────────────────
    metrics = compute_metrics(pipeline, X_test, y_test)
    print_metrics(metrics, model_name="XGBoost")
    save_metrics(metrics)

    # ── Save model ────────────────────────────────────────────────────────────
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"\nModel saved: {MODEL_PATH}")

    # ── MLflow ────────────────────────────────────────────────────────────────
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("DelayPredict")

    with mlflow.start_run(run_name="XGBoost_train_py"):
        mlflow.log_params({
            "model"   : "XGBoost",
            "encoding": "OrdinalEncoder",
            **PARAMS,
            "train_size": X_train.shape[0],
            "test_size" : X_test.shape[0],
            "features"  : ", ".join(CATEGORICAL + NUMERIC),
        })
        mlflow.log_metrics({
            "accuracy" : metrics["Accuracy"],
            "precision": metrics["Precision"],
            "recall"   : metrics["Recall"],
            "f1"       : metrics["F1"],
            "roc_auc"  : metrics["ROC-AUC"],
        })
        mlflow.sklearn.log_model(pipeline, name="model")

    print("MLflow run logged — experiment: DelayPredict")


if __name__ == "__main__":
    train()