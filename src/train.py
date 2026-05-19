"""
src/train.py
-------------
Trains the XGBoost model on a random 50% sample of the data.
The remaining 50% is used as simulated incoming data stream.
Also logs params and metrics to MLflow under the DelayPredict experiment.

Usage:
    python src/train.py                        # initial training on 50%
    python src/train.py --all-data             # retrain on all accumulated data
"""

import sys
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier

from data_preprocessing import load_data, CATEGORICAL, NUMERIC, FEATURES, TARGET
from evaluate import compute_metrics, print_metrics, save_metrics
from drift import save_training_reference

# ── Constants ─────────────────────────────────────────────────────────────────
MODEL_PATH      = Path("models/xgb_model.pkl")
BACKUP_PATH     = Path("models/xgb_model_backup.pkl")
STREAM_PATH     = Path("data/processed/stream_data.csv")   # accumulated new data
MLFLOW_URI      = "file:///app/notebooks/mlruns"
RANDOM_STATE    = 42
INITIAL_FRAC    = 0.5   # train on 50% initially

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


def train(retrain_with_stream: bool = False):
    """
    Train the model.
    - Initial run: random 50% of data
    - Retrain: original 50% + all accumulated stream data
    """

    # ── Load full dataset ──────────────────────────────────────────────────────
    df = load_data()

    # ── Initial 50% split ─────────────────────────────────────────────────────
    df_train_base = df.sample(frac=INITIAL_FRAC, random_state=RANDOM_STATE)
    df_stream_pool = df.drop(df_train_base.index)

    # Save stream pool for data_stream.py to use
    stream_pool_path = Path("data/processed/stream_pool.csv")
    stream_pool_path.parent.mkdir(parents=True, exist_ok=True)
    if not stream_pool_path.exists():
        df_stream_pool.to_csv(stream_pool_path, index=False)
        print(f"Stream pool saved: {stream_pool_path} ({len(df_stream_pool)} rows)")

    # ── Build training data ────────────────────────────────────────────────────
    if retrain_with_stream and STREAM_PATH.exists():
        df_stream = pd.read_csv(STREAM_PATH)
        df_combined = pd.concat([df_train_base, df_stream], ignore_index=True)
        run_name = f"XGBoost_retrain_{len(df_combined)}_rows"
        print(f"\nRetraining on {len(df_train_base)} base + {len(df_stream)} stream = {len(df_combined)} rows")
    else:
        df_combined = df_train_base
        run_name = f"XGBoost_initial_{len(df_combined)}_rows"
        print(f"\nInitial training on {len(df_combined)} rows (50% of dataset)")

    X_train = df_combined[FEATURES]
    y_train = df_combined[TARGET]

    # ── Test set — always the full held-out stream pool ────────────────────────
    X_test = df_stream_pool[FEATURES]
    y_test = df_stream_pool[TARGET]

    # ── Backup current model before overwriting ────────────────────────────────
    if MODEL_PATH.exists():
        import shutil
        shutil.copy(MODEL_PATH, BACKUP_PATH)
        print(f"Backup saved: {BACKUP_PATH}")

    # ── Train ──────────────────────────────────────────────────────────────────
    print("Training XGBoost...")
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    print("Training complete.")

    # ── Evaluate ──────────────────────────────────────────────────────────────
    metrics = compute_metrics(pipeline, X_test, y_test)
    print_metrics(metrics, model_name="XGBoost")
    save_metrics(metrics)

    # ── Save training reference for PSI drift detection ────────────────────────
    # Always use base 50% as reference — not combined data
    save_training_reference(df_train_base[FEATURES])

    # ── Save model ────────────────────────────────────────────────────────────
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"\nModel saved: {MODEL_PATH}")

    # ── MLflow ────────────────────────────────────────────────────────────────
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("DelayPredict")

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "model"        : "XGBoost",
            "encoding"     : "OrdinalEncoder",
            **PARAMS,
            "train_size"   : X_train.shape[0],
            "test_size"    : X_test.shape[0],
            "retrain"      : retrain_with_stream,
            "features"     : ", ".join(CATEGORICAL + NUMERIC),
        })
        mlflow.log_metrics({
            "accuracy" : metrics["Accuracy"],
            "precision": metrics["Precision"],
            "recall"   : metrics["Recall"],
            "f1"       : metrics["F1"],
            "roc_auc"  : metrics["ROC-AUC"],
        })
        mlflow.sklearn.log_model(pipeline, name="model")

    print(f"MLflow run logged: {run_name}")
    return metrics


if __name__ == "__main__":
    retrain = "--all-data" in sys.argv
    train(retrain_with_stream=retrain)