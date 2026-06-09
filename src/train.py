"""
src/train.py
-------------
Trains the XGBoost model on a random 50% sample of the data.
The remaining 50% is used as simulated incoming data stream.
Also logs params and metrics to MLflow under the DelayPredict experiment.

Usage:
    python src/train.py                        # initial training on 50%
    python src/train.py --all-data             # retrain on all accumulated data
    python src/train.py --incremental          # 90/10 split, 10 cumulative rounds
"""

import sys
import json
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
from database import log_training_run

# ── Constants ─────────────────────────────────────────────────────────────────
MODEL_PATH              = Path("models/xgb_model.pkl")
BACKUP_PATH             = Path("models/xgb_model_backup.pkl")
STREAM_PATH             = Path("data/processed/stream_data.csv")   # accumulated new data
INCREMENTAL_HISTORY_PATH = Path("data/processed/incremental_history.json")
MLFLOW_URI              = "file:///app/notebooks/mlruns"
RANDOM_STATE            = 42
INITIAL_FRAC            = 0.5   # train on 50% initially

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

    # ── Log to SQLite DB ───────────────────────────────────────────────────────
    try:
        log_training_run(
            run_type ="initial" if not retrain_with_stream else "retrain",
            train_size=X_train.shape[0],
            roc_auc  =metrics["ROC-AUC"],
            f1       =metrics["F1"],
            accuracy =metrics["Accuracy"],
            precision=metrics["Precision"],
            recall   =metrics["Recall"],
        )
    except Exception as e:
        print(f"[DB] Could not log training run: {e}")

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


def train_incremental(rounds: int = 10) -> list:
    """
    Incremental training with 90/10 split.

    - 90% of data → training pool, split into 10 cumulative rounds
    - 10% of data → fixed validation set, never touched during training
    - Each round trains on 10%, 20%, ..., 100% of the training pool
    - Evaluates every round against the same validation set
    - Results saved to data/processed/incremental_history.json

    Returns list of dicts with metrics per round.
    """

    print("\n" + "="*60)
    print("  Incremental Training — 90/10 Split, 10 Rounds")
    print("="*60)

    # ── Load full dataset ──────────────────────────────────────────────────────
    df = load_data()

    # ── 90/10 Split ────────────────────────────────────────────────────────────
    df_val        = df.sample(frac=0.1, random_state=RANDOM_STATE)           # 10% fixed validation
    df_train_pool = df.drop(df_val.index)                                     # 90% training pool

    X_val = df_val[FEATURES]
    y_val = df_val[TARGET]

    print(f"Training pool : {len(df_train_pool):,} rows")
    print(f"Validation set: {len(df_val):,} rows (fixed)")

    # ── MLflow setup ──────────────────────────────────────────────────────────
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("DelayPredict_Incremental")

    history = []

    # Clear history file immediately so the dashboard shows 0 rounds as soon as training starts
    INCREMENTAL_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    INCREMENTAL_HISTORY_PATH.write_text(json.dumps([], indent=2))

    # ── 10 Cumulative Rounds ───────────────────────────────────────────────────
    for round_num in range(1, rounds + 1):
        frac = round_num / rounds                          # 0.1, 0.2, ..., 1.0
        df_train = df_train_pool.sample(frac=frac, random_state=RANDOM_STATE)

        X_train = df_train[FEATURES]
        y_train = df_train[TARGET]

        print(f"\nRound {round_num:2}/{rounds} — training on {len(df_train):,} rows ({int(frac*100)}%)...")

        # ── Train ──────────────────────────────────────────────────────────────
        pipeline = build_pipeline()
        pipeline.fit(X_train, y_train)

        # ── Evaluate against fixed validation set ──────────────────────────────
        metrics = compute_metrics(pipeline, X_val, y_val)
        print_metrics(metrics, model_name=f"Round {round_num}")

        # ── Store result ───────────────────────────────────────────────────────
        result = {
            "round"      : round_num,
            "train_size" : len(df_train),
            "roc_auc"    : round(metrics["ROC-AUC"], 4),
            "f1"         : round(metrics["F1"], 4),
            "accuracy"   : round(metrics["Accuracy"], 4),
            "precision"  : round(metrics["Precision"], 4),
            "recall"     : round(metrics["Recall"], 4),
        }
        history.append(result)

        # Save partial results after each round so the API can report progress
        INCREMENTAL_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
        INCREMENTAL_HISTORY_PATH.write_text(json.dumps(history, indent=2))

        try:
            log_training_run(
                run_type ="incremental",
                round    =round_num,
                train_size=len(df_train),
                roc_auc  =metrics["ROC-AUC"],
                f1       =metrics["F1"],
                accuracy =metrics["Accuracy"],
                precision=metrics["Precision"],
                recall   =metrics["Recall"],
            )
        except Exception as e:
            print(f"[DB] Could not log incremental round: {e}")

        # ── Log to MLflow ──────────────────────────────────────────────────────
        with mlflow.start_run(run_name=f"incremental_round_{round_num}"):
            mlflow.log_params({
                "round"      : round_num,
                "train_size" : len(df_train),
                "val_size"   : len(df_val),
                "frac"       : frac,
                **PARAMS,
            })
            mlflow.log_metrics({
                "accuracy" : metrics["Accuracy"],
                "precision": metrics["Precision"],
                "recall"   : metrics["Recall"],
                "f1"       : metrics["F1"],
                "roc_auc"  : metrics["ROC-AUC"],
            })

    # ── Save final model (trained on 100% of pool) ─────────────────────────────
    if MODEL_PATH.exists():
        import shutil
        shutil.copy(MODEL_PATH, BACKUP_PATH)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"\nFinal model saved: {MODEL_PATH}")

    # ── Save history to disk ───────────────────────────────────────────────────
    INCREMENTAL_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    INCREMENTAL_HISTORY_PATH.write_text(json.dumps(history, indent=2))
    print(f"Incremental history saved: {INCREMENTAL_HISTORY_PATH}")

    print("\n" + "="*60)
    print(f"  Done — {rounds} rounds completed.")
    print("="*60)

    return history


if __name__ == "__main__":
    if "--incremental" in sys.argv:
        train_incremental()
    else:
        retrain = "--all-data" in sys.argv
        train(retrain_with_stream=retrain)