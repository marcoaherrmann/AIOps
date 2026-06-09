"""
src/database.py
----------------
SQLite persistence layer via SQLAlchemy.
Tables: predictions, training_runs, drift_scores
"""

from datetime import datetime
from pathlib import Path

from sqlalchemy import Boolean, Column, DateTime, Float, Integer, String, create_engine
from sqlalchemy.orm import declarative_base, Session

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "delaypredict.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

engine = create_engine(f"sqlite:///{DB_PATH}", connect_args={"check_same_thread": False})
Base = declarative_base()


class Prediction(Base):
    __tablename__ = "predictions"
    id                = Column(Integer, primary_key=True, autoincrement=True)
    timestamp         = Column(DateTime, nullable=False)
    airline           = Column(String)
    airport_from      = Column(String)
    airport_to        = Column(String)
    day_of_week       = Column(Integer)
    departure_hour    = Column(Integer)
    length            = Column(Integer)
    delay_predicted   = Column(Boolean)
    delay_probability = Column(Float)


class TrainingRun(Base):
    __tablename__ = "training_runs"
    id            = Column(Integer, primary_key=True, autoincrement=True)
    timestamp     = Column(DateTime, nullable=False)
    run_type      = Column(String)       # "initial" | "retrain" | "progressive" | "incremental" | "auto"
    round         = Column(Integer, nullable=True)
    train_size    = Column(Integer)
    roc_auc       = Column(Float)
    f1            = Column(Float)
    accuracy      = Column(Float)
    precision     = Column(Float, nullable=True)
    recall        = Column(Float, nullable=True)
    max_psi       = Column(Float, nullable=True)
    worst_feature = Column(String, nullable=True)


class DriftScore(Base):
    __tablename__ = "drift_scores"
    id        = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False)
    feature   = Column(String)
    psi_score = Column(Float)


Base.metadata.create_all(engine)


def log_prediction(airline, airport_from, airport_to, day_of_week, departure_hour,
                   length, delay_predicted, delay_probability):
    with Session(engine) as s:
        s.add(Prediction(
            timestamp         = datetime.utcnow(),
            airline           = airline,
            airport_from      = airport_from,
            airport_to        = airport_to,
            day_of_week       = day_of_week,
            departure_hour    = departure_hour,
            length            = length,
            delay_predicted   = bool(delay_predicted),
            delay_probability = float(delay_probability),
        ))
        s.commit()


def log_training_run(run_type, train_size, roc_auc, f1, accuracy,
                     precision=None, recall=None, max_psi=None, worst_feature=None, round=None):
    with Session(engine) as s:
        s.add(TrainingRun(
            timestamp     = datetime.utcnow(),
            run_type      = run_type,
            round         = round,
            train_size    = int(train_size),
            roc_auc       = float(roc_auc),
            f1            = float(f1),
            accuracy      = float(accuracy),
            precision     = float(precision) if precision is not None else None,
            recall        = float(recall) if recall is not None else None,
            max_psi       = float(max_psi) if max_psi is not None else None,
            worst_feature = worst_feature,
        ))
        s.commit()


def log_drift_scores(psi_scores: dict):
    if not psi_scores:
        return
    ts = datetime.utcnow()
    with Session(engine) as s:
        for feature, score in psi_scores.items():
            s.add(DriftScore(timestamp=ts, feature=feature, psi_score=float(score)))
        s.commit()
