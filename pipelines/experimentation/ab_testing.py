"""
A/B Testing & Experimentation Module.
Provides deterministic user bucketing and experiment tracking.
"""
import hashlib
from datetime import datetime
from typing import Optional, List
from sqlalchemy import Column, String, Integer, Float, DateTime, Boolean, text
from sqlalchemy.ext.declarative import declarative_base

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from api.app.db import Base, engine

# --- Database Models ---

class Experiment(Base):
    """Tracks active A/B experiments."""
    __tablename__ = "experiments"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, unique=True, nullable=False)
    description = Column(String)
    control_variant = Column(String, default="control")
    treatment_variant = Column(String, default="treatment")
    traffic_percent = Column(Float, default=50.0)  # % of users in treatment
    start_date = Column(DateTime, default=datetime.utcnow)
    end_date = Column(DateTime, nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class ExperimentEvent(Base):
    """Logs events (impressions, clicks) per experiment."""
    __tablename__ = "experiment_events"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    experiment_name = Column(String, nullable=False)
    user_id = Column(String, nullable=False)
    variant = Column(String, nullable=False)  # 'control' or 'treatment'
    event_type = Column(String, nullable=False)  # 'impression', 'click', 'conversion'
    item_id = Column(String, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)


# --- Bucketing Logic ---

def get_user_bucket(user_id: str, experiment_name: str, num_buckets: int = 100) -> int:
    """
    Deterministic hash-based bucketing.
    Same user + experiment always returns same bucket (0-99).
    """
    key = f"{user_id}:{experiment_name}"
    hash_value = int(hashlib.md5(key.encode()).hexdigest(), 16)
    return hash_value % num_buckets


def get_variant(user_id: str, experiment_name: str, traffic_percent: float = 50.0) -> str:
    """
    Assign user to 'control' or 'treatment' based on bucket.
    traffic_percent: percentage of users in treatment group.
    """
    bucket = get_user_bucket(user_id, experiment_name)
    if bucket < traffic_percent:
        return "treatment"
    return "control"


def get_active_experiments(db_session) -> List[dict]:
    """Fetch all active experiments."""
    results = db_session.execute(
        text("SELECT name, traffic_percent FROM experiments WHERE is_active = true")
    ).fetchall()
    return [{"name": r.name, "traffic_percent": r.traffic_percent} for r in results]


def assign_user_to_experiments(user_id: str, db_session) -> dict:
    """
    Assign a user to all active experiments.
    Returns: {"experiment_name": "variant", ...}
    """
    experiments = get_active_experiments(db_session)
    assignments = {}
    for exp in experiments:
        assignments[exp["name"]] = get_variant(user_id, exp["name"], exp["traffic_percent"])
    return assignments


def log_experiment_event(
    experiment_name: str,
    user_id: str,
    variant: str,
    event_type: str,
    item_id: Optional[str] = None,
    db_session = None
):
    """Log an experiment event (impression, click, etc.)."""
    if db_session:
        db_session.execute(
            text("""
                INSERT INTO experiment_events (experiment_name, user_id, variant, event_type, item_id, timestamp)
                VALUES (:exp, :user, :variant, :event, :item, :ts)
            """),
            {
                "exp": experiment_name,
                "user": user_id,
                "variant": variant,
                "event": event_type,
                "item": item_id,
                "ts": datetime.utcnow()
            }
        )
        db_session.commit()


# --- Initialize Tables ---

def init_experiment_tables():
    """Create experiment tables if they don't exist."""
    Base.metadata.create_all(engine)
    print("Experiment tables created.")


if __name__ == "__main__":
    init_experiment_tables()
    
    # Demo: Create a sample experiment
    from api.app.db import SessionLocal
    
    session = SessionLocal()
    
    # Check if demo experiment exists
    existing = session.execute(
        text("SELECT name FROM experiments WHERE name = 'ranker_v2_test'")
    ).fetchone()
    
    if not existing:
        session.execute(
            text("""
                INSERT INTO experiments (name, description, traffic_percent, is_active)
                VALUES ('ranker_v2_test', 'Testing new XGBoost ranker vs baseline', 50.0, true)
            """)
        )
        session.commit()
        print("Created demo experiment: ranker_v2_test")
    
    # Demo bucketing
    for uid in ["U1", "U2", "U3", "U4", "U5"]:
        variant = get_variant(uid, "ranker_v2_test", 50.0)
        print(f"User {uid} -> {variant}")
    
    session.close()
