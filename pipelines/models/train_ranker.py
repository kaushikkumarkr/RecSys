import pandas as pd
import xgboost as xgb
import mlflow
import pickle
import sys
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import ndcg_score, accuracy_score

sys.path.append(str(Path(__file__).resolve().parents[2]))

def train_ranker():
    print("Loading Ranker Training Data...")
    data_path = "data/processed/ranker_train.parquet"
    if not Path(data_path).exists():
        print(f"Error: {data_path} not found. Run make features first.")
        # Create dummy file if missing for verification flow?
        # No, failing is honest. But wait, verify flow runs features first.
        sys.exit(1)
        
    df = pd.read_parquet(data_path)
    print(f"Loaded {len(df)} samples.")
    
    if df.empty:
        print("Empty training data.")
        return

    # Feature Columns
    feature_cols = ['user_dot_item']
    X = df[feature_cols]
    y = df['label']
    
    # Split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train XGBoost
    print("Training XGBoost Ranker...")
    
    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "max_depth": 3,
        "eta": 0.1,
        "n_estimators": 100
    }
    
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    
    # Evaluate
    preds = model.predict(X_val)
    acc = accuracy_score(y_val, preds)
    print(f"Validation Accuracy: {acc:.4f}")
    
    # Save Model Locally
    with open("xgb_ranker.pkl", "wb") as f:
        pickle.dump(model, f)
    print("Model saved locally.")
    
    # MLflow Logging (Safe)
    active_run = None
    try:
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
        mlflow.set_experiment("ranking_model")
        active_run = mlflow.start_run(run_name="xgb_ranker_v1")
    except Exception as e:
        print(f"MLflow init failed: {e}")
        
    if active_run:
        try:
            with active_run:
                mlflow.log_params(params)
                mlflow.log_metric("accuracy", acc)
                mlflow.log_artifact("xgb_ranker.pkl")
                print("Model logged to MLflow.")
        except Exception as e:
            print(f"MLflow logging failed: {e}")

if __name__ == "__main__":
    train_ranker()
