import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import scipy.sparse as sparse
import implicit
import mlflow
import pickle
from tqdm import tqdm
from sqlalchemy import text, Column, String, Float, Integer

sys.path.append(str(Path(__file__).resolve().parents[2]))

from api.app.db import engine, Base, SessionLocal

# Define RecommendationCandidate model here or in models.py (better in models.py, but for speed putting logic here)
class RecommendationCandidate(Base):
    __tablename__ = "recommendation_candidates"
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String, index=True)
    item_id = Column(String)
    score = Column(Float)
    rank = Column(Integer)
    strategy = Column(String) # 'als', 'popular'

def train_cf_model():
    print("Loading interactions...")
    # Load user_items
    try:
        df = pd.read_sql("SELECT user_id, item_id, weight FROM interactions", engine)
    except Exception as e:
        print(f"Error loading interactions: {e}")
        return

    if df.empty:
        print("No interactions found. Exiting.")
        return

    # Map IDs to Indices
    print("Mapping IDs...")
    unique_users = df['user_id'].unique()
    unique_items = df['item_id'].unique()
    
    user_map = {u: i for i, u in enumerate(unique_users)}
    item_map = {i: u for u, i in enumerate(unique_items)} # Map Index -> ItemID
    item_to_idx = {u: i for i, u in enumerate(unique_items)}
    
    # Create CSR Matrix
    print("Creating sparse matrix...")
    user_ids = df['user_id'].map(user_map).values
    item_ids = df['item_id'].map(item_to_idx).values
    data = df['weight'].values
    
    # item_user matrix for Alternating Least Squares (implicit expects item_user)
    sparse_item_user = sparse.csr_matrix((data, (item_ids, user_ids)), shape=(len(unique_items), len(unique_users)))
    
    # Train Model
    print("Training ALS Model...")
    model = implicit.als.AlternatingLeastSquares(factors=64, regularization=0.05, alpha=2.0, iterations=15, random_state=42)
    model.fit(sparse_item_user)
    
    # Save Model locally first
    print("Saving model locally...")
    with open("als_model.pkl", "wb") as f:
        pickle.dump(model, f)
    
    # MLflow Logging (Safe)
    print("Logging to MLflow...")
    active_run = None
    try:
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
        mlflow.set_experiment("candidate_generation")
        active_run = mlflow.start_run(run_name="als_baseline")
    except Exception as e:
        print(f"MLflow init failed: {e}")
    
    if active_run:
        try:
            with active_run:
                mlflow.log_params({"factors": 64, "regularization": 0.05, "alpha": 2.0})
                mlflow.log_artifact("als_model.pkl")
        except Exception as e:
            print(f"MLflow logging failed: {e}")

    # Generate Candidates
    print("Generating candidates for stored users...")
    
    # Predict for all users (batch)
    user_items_matrix = sparse_item_user.T.tocsr() # User-Item matrix for recommend
    
    batch_size = 1000
    total_users = len(unique_users)
    
    # Ensure table exists
    Base.metadata.create_all(engine)
    
    # Clear old candidates
    with engine.connect() as conn:
        conn.execute(text("DELETE FROM recommendation_candidates WHERE strategy = 'als'"))
        conn.commit()
    
    print("Writing candidates to DB...")
    for start_idx in tqdm(range(0, total_users, batch_size)):
        end_idx = min(start_idx + batch_size, total_users)
        batch_user_indices = np.arange(start_idx, end_idx)
        
        try:
            # Wrap recommend in try/except to avoid IndexError
            ids, scores = model.recommend(batch_user_indices, user_items_matrix[batch_user_indices], N=50, filter_already_liked_items=True)
            
            batch_rows = []
            for i, user_idx in enumerate(batch_user_indices):
                if i >= len(ids): break # Safety
                
                real_user_id = unique_users[user_idx]
                user_recs = ids[i]
                user_scores = scores[i]
                
                for rank, (item_idx, score) in enumerate(zip(user_recs, user_scores)):
                    if item_idx in item_map:
                        real_item_id = item_map[item_idx]
                        batch_rows.append({
                            "user_id": real_user_id,
                            "item_id": real_item_id,
                            "score": float(score),
                            "rank": rank + 1,
                            "strategy": "als"
                        })
            
            # Bulk Insert
            if batch_rows:
                pd.DataFrame(batch_rows).to_sql("recommendation_candidates", engine, if_exists="append", index=False)
                
        except Exception as e:
            print(f"Error recommending for batch {start_idx}-{end_idx}: {e}")
            continue
            
    print("Training and Candidate Generation Complete!")

if __name__ == "__main__":
    train_cf_model()
