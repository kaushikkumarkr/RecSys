import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pickle
import datetime
from sqlalchemy import text
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[2]))
from api.app.db import engine, Base

def batch_score_users():
    print("Running Batch Scoring Job...")
    
    # 1. Load Model
    model_path = "xgb_ranker.pkl"
    if not Path(model_path).exists():
        print(f"Error: {model_path} not found. Train ranker first.")
        sys.exit(1)
        
    with open(model_path, "rb") as f:
        model = pickle.load(f)
        
    print("Loaded Ranker Model.")
    
    # 2. Get Candidates (All Users)
    # In prod, we would iterate users or load in batches.
    # For now, let's load all candidates.
    print("Fetching candidates...")
    df_candidates = pd.read_sql("SELECT user_id, item_id, strategy FROM recommendation_candidates", engine)
    
    # Filter out GLOBAL_POPULAR for separate handling or replicate them for every user?
    # Actually, we should only score "personalized" candidates if possible.
    # If a user has NO candidates, we should assign them Popular ones. 
    # But strictly, the Ranker needs User Features. GLOBAL_POPULAR user doesn't have features.
    
    # Let's filter to real users.
    real_candidates = df_candidates[df_candidates['user_id'] != 'GLOBAL_POPULAR'].copy()
    
    if real_candidates.empty:
        print("No personalized candidates found. Skipping reranking.")
        return

    print(f"Scoring {len(real_candidates)} candidates...")
    
    # 3. Build Features for Reranking
    # We need: User Vector, Item Vector -> Dot Product
    
    # Load Vectors (Cached)
    print("Loading vectors...")
    user_vecs = pd.read_sql("SELECT user_id, user_vector FROM user_features", engine)
    item_vecs = pd.read_sql("SELECT item_id, embedding FROM item_embeddings", engine)
    
    u_map = {r['user_id']: np.array(r['user_vector']) for _, r in user_vecs.iterrows()}
    i_map = {r['item_id']: np.array(r['embedding']) for _, r in item_vecs.iterrows()}
    
    # Compute Features
    features = []
    valid_indices = []
    
    for idx, row in tqdm(real_candidates.iterrows(), total=len(real_candidates)):
        uid = row['user_id']
        iid = row['item_id']
        
        u_vec = u_map.get(uid)
        i_vec = i_map.get(iid)
        
        if u_vec is not None and i_vec is not None:
            # Cosine Sim
            sim = np.dot(u_vec, i_vec) / (np.linalg.norm(u_vec) * np.linalg.norm(i_vec))
            features.append([sim])
            valid_indices.append(idx)
            
    # multidimensional array
    X_pred = np.array(features)
    
    # Predict
    print("Predicting scores...")
    scores = model.predict_proba(X_pred)[:, 1] # Probability of class 1
    
    # Assign back
    real_candidates.loc[valid_indices, 'prediction'] = scores
    
    # 4. Generate Final Table
    # Sort by User, Score DESC
    print("Ranking and generating explanations...")
    
    final_recs = []
    run_date = datetime.datetime.now()
    
    grouped = real_candidates.loc[valid_indices].groupby('user_id')
    
    for user_id, group in tqdm(grouped):
        top_k = group.sort_values('prediction', ascending=False).head(10)
        
        for rank, (idx, row) in enumerate(top_k.iterrows()):
            # Logic for Reason
            reasons = []
            if row['strategy'] == 'als':
                reasons.append("Based on your reading history")
            
            # If high dot product
            if row['prediction'] > 0.7:
                reasons.append("Matches your top categories")
                
            final_recs.append({
                "run_date": run_date,
                "user_id": user_id,
                "item_id": row['item_id'],
                "score": float(row['prediction']),
                "rank": rank + 1,
                "reasons": ", ".join(reasons)
            })
            
    # Write to DB
    # Clear old
    with engine.connect() as conn:
        conn.execute(text("DELETE FROM recommendation_daily_snapshot")) # For demo, keep only latest
        conn.commit()
        
    print(f"Writing {len(final_recs)} recommendations to DB...")
    pd.DataFrame(final_recs).to_sql("recommendation_daily_snapshot", engine, if_exists="append", index=False)
    print("Batch Scoring Complete.")

if __name__ == "__main__":
    batch_score_users()
