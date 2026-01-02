import sys
from pathlib import Path
import pandas as pd
import numpy as np
import json
from sqlalchemy import text
from tqdm import tqdm
import multiprocessing

sys.path.append(str(Path(__file__).resolve().parents[2]))
from api.app.db import engine

def parse_vector(val):
    if isinstance(val, str):
        try:
            return np.array(json.loads(val), dtype=np.float32)
        except:
            # Fallback for simple comma separated
            try:
                return np.array([float(x) for x in val.strip('[]').split(',')], dtype=np.float32)
            except:
                return np.zeros(384, dtype=np.float32)
    return np.array(val, dtype=np.float32)

def compute_user_profiles():
    print("Computing User Profiles (Avg Embedding)...")
    
    # 1. Fetch User History
    print("Fetching interactions for history...")
    # Get only positive history/clicks
    try:
        df_interactions = pd.read_sql("SELECT user_id, item_id FROM interactions WHERE weight > 0", engine)
    except:
         print("Interactions table not ready/empty.")
         return
    
    print("Fetching Item Embeddings...")
    try:
        df_embeddings = pd.read_sql("SELECT item_id, embedding FROM item_embeddings", engine)
    except:
         print("Item Embeddings missing.")
         return

    if df_interactions.empty or df_embeddings.empty:
        print("No data for profiles.")
        return
        
    # Create Item -> Vector map
    item_vec_map = {}
    for _, row in df_embeddings.iterrows():
        item_vec_map[row['item_id']] = parse_vector(row['embedding'])
        
    print(f"Loaded {len(item_vec_map)} item vectors.")
    
    # Group by User
    user_groups = df_interactions.groupby('user_id')['item_id'].apply(list)
    
    user_profiles = []
    
    print("Aggregating user vectors...")
    for user_id, history_items in tqdm(user_groups.items()):
        vectors = [item_vec_map[item] for item in history_items if item in item_vec_map]
        if vectors:
            avg_vec = np.mean(vectors, axis=0)
            user_profiles.append({
                "user_id": user_id,
                "user_vector": avg_vec.tolist() # Store as list for JSON/DB
            })
            
    # Save to DB (new table: user_features)
    # Using append workflow to preserve VECTOR type
    with engine.connect() as conn:
        conn.execute(text("DROP TABLE IF EXISTS user_features"))
        conn.execute(text("CREATE TABLE user_features (user_id TEXT PRIMARY KEY, user_vector VECTOR(384))"))
        conn.commit()
    
    # Store
    print("Storing User Features...")
    if user_profiles:
        df_profiles = pd.DataFrame(user_profiles)
        # We need to ensure 'user_vector' is formatted as a string/list that pgvector accepts via pandas?
        # Actually pandas to_sql might struggle with list -> vector.
        # It usually serializes lists to ARRAY or TEXT.
        # To strictly insert into VECTOR column, we might need to cast.
        # But psychopg2 supports list -> vector IF registered.
        # Let's try inserting. If pandas fails, we might need custom insert.
        
        # Convert lists to strings for safe insertion as string (pgvector casts it)
        df_profiles['user_vector'] = df_profiles['user_vector'].apply(lambda x: str(x))
        
        df_profiles.to_sql("user_features", engine, if_exists="append", index=False)
        
        # Create Index
        with engine.connect() as conn:
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_user_vector ON user_features USING hnsw (user_vector vector_cosine_ops)"))
            conn.commit()

def build_training_set():
    print("\nBuilding Ranker Training Set...")
    
    # 1. Fetch Raw Behaviors (Impressions)
    print("Fetching raw behaviors...")
    query = "SELECT user_id, impressions, time FROM raw_behaviors LIMIT 50000" # Limit for speed in Sprint
    try:
        df_beh = pd.read_sql(query, engine)
    except:
        print("Raw behaviors table missing? Skipping LTR training data build.")
        return
    
    if df_beh.empty:
        print("No behaviors found.")
        return

    dataset = []
    
    print("Parsing impressions for training data...")
    for _, row in tqdm(df_beh.iterrows()):
        user_id = row['user_id']
        impressions = str(row['impressions']).split(' ')
        
        for imp in impressions:
            parts = imp.split('-')
            if len(parts) == 2:
                item_id, label = parts
                dataset.append({
                    "user_id": user_id,
                    "item_id": item_id,
                    "label": int(label)
                })
    
    df_train = pd.DataFrame(dataset)
    print(f"Created training set with {len(df_train)} samples.")
    
    if df_train.empty:
        return

    # Feature Engineering
    
    # Load all user vectors
    try:
        df_user_vecs = pd.read_sql("SELECT user_id, user_vector FROM user_features", engine)
        user_vec_map = {r['user_id']: parse_vector(r['user_vector']) for _, r in df_user_vecs.iterrows()}
    except:
        print("User features missing.")
        user_vec_map = {}

    # Load all item vectors
    df_item_vecs = pd.read_sql("SELECT item_id, embedding FROM item_embeddings", engine)
    item_vec_map = {r['item_id']: parse_vector(r['embedding']) for _, r in df_item_vecs.iterrows()}
    
    print("Computing features...")
    
    # Add features to df_train
    feature_rows = []
    
    for _, row in tqdm(df_train.iterrows()):
        uid = row['user_id']
        iid = row['item_id']
        label = row['label']
        
        u_vec = user_vec_map.get(uid)
        i_vec = item_vec_map.get(iid)
        
        if u_vec is not None and i_vec is not None:
            # Cosine Sim
            sim = np.dot(u_vec, i_vec) / (np.linalg.norm(u_vec) * np.linalg.norm(i_vec))
            
            feature_rows.append({
                "user_id": uid,
                "item_id": iid,
                "label": label,
                "user_dot_item": float(sim),
            })
            
    final_df = pd.DataFrame(feature_rows)
    
    # Save to file for training script
    output_path = "data/processed/ranker_train.parquet"
    print(f"Saving {len(final_df)} samples to {output_path}")
    final_df.to_parquet(output_path)

if __name__ == "__main__":
    compute_user_profiles()
    build_training_set()
