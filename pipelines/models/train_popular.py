import sys
import pandas as pd
from sqlalchemy import text
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))
from api.app.db import engine, Base

def generate_popularity_candidates():
    print("Generating Popularity Candidates...")
    
    # Simple popularity: Count of interactions per item
    # In production, we'd use time-decayed popularity (last 24h)
    # For this dataset, we'll take top 100 items by total clicks
    
    sql = """
    SELECT item_id, COUNT(*) as popularity_score
    FROM interactions
    GROUP BY item_id
    ORDER BY popularity_score DESC
    LIMIT 100
    """
    
    popular_items = pd.read_sql(sql, engine)
    print(f"Found {len(popular_items)} popular items.")
    
    # In a real system, we might store this "global" list once and cache it.
    # For consistency with the candidates table (user_id, item_id), 
    # we usually assign these to a specific placeholder user OR 
    # the matching logic in API falls back to this list if no ALS candidates found.
    # To keep candidates table clean, we will store these with user_id='GLOBAL_POPULAR'
    
    with engine.connect() as conn:
        conn.execute(text("DELETE FROM recommendation_candidates WHERE strategy = 'popular'"))
        conn.commit()
    
    candidates = []
    for rank, row in popular_items.iterrows():
        candidates.append({
            "user_id": "GLOBAL_POPULAR",
            "item_id": row['item_id'],
            "score": float(row['popularity_score']),
            "rank": rank + 1,
            "strategy": "popular"
        })
        
    pd.DataFrame(candidates).to_sql("recommendation_candidates", engine, if_exists="append", index=False)
    print("Popularity candidates stored for user 'GLOBAL_POPULAR'.")

if __name__ == "__main__":
    generate_popularity_candidates()
