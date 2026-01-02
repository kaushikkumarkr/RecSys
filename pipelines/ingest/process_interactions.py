import sys
from pathlib import Path
from sqlalchemy.orm import Session
from sqlalchemy import text
from tqdm import tqdm
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[2]))

from api.app.db import engine, SessionLocal
from api.app.models import RawBehavior, Interaction

def process_interactions():
    print("Processing raw behaviors into interactions...")
    session = SessionLocal()
    
    # Check if interactions already exist
    count = session.query(Interaction).count()
    if count > 0:
        print(f"Interactions table has {count} rows. Skipping (truncate to re-run).")
        session.close()
        return

    # Process in chunks using raw SQL for speed or sqlalchemy for simplicity. 
    # MIND dataset behaviors format: "time", "history", "impressions"
    # Impressions format: "N55689-1 N35729-0" (ItemID-ClickLabel)
    # We only want strictly positive interactions (clicks) for implicit CF training usually,
    # but for evaluation we might need negatives.
    # For this Sprint (Candidates), we primarily need POSITIVE interactions (Clicks) + History.
    
    # 1. Process History (Past Clicks)
    # History format: "N1234 N5678 ..."
    
    print("Processing User History...")
    # Fetch all raw behaviors
    # Using pandas read_sql might be faster for large datasets than ORM objects loop
    
    query_chunks = pd.read_sql("SELECT user_id, history, impressions, time FROM raw_behaviors", engine, chunksize=10000)
    
    interactions_to_add = []
    BATCH_SIZE = 50000
    
    total_inserted = 0
    
    for chunk_idx, df in enumerate(query_chunks):
        new_rows = []
        for _, row in df.iterrows():
            user_id = row['user_id']
            ts = pd.to_datetime(row['time'])
            
            # History interactions (Implicit Click) - Weight 1.0
            if row['history'] and str(row['history']) != 'nan':
                 for item_id in str(row['history']).split(' '):
                     new_rows.append({
                         "user_id": user_id,
                         "item_id": item_id,
                         "timestamp": ts, # History doesn't have partial timestamps, use behavior time
                         "event_type": "history", # distinct from fresh click
                         "weight": 1.0
                     })

            # Impression interactions (Real-time Clicks) - Weight 1.0 (or higher)
            if row['impressions'] and str(row['impressions']) != 'nan':
                for imp in str(row['impressions']).split(' '):
                    parts = imp.split('-')
                    if len(parts) == 2:
                        item_id, clicked = parts
                        if clicked == '1':
                            new_rows.append({
                                "user_id": user_id,
                                "item_id": item_id,
                                "timestamp": ts,
                                "event_type": "click",
                                "weight": 5.0 # Give more weight to recent clicks in session
                            })
        
        # Batch insert
        if new_rows:
            # Pandas is helpful for bulk insert but we can also use SQL
            # Let's use pandas to sql for simplicity in this script
            minidf = pd.DataFrame(new_rows)
            minidf.to_sql("interactions", engine, if_exists="append", index=False, method="multi")
            total_inserted += len(minidf)
            print(f"Processed chunk {chunk_idx}, inserted {len(minidf)} interactions. Total: {total_inserted}")
            
    print(f"Done! Total interactions: {total_inserted}")
    session.close()

if __name__ == "__main__":
    process_interactions()
