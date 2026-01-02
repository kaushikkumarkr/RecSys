import pandas as pd
import sys
from pathlib import Path
from sqlalchemy import text

# Add parent directory to path to import api
sys.path.append(str(Path(__file__).resolve().parents[2]))

from api.app.db import engine, Base
from api.app.models import RawNews, RawBehavior, Item

DATA_DIR = Path("data/raw")

def load_news():
    print("Loading news.tsv...")
    df = pd.read_csv(
        DATA_DIR / "news.tsv",
        sep="\t",
        header=None,
        names=["item_id", "category", "subcategory", "title", "abstract", "url", "title_entities", "abstract_entities"],
        quoting=3 # Quote none to avoid errors
    )
    print(f"Read {len(df)} news items.")
    
    # Drop existing raw_news table and recreate
    print("Writing to Postres raw_news...")
    df.to_sql("raw_news", engine, if_exists="replace", index=False, chunksize=1000)
    
    # Also populate clean 'items' table
    print("Populating clean items table...")
    items_df = df[["item_id", "category", "subcategory", "title", "abstract", "url"]].copy()
    items_df.to_sql("items", engine, if_exists="replace", index=False, chunksize=1000)
    
def load_behaviors():
    print("Loading behaviors.tsv...")
    # Read in chunks as it might be large
    chunk_size = 100000
    reader = pd.read_csv(
        DATA_DIR / "behaviors.tsv",
        sep="\t",
        header=None,
        names=["impression_id", "user_id", "time", "history", "impressions"],
        chunksize=chunk_size,
        quoting=3
    )
    
    first = True
    for i, chunk in enumerate(reader):
        print(f"Writing behaviors chunk {i}...")
        if first:
            chunk.to_sql("raw_behaviors", engine, if_exists="replace", index=False)
            first = False
        else:
            chunk.to_sql("raw_behaviors", engine, if_exists="append", index=False)

def main():
    # Enable pgvector extension
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()

    # Create tables
    Base.metadata.create_all(engine)

    load_news()
    load_behaviors()
    print("Data loading complete.")

if __name__ == "__main__":
    main()
