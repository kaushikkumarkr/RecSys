import sys
from pathlib import Path
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sqlalchemy import text

sys.path.append(str(Path(__file__).resolve().parents[2]))

from api.app.db import engine, SessionLocal
from api.app.models import Item, ItemEmbedding

def generate_embeddings():
    print("Loading model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    db = SessionLocal()
    
    # Check if embeddings already exist
    count = db.query(ItemEmbedding).count()
    if count > 0:
        print(f"Found {count} embeddings. Skipping generation (truncate table to regenerate).")
        return

    print("Fetching items...")
    items = db.query(Item).all()
    print(f"Found {len(items)} items.")
    
    batch_size = 128
    
    # Create embeddings in batches
    for i in tqdm(range(0, len(items), batch_size)):
        batch_items = items[i:i+batch_size]
        texts = [f"{item.title}. {str(item.abstract or '')}" for item in batch_items]
        
        embeddings = model.encode(texts)
        
        db_embeddings = []
        for item, emb in zip(batch_items, embeddings):
            db_embeddings.append(ItemEmbedding(item_id=item.item_id, embedding=emb.tolist()))
        
        db.bulk_save_objects(db_embeddings)
        db.commit()
        
    # Create Index
    print("Creating HNSW index...")
    with engine.connect() as conn:
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_items_embedding ON item_embeddings USING hnsw (embedding vector_cosine_ops)"))
        conn.commit()
        
    print("Done!")

if __name__ == "__main__":
    generate_embeddings()
