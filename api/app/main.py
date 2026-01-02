from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import text
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel
from typing import List, Optional
import os

from .db import get_db, engine, Base
from .models import Item, ItemEmbedding

app = FastAPI(title="RecSys + Semantic Search")

# Global model cache
model = None

def get_model():
    global model
    if model is None:
        model = SentenceTransformer('all-MiniLM-L6-v2')
    return model

class SearchResult(BaseModel):
    item_id: str
    title: str
    similarity: float
    category: str

@app.on_event("startup")
def startup():
    # Ensure tables exist
    Base.metadata.create_all(engine)
    # Load model on startup
    get_model()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/search", response_model=List[SearchResult])
def search(query: str, k: int = 10, db: Session = Depends(get_db)):
    model = get_model()
    # Embed query
    query_vector = model.encode(query).tolist()
    
    # Vector search using pgvector
    # <=> is cosine distance. 1 - distance = cosine similarity
    sql = text("""
        SELECT i.item_id, i.title, i.category, 
               1 - (e.embedding <=> :vector) as similarity
        FROM item_embeddings e
        JOIN items i ON e.item_id = i.item_id
        ORDER BY e.embedding <=> :vector
        LIMIT :k
    """)
    
    results = db.execute(sql, {"vector": str(query_vector), "k": k}).fetchall()
    
    return [
        SearchResult(
            item_id=row.item_id,
            title=row.title,
            category=row.category,
            similarity=row.similarity
        ) for row in results
    ]

class RecommendationRequest(BaseModel):
    user_id: str
    k: int = 10

class RecommendationResponse(BaseModel):
    item_id: str
    title: str
    category: str
    score: float
    rank: int
    reasons: Optional[str] = None
    strategy: Optional[str] = "batch"

@app.post("/recommend", response_model=List[RecommendationResponse])
def recommend(req: RecommendationRequest, db: Session = Depends(get_db)):
    # 1. Fetch from Recommendation Daily Snapshot (Batch Pre-computed)
    candidates = db.execute(text("""
        SELECT r.item_id, r.score, r.rank, r.reasons, i.title, i.category
        FROM recommendation_daily_snapshot r
        JOIN items i ON r.item_id = i.item_id
        WHERE r.user_id = :user_id
        ORDER BY r.rank ASC
        LIMIT :k
    """), {"user_id": req.user_id, "k": req.k}).fetchall()
    
    # 2. Fallback to Popular (computed in candidates) if batch missing
    if not candidates:
        candidates = db.execute(text("""
            SELECT c.item_id, c.score, c.rank, 'Trending now' as reasons, i.title, i.category
            FROM recommendation_candidates c
            JOIN items i ON c.item_id = i.item_id
            WHERE c.user_id = 'GLOBAL_POPULAR'
            ORDER BY c.rank ASC
            LIMIT :k
        """), {"k": req.k}).fetchall()
        
    return [
        RecommendationResponse(
            item_id=row.item_id,
            title=row.title,
            category=row.category,
            score=row.score,
            rank=row.rank,
            reasons=row.reasons,
            strategy="batch" if hasattr(row, 'reasons') and row.reasons != 'Trending now' else 'popular'
        ) for row in candidates
    ]
