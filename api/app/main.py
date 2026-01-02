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
    experiment_variant: Optional[str] = None  # A/B test variant

# --- Experiment Bucketing ---
import hashlib

def get_experiment_variant(user_id: str, experiment_name: str = "ranker_v2", traffic_percent: float = 50.0) -> str:
    """Deterministic hash-based bucketing for A/B tests."""
    key = f"{user_id}:{experiment_name}"
    bucket = int(hashlib.md5(key.encode()).hexdigest(), 16) % 100
    return "treatment" if bucket < traffic_percent else "control"

@app.post("/recommend", response_model=List[RecommendationResponse])
def recommend(req: RecommendationRequest, db: Session = Depends(get_db)):
    # Assign user to experiment
    variant = get_experiment_variant(req.user_id, "ranker_v2", 50.0)
    
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
    
    # Log experiment impression (would go to experiment_events table in production)
    # For now, we just include variant in response
        
    return [
        RecommendationResponse(
            item_id=row.item_id,
            title=row.title,
            category=row.category,
            score=row.score,
            rank=row.rank,
            reasons=row.reasons,
            strategy="batch" if hasattr(row, 'reasons') and row.reasons != 'Trending now' else 'popular',
            experiment_variant=variant
        ) for row in candidates
    ]

# --- RAG Chat Endpoint ---
import requests

MLX_SERVER_URL = os.getenv("MLX_SERVER_URL", "http://host.docker.internal:8502")

class ChatRequest(BaseModel):
    query: str
    k: int = 5  # Number of context items to retrieve

class ChatResponse(BaseModel):
    answer: str
    sources: List[str]

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, db: Session = Depends(get_db)):
    """
    RAG (Retrieval Augmented Generation) endpoint.
    1. Retrieves relevant items via vector search.
    2. Builds context from retrieved items.
    3. Calls MLX LLM for answer generation.
    """
    model = get_model()
    query_vector = model.encode(req.query).tolist()
    
    # Retrieve context
    sql = text("""
        SELECT i.item_id, i.title, i.category
        FROM item_embeddings e
        JOIN items i ON e.item_id = i.item_id
        ORDER BY e.embedding <=> :vector
        LIMIT :k
    """)
    results = db.execute(sql, {"vector": str(query_vector), "k": req.k}).fetchall()
    
    if not results:
        return ChatResponse(answer="No relevant articles found.", sources=[])
    
    # Build context string
    context_items = []
    sources = []
    for row in results:
        context_items.append(f"- {row.title} (Category: {row.category})")
        sources.append(row.item_id)
    
    context = "\n".join(context_items)
    
    # Build prompt
    prompt = f"""You are a helpful assistant for a news recommendation system, please answer based on the provided context. Answer concisely.

Context:
{context}

Question: {req.query}

Answer:"""
    
    # Call MLX Server
    try:
        response = requests.post(
            f"{MLX_SERVER_URL}/generate",
            json={"prompt": prompt, "max_tokens": 200},
            timeout=60
        )
        if response.status_code == 200:
            data = response.json()
            answer = data.get("response", data.get("error", "No response from LLM."))
        else:
            answer = f"LLM Error: {response.text}"
    except requests.exceptions.RequestException as e:
        # Fallback: Return context summary without LLM
        answer = f"(LLM unavailable) Based on search, relevant articles include: {', '.join([r.title[:50] for r in results[:3]])}"
    
    return ChatResponse(answer=answer, sources=sources)
