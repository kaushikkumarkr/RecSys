from sqlalchemy import Column, Integer, String, Text, DateTime, Float
from sqlalchemy.dialects.postgresql import ARRAY
from pgvector.sqlalchemy import Vector
from .db import Base

class Item(Base):
    __tablename__ = "items"
    
    item_id = Column(String, primary_key=True, index=True)
    category = Column(String)
    subcategory = Column(String)
    title = Column(Text)
    abstract = Column(Text)
    url = Column(String)
    
class ItemEmbedding(Base):
    __tablename__ = "item_embeddings"
    
    item_id = Column(String, primary_key=True)
    embedding = Column(Vector(384)) # 384 for all-MiniLM-L6-v2

class Interaction(Base):
    __tablename__ = "interactions"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String, index=True)
    item_id = Column(String, index=True)
    timestamp = Column(DateTime)
    event_type = Column(String) # 'click' or 'view'
    weight = Column(Float)
    
# Raw tables for direct loading
class RawNews(Base):
    __tablename__ = "raw_news"
    item_id = Column(String, primary_key=True)
    category = Column(String)
    subcategory = Column(String)
    title = Column(Text)
    abstract = Column(Text)
    url = Column(String)
    title_entities = Column(Text)
    abstract_entities = Column(Text)

class RawBehavior(Base):
    __tablename__ = "raw_behaviors"
    impression_id = Column(Integer, primary_key=True)
    user_id = Column(String, index=True)
    time = Column(String)
    history = Column(Text)
    impressions = Column(Text)

class RecommendationCandidate(Base):
    __tablename__ = "recommendation_candidates"
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String, index=True)
    item_id = Column(String)
    score = Column(Float)
    rank = Column(Integer)
    strategy = Column(String) # 'als', 'popular'

class RecommendationScore(Base):
    __tablename__ = "recommendation_daily_snapshot" # Using the table name from user requirements
    id = Column(Integer, primary_key=True, autoincrement=True)
    run_date = Column(DateTime)
    user_id = Column(String, index=True)
    item_id = Column(String)
    score = Column(Float)
    rank = Column(Integer)
    reasons = Column(Text) # JSON list or string
