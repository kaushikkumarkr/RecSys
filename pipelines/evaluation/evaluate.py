import pandas as pd
import numpy as np
from sklearn.metrics import ndcg_score
from sqlalchemy import text
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))
from api.app.db import engine

def dcg_at_k(r, k):
    r = np.asfarray(r)[:k]
    if r.size:
        return np.sum(np.subtract(np.power(2, r), 1) / np.log2(np.arange(2, r.size + 2)))
    return 0.

def ndcg_at_k(r, k):
    idcg = dcg_at_k(sorted(r, reverse=True), k)
    if not idcg:
        return 0.
    return dcg_at_k(r, k) / idcg

def precision_at_k(r, k):
    r = np.asarray(r)[:k]
    return np.mean(r)

def recall_at_k(r, k, total_relevant):
    if total_relevant == 0: return 0.
    r = np.asarray(r)[:k]
    return np.sum(r) / total_relevant

def evaluate_models():
    print("Running Offline Evaluation...")
    # Fetch Held-out test data (e.g. last day interactions)
    # For Sprint, we'll simulate by fetching recent interactions
    # Assume 'interactions' table has all data.
    
    print("Fetching test interactions (simulated sample)...")
    # In real prod: WHERE timestamp > X
    df_test = pd.read_sql("SELECT user_id, item_id, weight FROM interactions LIMIT 20000", engine)
    
    # Ground Truth: user -> set(relevant_items)
    ground_truth = df_test[df_test['weight'] > 0].groupby('user_id')['item_id'].apply(set).to_dict()
    
    test_users = list(ground_truth.keys())[:100] # Evaluat 100 users for speed
    print(f"Evaluating on {len(test_users)} users...")
    
    # Evaluate Candidates (Stage 1) vs Ranker (Stage 2)
    # Note: Ranker needs to run on candidates. We need to fetch candidates for these users.
    
    # Ensure we have candidates generated (from Sprint 2)
    # Join with features? For sim, just use score in candidates table as proxy for Unranked
    
    metrics = {"ndcg@10": [], "precision@10": [], "recall@10": []}
    
    for user in test_users:
        relevant_items = ground_truth[user]
        if not relevant_items: continue
        
        # Get System Predictions (Candidates)
        # We check both ALS and Popular strategies
        
        # 1. Fetch Candidates (simulate retrieval)
        with engine.connect() as conn:
            rows = conn.execute(text("SELECT item_id, score FROM recommendation_candidates WHERE user_id = :u ORDER BY rank ASC LIMIT 50"), {"u": user}).fetchall()
            
        if not rows:
             # Fallback to Popular
             rows = conn.execute(text("SELECT item_id, score FROM recommendation_candidates WHERE strategy = 'popular' LIMIT 50")).fetchall()
             
        # Convert to relevancy vector (1 if relevant, 0 else)
        preds = [r.item_id for r in rows]
        relevance_vector = [1 if item in relevant_items else 0 for item in preds]
        
        # Calculate Metrics
        ndcg = ndcg_at_k(relevance_vector, 10)
        p_at_10 = precision_at_k(relevance_vector, 10)
        r_at_10 = recall_at_k(relevance_vector, 10, len(relevant_items))
        
        metrics["ndcg@10"].append(ndcg)
        metrics["precision@10"].append(p_at_10)
        metrics["recall@10"].append(r_at_10)
        
    # Aggregate
    print("\n--- Evaluation Results ---")
    print(f"Mean NDCG@10: {np.mean(metrics['ndcg@10']):.4f}")
    print(f"Mean Precision@10: {np.mean(metrics['precision@10']):.4f}")
    print(f"Mean Recall@10: {np.mean(metrics['recall@10']):.4f}")
    
    # Generate Report
    report_path = "reports/evaluation_results.md"
    Path("reports").mkdir(exist_ok=True)
    with open(report_path, "w") as f:
        f.write("# Offline Evaluation Report\n\n")
        f.write("| Metric | Score |\n")
        f.write("|---|---|\n")
        f.write(f"| NDCG@10 | {np.mean(metrics['ndcg@10']):.4f} |\n")
        f.write(f"| Precision@10 | {np.mean(metrics['precision@10']):.4f} |\n")
        f.write(f"| Recall@10 | {np.mean(metrics['recall@10']):.4f} |\n")
        
    print(f"Report saved to {report_path}")

if __name__ == "__main__":
    evaluate_models()
