"""
Fairness & Bias Auditing Module.
Measures recommendation fairness and provides diversity optimization.
"""
import numpy as np
from typing import List, Dict, Tuple
from collections import Counter
from dataclasses import dataclass


@dataclass
class FairnessReport:
    """Results of a fairness audit."""
    category_exposure: Dict[str, float]
    catalog_distribution: Dict[str, float]
    exposure_deviation: Dict[str, float]
    gini_coefficient: float
    popularity_bias_score: float


def calculate_gini(values: List[float]) -> float:
    """
    Calculate Gini coefficient for measuring inequality.
    0 = perfect equality, 1 = maximum inequality.
    
    Used to measure if recommendations are concentrated on few items (popularity bias).
    """
    if len(values) == 0:
        return 0.0
    
    sorted_values = sorted(values)
    n = len(sorted_values)
    cumulative_values = np.cumsum(sorted_values)
    
    gini = (2 * np.sum((np.arange(1, n + 1) * sorted_values))) / (n * cumulative_values[-1]) - (n + 1) / n
    return max(0, gini)


def calculate_category_exposure(
    recommendations: List[Dict],
    category_field: str = "category"
) -> Dict[str, float]:
    """
    Calculate exposure (% of recommendations) per category.
    """
    if not recommendations:
        return {}
    
    categories = [r.get(category_field, "unknown") for r in recommendations]
    counts = Counter(categories)
    total = len(categories)
    
    return {cat: count / total for cat, count in counts.items()}


def calculate_catalog_distribution(
    catalog: List[Dict],
    category_field: str = "category"
) -> Dict[str, float]:
    """
    Calculate distribution of items in catalog per category.
    """
    if not catalog:
        return {}
    
    categories = [item.get(category_field, "unknown") for item in catalog]
    counts = Counter(categories)
    total = len(categories)
    
    return {cat: count / total for cat, count in counts.items()}


def calculate_exposure_deviation(
    exposure: Dict[str, float],
    catalog: Dict[str, float]
) -> Dict[str, float]:
    """
    Calculate deviation between recommendation exposure and catalog distribution.
    Positive = over-represented, Negative = under-represented.
    """
    all_categories = set(exposure.keys()) | set(catalog.keys())
    deviation = {}
    
    for cat in all_categories:
        exp = exposure.get(cat, 0)
        cat_dist = catalog.get(cat, 0)
        deviation[cat] = exp - cat_dist
    
    return deviation


def calculate_popularity_bias(
    recommendations: List[Dict],
    item_popularity: Dict[str, int],
    item_id_field: str = "item_id"
) -> float:
    """
    Calculate average popularity percentile of recommended items.
    Higher = more biased towards popular items.
    """
    if not recommendations or not item_popularity:
        return 0.0
    
    # Calculate percentiles
    all_pops = sorted(item_popularity.values())
    
    rec_item_ids = [r.get(item_id_field) for r in recommendations]
    rec_pops = [item_popularity.get(iid, 0) for iid in rec_item_ids if iid]
    
    if not rec_pops:
        return 0.0
    
    # Average percentile of recommended items
    percentiles = []
    for pop in rec_pops:
        # Find percentile
        idx = np.searchsorted(all_pops, pop)
        percentile = idx / len(all_pops) * 100
        percentiles.append(percentile)
    
    return np.mean(percentiles)


def mmr_rerank(
    candidates: List[Dict],
    similarity_scores: List[float],
    item_embeddings: np.ndarray,
    lambda_param: float = 0.5,
    k: int = 10
) -> List[Dict]:
    """
    Maximal Marginal Relevance (MMR) re-ranking for diversity.
    
    MMR = λ * Sim(q, d) - (1-λ) * max(Sim(d, d_selected))
    
    Args:
        candidates: List of candidate items
        similarity_scores: Relevance scores for each candidate
        item_embeddings: Embeddings for computing item-item similarity
        lambda_param: Trade-off between relevance and diversity (0.5 = balanced)
        k: Number of items to return
    """
    if len(candidates) == 0:
        return []
    
    n = len(candidates)
    selected_indices = []
    remaining_indices = list(range(n))
    
    # Normalize similarity scores
    sim_scores = np.array(similarity_scores)
    if sim_scores.max() > 0:
        sim_scores = sim_scores / sim_scores.max()
    
    for _ in range(min(k, n)):
        mmr_scores = []
        
        for idx in remaining_indices:
            relevance = sim_scores[idx]
            
            if len(selected_indices) == 0:
                diversity_penalty = 0
            else:
                # Max similarity to already selected items
                selected_embeddings = item_embeddings[selected_indices]
                item_embedding = item_embeddings[idx]
                
                # Cosine similarity
                sims = np.dot(selected_embeddings, item_embedding) / (
                    np.linalg.norm(selected_embeddings, axis=1) * np.linalg.norm(item_embedding) + 1e-8
                )
                diversity_penalty = np.max(sims)
            
            mmr = lambda_param * relevance - (1 - lambda_param) * diversity_penalty
            mmr_scores.append((idx, mmr))
        
        # Select item with highest MMR score
        best_idx = max(mmr_scores, key=lambda x: x[1])[0]
        selected_indices.append(best_idx)
        remaining_indices.remove(best_idx)
    
    return [candidates[i] for i in selected_indices]


def generate_fairness_report(
    recommendations: List[Dict],
    catalog: List[Dict],
    item_popularity: Dict[str, int]
) -> FairnessReport:
    """Generate a comprehensive fairness audit report."""
    
    exposure = calculate_category_exposure(recommendations)
    catalog_dist = calculate_catalog_distribution(catalog)
    deviation = calculate_exposure_deviation(exposure, catalog_dist)
    
    # Gini on exposure values
    gini = calculate_gini(list(exposure.values())) if exposure else 0
    
    # Popularity bias
    pop_bias = calculate_popularity_bias(recommendations, item_popularity)
    
    return FairnessReport(
        category_exposure=exposure,
        catalog_distribution=catalog_dist,
        exposure_deviation=deviation,
        gini_coefficient=gini,
        popularity_bias_score=pop_bias
    )


def fairness_report_to_markdown(report: FairnessReport) -> str:
    """Convert fairness report to Markdown format."""
    lines = ["# Fairness Audit Report\n"]
    
    lines.append("## Category Exposure Analysis\n")
    lines.append("| Category | Exposure | Catalog | Deviation |")
    lines.append("|----------|----------|---------|-----------|")
    
    for cat in report.category_exposure.keys():
        exp = report.category_exposure.get(cat, 0)
        cat_dist = report.catalog_distribution.get(cat, 0)
        dev = report.exposure_deviation.get(cat, 0)
        
        dev_str = f"+{dev:.2%}" if dev > 0 else f"{dev:.2%}"
        lines.append(f"| {cat} | {exp:.2%} | {cat_dist:.2%} | {dev_str} |")
    
    lines.append("\n## Bias Metrics\n")
    lines.append(f"- **Gini Coefficient**: {report.gini_coefficient:.3f}")
    if report.gini_coefficient < 0.3:
        lines.append("  - ✅ Low inequality (diverse recommendations)")
    elif report.gini_coefficient < 0.6:
        lines.append("  - ⚠️ Moderate inequality")
    else:
        lines.append("  - ❌ High inequality (concentrated on few categories)")
    
    lines.append(f"- **Popularity Bias Score**: {report.popularity_bias_score:.1f}%")
    if report.popularity_bias_score > 70:
        lines.append("  - ⚠️ High popularity bias (recommending mostly popular items)")
    else:
        lines.append("  - ✅ Balanced popularity distribution")
    
    return "\n".join(lines)


if __name__ == "__main__":
    # Demo
    print("=== Fairness Audit Demo ===\n")
    
    # Sample data
    recommendations = [
        {"item_id": "N1", "category": "sports"},
        {"item_id": "N2", "category": "sports"},
        {"item_id": "N3", "category": "tech"},
        {"item_id": "N4", "category": "finance"},
        {"item_id": "N5", "category": "sports"},
    ]
    
    catalog = [
        {"item_id": f"N{i}", "category": cat}
        for i, cat in enumerate(["sports"] * 30 + ["tech"] * 40 + ["finance"] * 20 + ["music"] * 10)
    ]
    
    item_popularity = {f"N{i}": 100 - i for i in range(100)}
    
    report = generate_fairness_report(recommendations, catalog, item_popularity)
    print(fairness_report_to_markdown(report))
    
    # Gini demo
    print("\n=== Gini Coefficient Examples ===")
    print(f"Uniform [1,1,1,1]: {calculate_gini([1,1,1,1]):.3f}")
    print(f"Skewed [10,1,1,1]: {calculate_gini([10,1,1,1]):.3f}")
    print(f"Extreme [100,0,0,0]: {calculate_gini([100,0,0,0]):.3f}")
