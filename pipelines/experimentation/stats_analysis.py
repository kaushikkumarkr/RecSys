"""
Statistical Analysis for A/B Tests.
Implements significance testing for experiment results.
"""
import numpy as np
from scipy import stats
from typing import Tuple, Dict
from dataclasses import dataclass


@dataclass
class ExperimentResult:
    """Results of a statistical test."""
    metric_name: str
    control_value: float
    treatment_value: float
    lift_percent: float
    p_value: float
    is_significant: bool
    confidence_level: float = 0.95


def calculate_ctr(impressions: int, clicks: int) -> float:
    """Calculate Click-Through Rate."""
    if impressions == 0:
        return 0.0
    return clicks / impressions


def chi_square_test(
    control_impressions: int,
    control_clicks: int,
    treatment_impressions: int,
    treatment_clicks: int,
    alpha: float = 0.05
) -> ExperimentResult:
    """
    Chi-square test for CTR comparison.
    Used when comparing proportions (clicks/impressions).
    """
    # Contingency table
    # [[control_clicks, control_no_click], [treatment_clicks, treatment_no_click]]
    control_no_click = control_impressions - control_clicks
    treatment_no_click = treatment_impressions - treatment_clicks
    
    table = np.array([
        [control_clicks, control_no_click],
        [treatment_clicks, treatment_no_click]
    ])
    
    # Chi-square test
    chi2, p_value, dof, expected = stats.chi2_contingency(table)
    
    control_ctr = calculate_ctr(control_impressions, control_clicks)
    treatment_ctr = calculate_ctr(treatment_impressions, treatment_clicks)
    
    lift = ((treatment_ctr - control_ctr) / control_ctr * 100) if control_ctr > 0 else 0
    
    return ExperimentResult(
        metric_name="CTR",
        control_value=control_ctr,
        treatment_value=treatment_ctr,
        lift_percent=lift,
        p_value=p_value,
        is_significant=p_value < alpha,
        confidence_level=1 - alpha
    )


def t_test(
    control_values: list,
    treatment_values: list,
    alpha: float = 0.05
) -> ExperimentResult:
    """
    Two-sample t-test for continuous metrics.
    Used for comparing means (e.g., revenue per user, time spent).
    """
    control_mean = np.mean(control_values)
    treatment_mean = np.mean(treatment_values)
    
    # Welch's t-test (doesn't assume equal variance)
    t_stat, p_value = stats.ttest_ind(control_values, treatment_values, equal_var=False)
    
    lift = ((treatment_mean - control_mean) / control_mean * 100) if control_mean > 0 else 0
    
    return ExperimentResult(
        metric_name="Mean",
        control_value=control_mean,
        treatment_value=treatment_mean,
        lift_percent=lift,
        p_value=p_value,
        is_significant=p_value < alpha,
        confidence_level=1 - alpha
    )


def calculate_sample_size(
    baseline_rate: float,
    minimum_detectable_effect: float,
    alpha: float = 0.05,
    power: float = 0.8
) -> int:
    """
    Calculate required sample size per variant.
    Uses power analysis for proportion comparison.
    """
    from scipy.stats import norm
    
    p1 = baseline_rate
    p2 = baseline_rate * (1 + minimum_detectable_effect)
    
    pooled_p = (p1 + p2) / 2
    
    z_alpha = norm.ppf(1 - alpha / 2)
    z_beta = norm.ppf(power)
    
    n = (2 * pooled_p * (1 - pooled_p) * (z_alpha + z_beta) ** 2) / ((p2 - p1) ** 2)
    
    return int(np.ceil(n))


def generate_experiment_report(results: list) -> str:
    """Generate a Markdown report for experiment results."""
    report = ["# Experiment Results Report\n"]
    report.append(f"Generated: {np.datetime64('today')}\n\n")
    
    for result in results:
        status = "✅ Significant" if result.is_significant else "⏳ Not Significant"
        report.append(f"## {result.metric_name}\n")
        report.append(f"| Metric | Control | Treatment | Lift | P-Value | Status |")
        report.append(f"|--------|---------|-----------|------|---------|--------|")
        report.append(
            f"| {result.metric_name} | {result.control_value:.4f} | "
            f"{result.treatment_value:.4f} | {result.lift_percent:+.2f}% | "
            f"{result.p_value:.4f} | {status} |"
        )
        report.append("\n")
    
    return "\n".join(report)


if __name__ == "__main__":
    # Demo: Chi-square test
    print("=== Chi-Square Test (CTR) ===")
    result = chi_square_test(
        control_impressions=10000,
        control_clicks=150,
        treatment_impressions=10000,
        treatment_clicks=180
    )
    print(f"Control CTR: {result.control_value:.4f}")
    print(f"Treatment CTR: {result.treatment_value:.4f}")
    print(f"Lift: {result.lift_percent:+.2f}%")
    print(f"P-Value: {result.p_value:.4f}")
    print(f"Significant: {result.is_significant}")
    
    # Demo: T-test
    print("\n=== T-Test (Revenue) ===")
    np.random.seed(42)
    control = np.random.normal(10, 2, 1000)
    treatment = np.random.normal(10.5, 2, 1000)
    result = t_test(control.tolist(), treatment.tolist())
    print(f"Control Mean: ${result.control_value:.2f}")
    print(f"Treatment Mean: ${result.treatment_value:.2f}")
    print(f"Lift: {result.lift_percent:+.2f}%")
    print(f"P-Value: {result.p_value:.4f}")
    print(f"Significant: {result.is_significant}")
    
    # Sample size calculation
    print("\n=== Sample Size Calculator ===")
    n = calculate_sample_size(baseline_rate=0.02, minimum_detectable_effect=0.1)
    print(f"Required sample per variant: {n:,}")
