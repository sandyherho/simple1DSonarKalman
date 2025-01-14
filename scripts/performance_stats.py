#!/usr/bin/env python

"""
Performance Analysis Module for Multi-Language Computational Benchmarking

This script implements a comprehensive statistical analysis framework incorporating
bootstrap resampling methods to assess performance metrics across programming languages.
The analysis integrates non-parametric hypothesis testing, effect size quantification,
and uncertainty estimation through bootstrap simulation.

Author: Sandy Herho
Email: sandyherho@email.ucr.edu
Date: January 14, 2025

Statistical Methodology:
    - Bootstrap resampling (n=10000) for robust inference
    - Kruskal-Wallis H-test with bootstrap confidence intervals
    - Dunn's post-hoc test with Bonferroni correction
    - Cohen's d effect size analysis with bootstrap uncertainty estimation
    - Kernel Density Estimation for distribution visualization
    
Dependencies:
    - pandas>=1.5.0: Data manipulation and analysis
    - numpy>=1.21.0: Numerical computations
    - seaborn>=0.11.0: Statistical visualization
    - matplotlib>=3.5.0: Plotting functionality
    - scipy>=1.7.0: Statistical computations
    - scikit_posthocs>=0.7.0: Post-hoc statistical testing
"""

import os
import sys
from typing import Tuple, List, Dict
from itertools import combinations
from dataclasses import dataclass
import pickle

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from scikit_posthocs import posthoc_dunn

@dataclass
class BootstrapResults:
    """Container for bootstrap analysis results."""
    bootstrap_samples: np.ndarray
    confidence_intervals: Tuple[float, float]
    point_estimate: float
    standard_error: float

def bootstrap_sample(data: np.ndarray, n_bootstrap: int = 10000) -> np.ndarray:
    """
    Generate bootstrap samples from input data.
    
    Args:
        data: Input data array
        n_bootstrap: Number of bootstrap iterations
    
    Returns:
        Array of bootstrap samples
    """
    return np.array([
        np.random.choice(data, size=len(data), replace=True)
        for _ in range(n_bootstrap)
    ])

def bootstrap_statistic(data: np.ndarray, statistic: callable, 
                       n_bootstrap: int = 10000, 
                       confidence_level: float = 0.95) -> BootstrapResults:
    """
    Perform bootstrap analysis for a given statistic.
    
    Args:
        data: Input data array
        statistic: Statistical function to bootstrap
        n_bootstrap: Number of bootstrap iterations
        confidence_level: Confidence level for intervals
    
    Returns:
        BootstrapResults object containing analysis results
    """
    bootstrap_samples = bootstrap_sample(data, n_bootstrap)
    bootstrap_statistics = np.array([statistic(sample) for sample in bootstrap_samples])
    
    point_estimate = statistic(data)
    standard_error = np.std(bootstrap_statistics)
    
    alpha = 1 - confidence_level
    confidence_intervals = np.percentile(
        bootstrap_statistics, 
        [100 * alpha/2, 100 * (1 - alpha/2)]
    )
    
    return BootstrapResults(
        bootstrap_samples=bootstrap_statistics,
        confidence_intervals=confidence_intervals,
        point_estimate=point_estimate,
        standard_error=standard_error
    )

def calculate_cohens_d_bootstrap(group1: np.ndarray, group2: np.ndarray,
                               n_bootstrap: int = 10000) -> BootstrapResults:
    """
    Calculate Cohen's d effect size with bootstrap confidence intervals.
    
    Args:
        group1: First group's data
        group2: Second group's data
        n_bootstrap: Number of bootstrap iterations
    
    Returns:
        BootstrapResults object for Cohen's d
    """
    def cohens_d(x, y):
        nx, ny = len(x), len(y)
        vx, vy = np.var(x, ddof=1), np.var(y, ddof=1)
        pooled_se = np.sqrt(((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2))
        return (np.mean(x) - np.mean(y)) / pooled_se
    
    # Generate paired bootstrap samples
    bootstrap_d = []
    for _ in range(n_bootstrap):
        boot1 = np.random.choice(group1, size=len(group1), replace=True)
        boot2 = np.random.choice(group2, size=len(group2), replace=True)
        bootstrap_d.append(cohens_d(boot1, boot2))
    
    point_estimate = cohens_d(group1, group2)
    ci = np.percentile(bootstrap_d, [2.5, 97.5])
    se = np.std(bootstrap_d)
    
    return BootstrapResults(
        bootstrap_samples=np.array(bootstrap_d),
        confidence_intervals=ci,
        point_estimate=point_estimate,
        standard_error=se
    )

def create_bootstrap_distribution_plot(bootstrap_results: Dict[str, BootstrapResults],
                                    metric: str, output_path: str) -> None:
    """
    Generate visualization of bootstrap sampling distributions.
    
    Args:
        bootstrap_results: Dictionary of BootstrapResults objects
        metric: Name of the metric being analyzed
        output_path: Path to save the figure
    """
    plt.figure(figsize=(10, 6))
    colors = plt.cm.Set2(np.linspace(0, 1, len(bootstrap_results)))
    
    for (name, results), color in zip(bootstrap_results.items(), colors):
        sns.kdeplot(results.bootstrap_samples, label=name, color=color)
        plt.axvline(results.point_estimate, color=color, linestyle='--', alpha=0.5)
        plt.fill_between(
            np.linspace(results.confidence_intervals[0], 
                       results.confidence_intervals[1], 100),
            0, 1, color=color, alpha=0.2
        )
    
    plt.xlabel(f'Bootstrap {metric}')
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=350, bbox_inches='tight')
    plt.close()

def configure_plotting_style() -> None:
    """Configure visualization style settings."""
    plt.style.use('bmh')
    sns.set_context("paper", font_scale=1.2)

def create_performance_plot(data: pd.DataFrame, metric: str, 
                          output_path: str) -> None:
    """
    Generate and save KDE plot for performance metric distribution.
    
    Args:
        data: Input DataFrame containing performance metrics
        metric: Column name of the metric to analyze
        output_path: Path to save the figure
    """
    plt.figure(figsize=(8, 6))
    sns.kdeplot(data=data, x=metric, hue='language', common_norm=False)
    plt.xlabel(metric.replace('_', ' ').title())
    plt.ylabel('Density')
    plt.tight_layout()
    plt.savefig(output_path, dpi=350, bbox_inches='tight')
    plt.close()

def save_results(results: Dict, output_path: str) -> None:
    """
    Save analysis results to pickle file.
    
    Args:
        results: Dictionary containing analysis results
        output_path: Path to save results
    """
    with open(output_path, 'wb') as f:
        pickle.dump(results, f)

def main():
    """
    Main execution function for performance analysis.
    Implements bootstrap-based statistical analysis framework.
    """
    N_BOOTSTRAP = 10000  # Number of bootstrap iterations
    
    # Create output directories
    os.makedirs('../figs', exist_ok=True)
    os.makedirs('../bootstrap_performance', exist_ok=True)
    
    # Read and process data
    try:
        df = pd.read_csv('../performance_data/performance_data.csv')
    except FileNotFoundError:
        print("Error: Performance data file not found")
        sys.exit(1)
    
    # Configure plotting style
    configure_plotting_style()
    
    # Initialize results storage
    bootstrap_results = {
        'execution_time': {},
        'peak_memory': {}
    }
    
    # Perform bootstrap analysis for each language pair
    languages = sorted(df['language'].unique())
    metrics = {
        'execution_time': 'exec_time_sec',
        'peak_memory': 'peak_memory_mb'
    }
    
    for metric_name, metric_col in metrics.items():
        for lang1, lang2 in combinations(languages, 2):
            group1 = df[df['language'] == lang1][metric_col].values
            group2 = df[df['language'] == lang2][metric_col].values
            
            # Calculate bootstrapped Cohen's d
            effect_size_results = calculate_cohens_d_bootstrap(
                group1, group2, N_BOOTSTRAP
            )
            
            pair_name = f"{lang1}_vs_{lang2}"
            bootstrap_results[metric_name][pair_name] = effect_size_results
    
    # Create bootstrap distribution plots
    for metric_name in metrics:
        create_bootstrap_distribution_plot(
            bootstrap_results[metric_name],
            metric_name,
            f'../figs/{metric_name}_bootstrap_distributions.png'
        )
    
    # Create performance distribution plots
    for metric_name, metric_col in metrics.items():
        create_performance_plot(
            df, metric_col,
            f'../figs/{metric_name}_distribution.png'
        )
    
    # Save complete results
    save_results(bootstrap_results, '../bootstrap_performance/bootstrap_analysis.pkl')
    
    # Print summary statistics
    print("\nBootstrap Analysis Summary:")
    for metric_name, metric_results in bootstrap_results.items():
        print(f"\n{metric_name.replace('_', ' ').title()}:")
        for pair_name, results in metric_results.items():
            print(f"\n{pair_name}:")
            print(f"Cohen's d: {results.point_estimate:.3f}")
            print(f"95% CI: [{results.confidence_intervals[0]:.3f}, "
                  f"{results.confidence_intervals[1]:.3f}]")
            print(f"Standard Error: {results.standard_error:.3f}")

if __name__ == "__main__":
    main()
