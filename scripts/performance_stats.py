#!/usr/bin/env python

"""
Performance Analysis Module for Multi-Language Computational Benchmarking

This script performs statistical analysis of computational performance metrics
across programming languages using Kruskal-Wallis tests, Cohen's d effect size,
and visualization through violin plots.

Author: Sandy Herho
Email: sandy.herho@email.ucr.edu
Date: January 14, 2025
License: WTFPL
"""

import os
from typing import Dict, Tuple
from itertools import combinations

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from scikit_posthocs import posthoc_dunn

def calculate_cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    """
    Calculate Cohen's d effect size.
    
    Args:
        x: First sample
        y: Second sample
    
    Returns:
        Cohen's d value
    """
    nx, ny = len(x), len(y)
    vx, vy = np.var(x, ddof=1), np.var(y, ddof=1)
    
    # Pooled standard deviation
    pooled_sd = np.sqrt(((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2))
    
    # Handle zero standard deviation
    if pooled_sd == 0:
        return 0.0 if np.mean(x) == np.mean(y) else float('inf')
    
    return (np.mean(x) - np.mean(y)) / pooled_sd

def interpret_cohens_d(d: float) -> str:
    """Interpret Cohen's d effect size."""
    d = abs(d)
    if d < 0.2:
        return "negligible"
    elif d < 0.5:
        return "small"
    elif d < 0.8:
        return "medium"
    else:
        return "large"

def interpret_pvalue(p: float) -> str:
    """Interpret p-value significance."""
    if p < 0.001:
        return "highly significant"
    elif p < 0.01:
        return "very significant"
    elif p < 0.05:
        return "significant"
    else:
        return "not significant"

def create_violin_plots(data: pd.DataFrame, output_dir: str) -> None:
    """
    Create violin plots for execution time and memory usage.
    
    Args:
        data: DataFrame containing performance metrics
        output_dir: Directory to save plots
    """
    plt.style.use('bmh')
    
    # Execution Time Violin Plot
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=data, x='language', y='exec_time_sec')
    plt.xlabel('Programming Language')
    plt.ylabel('Execution Time (seconds)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/execution_time_box.png', dpi=350, bbox_inches='tight')
    plt.close()
    
    # Memory Usage Violin Plot
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=data, x='language', y='peak_memory_mb')
    plt.xlabel('Programming Language')
    plt.ylabel('Peak Memory (MB)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/peak_memory_box.png', dpi=350, bbox_inches='tight')
    plt.close()

def perform_statistical_analysis(data: pd.DataFrame) -> Dict:
    """
    Perform statistical analysis including Kruskal-Wallis and Cohen's d.
    
    Args:
        data: DataFrame containing performance metrics
    
    Returns:
        Dictionary containing analysis results
    """
    metrics = {
        'exec_time_sec': 'Execution Time',
        'peak_memory_mb': 'Peak Memory'
    }
    
    results = {}
    for col, name in metrics.items():
        # Kruskal-Wallis test
        groups = [group[col].values for _, group in data.groupby('language')]
        h_stat, p_value = stats.kruskal(*groups)
        
        # Dunn's post-hoc test
        dunn = posthoc_dunn(data, val_col=col, group_col='language', p_adjust='bonferroni')
        
        # Cohen's d for each pair
        languages = sorted(data['language'].unique())
        effect_sizes = {}
        for lang1, lang2 in combinations(languages, 2):
            group1 = data[data['language'] == lang1][col].values
            group2 = data[data['language'] == lang2][col].values
            d = calculate_cohens_d(group1, group2)
            effect_sizes[f"{lang1}_vs_{lang2}"] = d
        
        results[name] = {
            'kruskal_wallis': {'statistic': h_stat, 'p_value': p_value},
            'dunn': dunn,
            'cohens_d': effect_sizes
        }
    
    return results

def save_results_with_interpretation(results: Dict, output_dir: str) -> None:
    """
    Save statistical results with interpretation to CSV files.
    
    Args:
        results: Dictionary containing analysis results
        output_dir: Directory to save results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for metric_name, metric_results in results.items():
        # Kruskal-Wallis results
        kw_df = pd.DataFrame({
            'Metric': [metric_name],
            'H-statistic': [metric_results['kruskal_wallis']['statistic']],
            'p-value': [metric_results['kruskal_wallis']['p_value']],
            'Interpretation': [interpret_pvalue(metric_results['kruskal_wallis']['p_value'])]
        })
        kw_df.to_csv(f'{output_dir}/kruskal_wallis_{metric_name.lower().replace(" ", "_")}.csv', 
                     index=False)
        
        # Dunn's test results with interpretation
        dunn_df = metric_results['dunn'].copy()
        dunn_df.to_csv(f'{output_dir}/dunns_test_{metric_name.lower().replace(" ", "_")}.csv')
        
        # Cohen's d results with interpretation
        cohen_data = []
        for pair, d in metric_results['cohens_d'].items():
            cohen_data.append({
                'Comparison': pair,
                'Cohens_d': d,
                'Interpretation': interpret_cohens_d(d),
                'Effect Size': f"{d:.3f} ({interpret_cohens_d(d)})"
            })
        pd.DataFrame(cohen_data).to_csv(
            f'{output_dir}/cohens_d_{metric_name.lower().replace(" ", "_")}.csv',
            index=False
        )
        
        # Create interpretation summary
        with open(f'{output_dir}/interpretation_{metric_name.lower().replace(" ", "_")}.txt', 'w') as f:
            f.write(f"Statistical Analysis Summary for {metric_name}\n")
            f.write("=" * 50 + "\n\n")
            
            # Kruskal-Wallis interpretation
            kw_result = metric_results['kruskal_wallis']
            f.write("1. Kruskal-Wallis Test:\n")
            f.write(f"   H-statistic: {kw_result['statistic']:.3f}\n")
            f.write(f"   p-value: {kw_result['p_value']:.3e}\n")
            f.write(f"   Interpretation: {interpret_pvalue(kw_result['p_value'])}\n\n")
            
            # Effect size interpretation
            f.write("2. Effect Size Analysis (Cohen's d):\n")
            for pair, d in metric_results['cohens_d'].items():
                f.write(f"   {pair}:\n")
                f.write(f"   - Effect size: {d:.3f}\n")
                f.write(f"   - Interpretation: {interpret_cohens_d(d)}\n")
                
                # Add practical significance
                if abs(d) > 0.8:
                    f.write("   - Practical significance: Large and meaningful difference\n")
                elif abs(d) > 0.5:
                    f.write("   - Practical significance: Moderate and noticeable difference\n")
                else:
                    f.write("   - Practical significance: Small or negligible difference\n")
                f.write("\n")

def main():
    """Main execution function for performance analysis."""
    # Create output directories
    os.makedirs('../figs', exist_ok=True)
    os.makedirs('../analysis_results', exist_ok=True)
    
    # Read data
    df = pd.read_csv('../performance_data/performance_data.csv')
    
    # Create violin plots
    create_violin_plots(df, '../figs')
    
    # Perform statistical analysis
    results = perform_statistical_analysis(df)
    
    # Save results with interpretation
    save_results_with_interpretation(results, '../analysis_results')
    
    # Print summary to console
    print("\nAnalysis Summary:")
    for metric_name, metric_results in results.items():
        print(f"\n{metric_name}:")
        print(f"Kruskal-Wallis:")
        print(f"H-statistic: {metric_results['kruskal_wallis']['statistic']:.3f}")
        print(f"p-value: {metric_results['kruskal_wallis']['p_value']:.3e}")
        print(f"Interpretation: {interpret_pvalue(metric_results['kruskal_wallis']['p_value'])}")
        
        print("\nCohen's d:")
        for pair, d in metric_results['cohens_d'].items():
            print(f"{pair}: {d:.3f} ({interpret_cohens_d(d)})")

if __name__ == "__main__":
    main()
