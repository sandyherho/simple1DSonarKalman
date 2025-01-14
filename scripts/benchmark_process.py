#!/usr/bin/env python
"""
Fast performance benchmarking script for Kalman filter implementations.
Measures execution time and memory usage for Python, Julia, and R implementations.

Author: Sandy Herho (sandy.herho@email.ucr.edu)
Date: January 12, 2025
"""

import subprocess
import time
import psutil
import pandas as pd
from pathlib import Path

def get_performance_metrics(cmd):
    """Measure execution time and peak memory usage."""
    start_time = time.time()
    process = subprocess.run(cmd, capture_output=True)
    exec_time = time.time() - start_time
    
    process_info = psutil.Process()
    peak_memory = process_info.memory_info().rss / (1024 * 1024)  # MB
    
    return exec_time, peak_memory

def run_benchmarks(n_warmup=5, n_runs=500):
    """Run benchmarks for all implementations."""
    current_dir = Path.cwd()
    
    commands = {
        'Python': ['python', current_dir / 'simpleKalman.py'],
        'Julia': ['julia', current_dir / 'simpleKalman.jl'],
        'R': ['Rscript', current_dir / 'simpleKalman.R']
    }
    
    results = []
    
    for lang, cmd in commands.items():
        print(f"\n{lang}: ", end="", flush=True)
        
        # Warm-up runs
        for _ in range(n_warmup):
            get_performance_metrics(cmd)
        print("warmup done, ", end="", flush=True)
        
        # Measurement runs
        for i in range(n_runs):
            exec_time, peak_memory = get_performance_metrics(cmd)
            
            results.append({
                'language': lang,
                'run': i + 1,
                'exec_time_sec': exec_time,
                'peak_memory_mb': peak_memory
            })
            
            if (i + 1) % 10 == 0:
                print(f"{i+1} ", end="", flush=True)
    
    return pd.DataFrame(results)

def main():
    # Create performance_data directory in parent directory
    output_dir = Path(__file__).parent.parent / 'performance_data'
    output_dir.mkdir(exist_ok=True)
    
    print("Benchmarking: 5 warmups, 500 runs per language")
    
    # Run benchmarks
    results = run_benchmarks()
    
    # Save detailed results
    results.to_csv(output_dir / 'performance_data.csv', index=False)
    
    # Calculate and save summary with median
    summary = results.groupby('language').agg({
        'exec_time_sec': ['median', 'mean', 'std', 'min', 'max'],
        'peak_memory_mb': ['median', 'mean', 'std', 'min', 'max']
    }).round(4)
    
    # Reorder columns to put median first
    for col in summary.columns.levels[0]:
        summary[col] = summary[col][['median', 'mean', 'std', 'min', 'max']]
    
    summary.columns = [f"{col[0]}_{col[1]}" for col in summary.columns]
    summary.to_csv(output_dir / 'performance_summary.csv')
    
    # Print summary in a more readable format
    print("\n\nSummary of results:")
    print("\nExecution Time (seconds):")
    exec_time_summary = summary[[col for col in summary.columns if 'exec_time' in col]]
    print(exec_time_summary)
    
    print("\nPeak Memory Usage (MB):")
    memory_summary = summary[[col for col in summary.columns if 'peak_memory' in col]]
    print(memory_summary)
    
    print(f"\nDetailed results saved to: {output_dir}")

if __name__ == "__main__":
    main()
