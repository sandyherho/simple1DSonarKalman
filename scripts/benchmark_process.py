#!/usr/bin/env python
"""
Simple performance benchmarking script for Kalman filter implementations.
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
    process = psutil.Popen(cmd)
    peak_memory = 0
    
    try:
        while process.is_running():
            try:
                memory_info = process.memory_info()
                peak_memory = max(peak_memory, memory_info.rss / (1024 * 1024))  # MB
                time.sleep(0.1)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                break
    finally:
        try:
            process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
    
    exec_time = time.time() - start_time
    return exec_time, peak_memory

def run_benchmarks(n_warmup=2, n_runs=5):
    """Run benchmarks for all implementations."""
    # Use current directory for scripts
    current_dir = Path.cwd()
    
    commands = {
        'Python': ['python', current_dir / 'simpleKalman.py'],
        'Julia': ['julia', current_dir / 'simpleKalman.jl'],
        'R': ['Rscript', current_dir / 'simpleKalman.R']
    }
    
    results = []
    
    for lang, cmd in commands.items():
        print(f"\nBenchmarking {lang} implementation...")
        
        # Warm-up runs
        print(f"Warm-up runs...")
        for _ in range(n_warmup):
            get_performance_metrics(cmd)
        
        # Measurement runs
        print(f"Running {n_runs} measurements...")
        for i in range(n_runs):
            exec_time, peak_memory = get_performance_metrics(cmd)
            
            results.append({
                'language': lang,
                'run': i + 1,
                'exec_time_sec': exec_time,
                'peak_memory_mb': peak_memory
            })
            
            if (i + 1) % 100 == 0:
                print(f"Completed {i+1}/{n_runs} runs")
    
    return pd.DataFrame(results)

def main():
    print("Starting performance benchmarks...")
    
    # Create performance_data directory in parent directory
    output_dir = Path(__file__).parent.parent / 'performance_data'
    output_dir.mkdir(exist_ok=True)
    
    # Run benchmarks
    results = run_benchmarks()
    
    # Save results
    results.to_csv(output_dir / 'performance_data.csv', index=False)
    
    # Save summary
    summary = results.groupby('language').agg({
        'exec_time_sec': ['mean', 'std', 'min', 'max'],
        'peak_memory_mb': ['mean', 'std', 'min', 'max']
    }).round(4)
    
    summary.columns = [f"{col[0]}_{col[1]}" for col in summary.columns]
    summary.to_csv(output_dir / 'performance_summary.csv')
    
    print("\nBenchmarking completed!")
    print(f"Results saved to: {output_dir}")

if __name__ == "__main__":
    main()
