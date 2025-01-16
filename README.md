# Supporting Material for "Performance Analysis of 1D Linear Kalman Filter in Modern Scientific Computing Environments"

[![DOI](https://zenodo.org/badge/915642543.svg)](https://doi.org/10.5281/zenodo.14663442)
[![License: WTFPL](https://img.shields.io/badge/License-WTFPL-brightgreen.svg)](http://www.wtfpl.net/about/)
[![No Maintenance Intended](http://unmaintained.tech/badge.svg)](http://unmaintained.tech/)

This repository contains the code, data, and visualization outputs for benchmarking implementations of a 1D Linear Kalman Filter in modern scientific computing environments (Python, R, and Julia) applied to toy sonar altitude tracking. The project compares implementation performance across these platforms, analyzing execution time, memory usage, and estimation accuracy for real-time 1D sonar altitude measurements.

## Repository Structure
```plaintext
.
├── README.md                      # This file
├── LICENSE.txt                    # License information
├── raw_data/                      # Raw sonar measurement data
│   └── SonarAlt.mat              # MATLAB format sonar altitude data
├── processed_data/                # Processed Kalman filter outputs
│   ├── kalman_estimates.csv       # Position and velocity estimates
│   ├── kalman_covariances.csv    # State covariance data
│   └── velocity_comparisons.csv   # Velocity estimation comparisons
├── performance_data/              # Performance analysis data
│   ├── performance_data.csv       # Raw performance measurements
│   └── performance_summary.csv    # Statistical summary
├── analysis_results/              # Statistical analysis outputs
│   ├── kruskal_wallis_*.csv      # Kruskal-Wallis test results
│   ├── dunns_test_*.csv          # Dunn's post-hoc test results
│   ├── cohens_d_*.csv            # Effect size analysis
│   └── interpretation_*.txt       # Statistical interpretation
└── figs/                         # Generated figures
    ├── execution_time_box.png    # Runtime performance plots
    ├── peak_memory_box.png       # Memory usage plots
    ├── position_velocity.png     # Kalman filter estimates
    ├── covariances.png          # State covariances
    └── velocity_comparison.png   # Velocity estimation comparison
```

## Implementation Files
- **simpleKalman.py**: Python implementation using NumPy and SciPy
- **simpleKalman.R**: R implementation using tidyverse
- **simpleKalman.jl**: Julia implementation using LinearAlgebra and DataFrames
- **benchmark_process.py**: Performance benchmarking script
- **performance_stats.py**: Statistical analysis of performance data
- **viz_kalman.py**: Visualization generation script

## Citation
If you use this repository in your research, please cite:
```bibtex
@article{herho2024kalman,
  author = {Herho, Sandy},
  title = {{Performance Analysis of 1D Linear Kalman Filter in Modern Scientific Computing Environments}},
  journal = {TBD},
  year = {2025},
  volume = {TBD},
  number = {TBD},
  pages = {TBD},
  doi = {TBD}
}
```

## Usage
### Prerequisites
- **Python**: numpy, scipy, pandas, matplotlib, seaborn, scikit-posthocs, psutil
- **R**: tidyverse, R.matlab
- **Julia**: MAT, CSV, DataFrames, LinearAlgebra, Dates

### Data Processing
Run the Kalman filter implementations:
```bash
python simpleKalman.py
Rscript simpleKalman.R
julia simpleKalman.jl
```

### Performance Analysis
Run the benchmarking and analysis:
```bash
python benchmark_process.py  # Runs performance tests
python performance_stats.py  # Statistical analysis
```

### Visualization
Generate all visualization plots:
```bash
python viz_kalman.py  # Creates Kalman filter result plots
```

### Outputs
- Raw performance measurements are saved in `performance_data/`
- Statistical analysis results are saved in `analysis_results/`
- Visualization plots are saved in `figs/`

### Available Figures
- **Kalman Filter Results**:
  - `position_velocity.png`: Raw measurements and Kalman estimates
  - `covariances.png`: State covariance evolution
  - `velocity_comparison.png`: Velocity estimation comparison
- **Performance Analysis**:
  - `execution_time_box.png`: Runtime performance boxplots
  - `peak_memory_box.png`: Memory usage boxplots

## License
This repository is licensed under the WTFPL License - see [WTFPL](http://www.wtfpl.net/about/) for details.
