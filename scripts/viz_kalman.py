#!/usr/bin/env python
"""
Visualization of Kalman filter results for sonar data.

This script creates publication-ready figures from processed Kalman filter data.

Author: Sandy Herho (sandy.herho@email.ucr.edu)
Date: January 12, 2025
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import seaborn as sns


def setup_plotting_style():
    """Configure plotting style for publication-quality figures."""
    plt.style.use("bmh")
    sns.set_palette("deep")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "axes.labelsize": 12,
            "axes.titlesize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.1,
        }
    )


def ensure_dir(directory):
    """Create directory if it doesn't exist."""
    Path(directory).mkdir(parents=True, exist_ok=True)


def plot_position_velocity(df, output_dir):
    """Create position and velocity plot."""
    fig, ax1 = plt.subplots(figsize=(8, 6))

    # Position plot
    ax2 = ax1.twinx()

    # Raw and filtered position
    ax1.plot(
        df["time"],
        df["raw_position"],
        "r.",
        markersize=4,
        alpha=0.5,
        label="Raw Measurements",
    )
    ax1.plot(
        df["time"],
        df["kalman_position"],
        "k-",
        linewidth=1.5,
        label="Position (Kalman)",
    )
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Position (m)")

    # Velocity
    ax2.plot(
        df["time"],
        df["kalman_velocity"],
        "b-",
        linewidth=1.5,
        label="Velocity (Kalman)",
    )
    ax2.set_ylabel("Velocity (m/s)")

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", framealpha=0.9)

    plt.savefig(output_dir / "position_velocity.png")
    plt.close()


def plot_covariances(df_cov, output_dir):
    """Create covariance plots."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Position covariance
    ax1.plot(df_cov["time"], df_cov["position_variance"], "k-", linewidth=1.5)
    ax1.set_ylabel("Position Variance")
    ax1.set_xlabel("Time (s)")

    # Velocity covariance
    ax2.plot(df_cov["time"], df_cov["velocity_variance"], "b-", linewidth=1.5)
    ax2.set_ylabel("Velocity Variance")
    ax2.set_xlabel("Time (s)")

    plt.savefig(output_dir / "covariances.png")
    plt.close()


def plot_velocity_comparison(df_vel, output_dir):
    """Create velocity comparison plot."""
    plt.figure(figsize=(8, 6))

    plt.plot(
        df_vel["time"],
        df_vel["measurement_velocity"],
        "r.",
        markersize=4,
        alpha=0.5,
        label="Measurement Derivative",
    )
    plt.plot(
        df_vel["time"],
        df_vel["position_derivative"],
        "k-",
        linewidth=1,
        alpha=0.7,
        label="Position Derivative",
    )
    plt.plot(
        df_vel["time"],
        df_vel["kalman_velocity"],
        "b-",
        linewidth=1.5,
        label="Kalman Estimate",
    )

    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (m/s)")
    plt.legend(framealpha=0.9)

    plt.savefig(output_dir / "velocity_comparison.png")
    plt.close()


def main():
    """Main function to create all plots."""
    # Setup paths
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / "processed_data"
    fig_dir = base_dir / "figs"

    # Ensure output directory exists
    ensure_dir(fig_dir)

    # Setup plotting style
    setup_plotting_style()

    # Load data
    df_main = pd.read_csv(data_dir / "kalman_estimates.csv")
    df_cov = pd.read_csv(data_dir / "kalman_covariances.csv")
    df_vel = pd.read_csv(data_dir / "velocity_comparisons.csv")

    # Create plots
    plot_position_velocity(df_main, fig_dir)
    plot_covariances(df_cov, fig_dir)
    plot_velocity_comparison(df_vel, fig_dir)


if __name__ == "__main__":
    main()
