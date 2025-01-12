#!/usr/bin/env python
"""
Sonar data processing using Kalman filter.

This script processes raw sonar data using a Kalman filter to estimate position 
and velocity. Results are saved to CSV files for further analysis.

Author: Sandy Herho (sandy.herho@email.ucr.edu)
Date: January 12, 2025
"""

import numpy as np
from scipy import io
import os
from pathlib import Path
import pandas as pd

class KalmanFilter:
    """Implementation of Kalman filter for position and velocity estimation."""
    
    def __init__(self):
        self.dt = 0.02  # Time step (seconds)
        
        # State transition matrix
        self.A = np.array([[1, self.dt],
                          [0, 1]])
        
        # Measurement matrix
        self.H = np.array([[1, 0]])
        
        # Process noise covariance
        self.Q = np.array([[1, 0],
                          [0, 3]])
        
        # Measurement noise covariance
        self.R = np.array([[10]])
        
        # Initial state
        self.x = np.array([[0],
                          [20]])
        
        # Initial state covariance
        self.P = 5 * np.eye(2)

    def update(self, measurement):
        """
        Update state estimate using Kalman filter algorithm.
        
        Args:
            measurement (np.ndarray): Current measurement value
            
        Returns:
            tuple: Position, velocity, and covariance matrix
        """
        # Prediction
        x_pred = self.A @ self.x
        P_pred = self.A @ self.P @ self.A.T + self.Q

        # Kalman gain
        S = self.H @ P_pred @ self.H.T + self.R
        K = P_pred @ self.H.T @ np.linalg.inv(S)

        # Update
        self.x = x_pred + K @ (measurement - self.H @ x_pred)
        self.P = P_pred - K @ self.H @ P_pred

        return self.x[0, 0], self.x[1, 0], self.P

class SonarData:
    """Handler for sonar measurement data."""
    
    def __init__(self, filename):
        """
        Load sonar data from MATLAB file.
        
        Args:
            filename (str): Path to the MATLAB data file
        """
        mat_data = io.loadmat(filename)
        self.measurements = mat_data['sonarAlt'].flatten()
        self.current_idx = 0
    
    def get_measurement(self):
        """Get next measurement from the dataset."""
        if self.current_idx >= len(self.measurements):
            return None
        measurement = self.measurements[self.current_idx]
        self.current_idx += 1
        return measurement

def ensure_dir(directory):
    """Create directory if it doesn't exist."""
    Path(directory).mkdir(parents=True, exist_ok=True)

def process_data():
    """Process sonar data and save results."""
    # Setup paths
    base_dir = Path(__file__).parent.parent
    raw_data_dir = base_dir / 'raw_data'
    processed_data_dir = base_dir / 'processed_data'
    
    # Ensure output directory exists
    ensure_dir(processed_data_dir)
    
    # Parameters
    n_samples = 1500
    dt = 0.02
    t = np.arange(0, n_samples * dt, dt)

    # Initialize arrays
    X_saved = np.zeros((n_samples, 2))  # Position and velocity
    Z_saved = np.zeros(n_samples)       # Measurements
    P_saved = np.zeros((n_samples, 2))  # State covariances

    # Process data
    sonar = SonarData(raw_data_dir / 'SonarAlt.mat')
    kf = KalmanFilter()

    for k in range(n_samples):
        z = sonar.get_measurement()
        if z is None:
            break
            
        pos, vel, P = kf.update(np.array([[z]]))

        X_saved[k, :] = [pos, vel]
        Z_saved[k] = z
        P_saved[k, :] = [P[0, 0], P[1, 1]]
    
    # Calculate velocities using finite differences
    vel_meas = np.diff(Z_saved) / dt
    vel_pos = np.diff(X_saved[:, 0]) / dt
    t_vel = t[:-1]
    
    # Save results
    # Main results
    df_main = pd.DataFrame({
        'time': t,
        'raw_position': Z_saved,
        'kalman_position': X_saved[:, 0],
        'kalman_velocity': X_saved[:, 1]
    })
    df_main.to_csv(processed_data_dir / 'kalman_estimates.csv', index=False)
    
    # Covariance data
    df_cov = pd.DataFrame({
        'time': t,
        'position_variance': P_saved[:, 0],
        'velocity_variance': P_saved[:, 1]
    })
    df_cov.to_csv(processed_data_dir / 'kalman_covariances.csv', index=False)
    
    # Velocity comparisons
    df_vel = pd.DataFrame({
        'time': t_vel,
        'measurement_velocity': vel_meas,
        'position_derivative': vel_pos,
        'kalman_velocity': X_saved[:-1, 1]
    })
    df_vel.to_csv(processed_data_dir / 'velocity_comparisons.csv', index=False)

if __name__ == "__main__":
    process_data()
