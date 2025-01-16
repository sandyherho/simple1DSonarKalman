#!/usr/bin/env julia
"""
Sonar data processing using Kalman filter.

This script processes raw sonar data using a Kalman filter to estimate position 
and velocity. Results are saved to CSV files for further analysis.

Author: Sandy Herho (sandy.herho@email.ucr.edu)
Date: January 12, 2025
License: WTFPL
"""

using MAT
using CSV
using DataFrames
using LinearAlgebra
using Dates

"""
Initialize Kalman filter parameters.

Returns:
    NamedTuple containing all necessary Kalman filter parameters
"""
function init_kalman_params()
    dt = 0.02  # Time step (seconds)
    
    # State transition matrix
    A = [1 dt; 0 1]
    
    # Measurement matrix
    H = [1 0]
    
    # Process noise covariance
    Q = [1 0; 0 3]
    
    # Measurement noise covariance
    R = [10]
    
    # Initial state
    x = [0; 20]
    
    # Initial state covariance
    P = 5 * Matrix(1.0I, 2, 2)
    
    return (;A, H, Q, R, x, P)
end

"""
Update Kalman filter state.

Args:
    params: Current Kalman filter parameters
    measurement: New measurement value

Returns:
    Tuple containing updated parameters and estimates (position, velocity, covariance)
"""
function kalman_update(params, measurement)
    # Unpack parameters
    (; A, H, Q, R, x, P) = params
    
    # Prediction
    x_pred = A * x
    P_pred = A * P * A' + Q
    
    # Kalman gain
    S = H * P_pred * H' .+ R
    K = P_pred * H' * inv(S)
    
    # Update
    x_new = x_pred + K * (measurement .- H * x_pred)
    P_new = P_pred - K * H * P_pred
    
    # Return updated parameters and estimates
    new_params = (; params..., x=x_new, P=P_new)
    pos = x_new[1]
    vel = x_new[2]
    
    return new_params, pos, vel, P_new
end

"""
Ensure directory exists, create if necessary.
"""
function ensure_dir(dir_path)
    isdir(dir_path) || mkpath(dir_path)
end

"""
Main processing function.
"""
function process_data()
    # Setup paths
    base_dir = dirname(dirname(abspath(@__FILE__)))
    raw_data_dir = joinpath(base_dir, "raw_data")
    processed_data_dir = joinpath(base_dir, "processed_data")
    
    # Ensure output directory exists
    ensure_dir(processed_data_dir)
    
    # Parameters
    n_samples = 1500
    dt = 0.02
    t = range(0, step=dt, length=n_samples)
    
    # Initialize arrays
    X_saved = zeros(n_samples, 2)  # Position and velocity
    Z_saved = zeros(n_samples)     # Measurements
    P_saved = zeros(n_samples, 2)  # State covariances
    
    # Load sonar data
    file = matread(joinpath(raw_data_dir, "SonarAlt.mat"))
    measurements = vec(file["sonarAlt"])
    
    # Process data
    params = init_kalman_params()
    
    for k in 1:n_samples
        # Get measurement
        z = measurements[k]
        
        # Update Kalman filter
        params, pos, vel, P = kalman_update(params, z)
        
        # Save results
        X_saved[k, :] = [pos, vel]
        Z_saved[k] = z
        P_saved[k, :] = [P[1,1], P[2,2]]
    end
    
    # Calculate velocities using finite differences
    vel_meas = diff(Z_saved) ./ dt
    vel_pos = diff(X_saved[:, 1]) ./ dt
    t_vel = t[1:end-1]
    
    # Save results to CSV
    # Main results
    df_main = DataFrame(
        time = t,
        raw_position = Z_saved,
        kalman_position = X_saved[:, 1],
        kalman_velocity = X_saved[:, 2]
    )
    CSV.write(joinpath(processed_data_dir, "kalman_estimates_jl.csv"), df_main)
    
    # Covariance data
    df_cov = DataFrame(
        time = t,
        position_variance = P_saved[:, 1],
        velocity_variance = P_saved[:, 2]
    )
    CSV.write(joinpath(processed_data_dir, "kalman_covariances_jl.csv"), df_cov)
    
    # Velocity comparisons
    df_vel = DataFrame(
        time = t_vel,
        measurement_velocity = vel_meas,
        position_derivative = vel_pos,
        kalman_velocity = X_saved[1:end-1, 2]
    )
    CSV.write(joinpath(processed_data_dir, "velocity_comparisons_jl.csv"), df_vel)
end

# Run the processing if this is the main script
if abspath(PROGRAM_FILE) == @__FILE__
    process_data()
end
