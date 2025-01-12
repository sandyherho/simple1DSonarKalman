#!/usr/bin/env Rscript

#' Sonar data processing using Kalman filter.
#' 
#' This script processes raw sonar data using a Kalman filter to estimate position 
#' and velocity. Results are saved to CSV files for further analysis.
#' 
#' Author: Sandy Herho (sandy.herho@email.ucr.edu)
#' Date: January 12, 2025

# Load required packages with error handling
suppressPackageStartupMessages({
  library(tidyverse)
  library(R.matlab)
})

#' Initialize Kalman filter parameters
#' 
#' @return A list containing all necessary Kalman filter parameters
#' @noRd
init_kalman_params <- function() {
  dt <- 0.02  # Time step (seconds)
  
  # State transition matrix
  A <- matrix(c(1, 0, dt, 1), nrow = 2)
  
  # Measurement matrix
  H <- matrix(c(1, 0), nrow = 1)
  
  # Process noise covariance
  Q <- matrix(c(1, 0, 0, 3), nrow = 2)
  
  # Measurement noise covariance
  R <- matrix(10)
  
  # Initial state
  x <- matrix(c(0, 20))
  
  # Initial state covariance
  P <- 5 * diag(2)
  
  list(
    A = A,
    H = H,
    Q = Q,
    R = R,
    x = x,
    P = P
  )
}

#' Update Kalman filter state
#' 
#' @param params List of current Kalman filter parameters
#' @param measurement New measurement value
#' @return List containing updated parameters and estimates
#' @noRd
kalman_update <- function(params, measurement) {
  # Prediction
  x_pred <- params$A %*% params$x
  P_pred <- params$A %*% params$P %*% t(params$A) + params$Q
  
  # Kalman gain
  S <- params$H %*% P_pred %*% t(params$H) + params$R
  K <- P_pred %*% t(params$H) %*% solve(S)
  
  # Update
  x_new <- x_pred + K %*% (measurement - params$H %*% x_pred)
  P_new <- P_pred - K %*% params$H %*% P_pred
  
  # Update parameters
  params$x <- x_new
  params$P <- P_new
  
  # Return estimates
  list(
    params = params,
    position = x_new[1],
    velocity = x_new[2],
    P = P_new
  )
}

#' Ensure directory exists and is writable
#' 
#' @param dir_path Path to check/create
#' @noRd
ensure_dir <- function(dir_path) {
  if (!dir.exists(dir_path)) {
    dir.create(dir_path, recursive = TRUE)
  }
}

#' Process sonar data using Kalman filter
#' 
#' @return NULL (saves results to files)
#' @noRd
process_data <- function() {
  tryCatch({
    # Setup paths
    base_path <- "../"
    raw_data_dir <- file.path(base_path, "raw_data")
    processed_data_dir <- file.path(base_path, "processed_data")
    
    # Print paths for debugging
    message("Raw data directory: ", raw_data_dir)
    message("Processed data directory: ", processed_data_dir)
    
    # Check input file
    input_file <- file.path(raw_data_dir, "SonarAlt.mat")
    if (!file.exists(input_file)) {
      stop("Input file not found: ", input_file)
    }
    
    # Create output directory
    ensure_dir(processed_data_dir)
    
    # Parameters
    n_samples <- 1500
    dt <- 0.02
    t <- seq(0, by = dt, length.out = n_samples)
    
    # Initialize storage matrices
    X_saved <- matrix(0, nrow = n_samples, ncol = 2)  # Position and velocity
    Z_saved <- numeric(n_samples)                      # Measurements
    P_saved <- matrix(0, nrow = n_samples, ncol = 2)  # State covariances
    
    # Load sonar data
    message("Reading MATLAB file...")
    mat_data <- readMat(input_file)
    measurements <- as.vector(mat_data$sonarAlt)
    message("Loaded ", length(measurements), " measurements")
    
    # Process data
    message("Processing data with Kalman filter...")
    params <- init_kalman_params()
    
    for (k in seq_len(n_samples)) {
      # Get measurement
      z <- measurements[k]
      
      # Update Kalman filter
      result <- kalman_update(params, z)
      params <- result$params
      
      # Save results
      X_saved[k, ] <- c(result$position, result$velocity)
      Z_saved[k] <- z
      P_saved[k, ] <- c(result$P[1,1], result$P[2,2])
    }
    
    # Calculate velocities using finite differences
    vel_meas <- diff(Z_saved) / dt
    vel_pos <- diff(X_saved[, 1]) / dt
    t_vel <- t[-length(t)]
    
    # Save results
    message("Saving results...")
    
    # Main results
    write_csv(
      tibble(
        time = t,
        raw_position = Z_saved,
        kalman_position = X_saved[, 1],
        kalman_velocity = X_saved[, 2]
      ),
      file.path(processed_data_dir, "kalman_estimates_R.csv")
    )
    
    # Covariance data
    write_csv(
      tibble(
        time = t,
        position_variance = P_saved[, 1],
        velocity_variance = P_saved[, 2]
      ),
      file.path(processed_data_dir, "kalman_covariances_R.csv")
    )
    
    # Velocity comparisons
    write_csv(
      tibble(
        time = t_vel,
        measurement_velocity = vel_meas,
        position_derivative = vel_pos,
        kalman_velocity = X_saved[-nrow(X_saved), 2]
      ),
      file.path(processed_data_dir, "velocity_comparisons_R.csv")
    )
    
    message("Processing completed successfully!")
    
  }, error = function(e) {
    message("\nError: ", e$message)
    quit(status = 1)
  })
}

# Main execution
process_data()
