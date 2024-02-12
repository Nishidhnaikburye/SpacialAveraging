# Fluid Dynamics Analysis Script

This Python script analyzes fluid dynamics simulation data from a CSV file using Pandas, NumPy, and Matplotlib. It performs various tasks, including extracting relevant columns, computing and visualizing mean velocity profiles, Reynolds stresses, turbulent viscosity, and more.

## Prerequisites

Ensure you have Python installed. Install required libraries:

```bash
pip install pandas numpy matplotlib

# Overview

The script performs the following tasks:

    Loads fluid dynamics simulation data from a CSV file into a Pandas DataFrame.
    Extracts relevant columns such as coordinates, velocity, Reynolds stresses, turbulent viscosity, etc.
    Computes and plots the mean velocity profile, Reynolds stresses, turbulent viscosity, and other parameters.
    Displays visualizations using Matplotlib.

## Parameters

    X, Y, Z: Coordinates from the simulation data.
    u_mean: Mean velocity profile.
    uprime2mean: Reynolds stresses (uv, uw).
    nut: Turbulent viscosity.
    wall_shear_stress: Wall shear stress.
    y_plus: Y Plus values.
    Nx, Ny, Nz: Grid dimensions.
    avgProf, avgProf_uv, avgProf_uw, avgProf_nut: Averaged profiles.
    k, csv_file: Parameters for data extraction and storage.

## Output

The script generates various visualizations representing different aspects of fluid dynamics, such as mean velocity profile, Reynolds stresses, turbulent viscosity, and more. These visualizations are displayed using Matplotlib.
