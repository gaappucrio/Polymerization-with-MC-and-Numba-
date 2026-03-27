# Polymerization with Kinetic Monte Carlo and Numba

Access the paper at: https://dx.doi.org/10.1021/acs.iecr.0c01069

This repository provides a high-performance implementation of **Kinetic Monte Carlo (KMC)** simulations for polymerization processes, specifically focused on the thermal polymerization of styrene. The project explores optimization techniques in Python, comparing standard execution with Just-In-Time (JIT) compilation using **Numba** to achieve significant speedups.

## Project Overview

Kinetic Monte Carlo is a powerful stochastic method used to simulate the time evolution of chemical reactions. However, it is computationally intensive due to the massive number of discrete events and iterations required.

This project demonstrates a **General Method for Speeding Up Kinetic Monte Carlo Simulations** by:
1.  Implementing the core Gillespie algorithm (Stochastic Simulation Algorithm).
2.  Using **Numba** to compile Python code into machine code at runtime.
3.  Benchmarking performance across different implementations: Standard, Loop-optimized, and Numba-accelerated.

## Repository Structure & File Management

The repository contains approximately 200 files. To navigate them effectively, please note the following categorization:

### Core Scripts (The "Engines")
* `MC_Estireno_V2_Numba.py`: The high-performance version using `@njit` decorators. **(Recommended for use)**.
* `MC_Estireno.py`: The standard Python implementation (for benchmarking).
* `lib_MC.py`: The central library containing physical constants, reaction rates, and common functions.
* `Graf_comparison.py`: A utility script to visualize results and compare different simulation methods.

### Data and Results
The bulk of the files (~200) consist of:
* **Input/Parameter Files (`.txt`):** Files like `monomer.txt` and `initiator.txt` contain initial conditions.
* **Simulation Outputs:** Data files (e.g., `Moments_article.txt`, `conversion.txt`) containing the results of stochastic runs (moments, conversion, and concentrations).
* **Visualization Assets:** Graphical outputs in `.eps` and `.png` formats (e.g., `monomer-initiator.eps`).

## Prerequisites

To run the simulations, ensure you have the following installed:
* Python 3.8+
* **NumPy:** For numerical operations.
* **Matplotlib:** For generating plots.
* **Numba:** For Just-In-Time compilation.

Install dependencies via pip:
```bash
pip install numpy matplotlib numba
```

How to Use
1. Run the Optimized Simulation
To execute the fastest version of the simulation using Numba:

```bash
python MC_Estireno_V2_Numba.py
```

2. Run Benchmarking
To compare the standard Python loop performance against the optimized versions:

```bash
python MC_Estireno.py
```

3. Analyze Results
Use the comparison script to validate the stochastic results against the provided datasets:

```bash
python Graf_comparison.py
```
Performance Impact
By using the Numba-optimized approach, this project achieves execution speeds comparable to C++ while maintaining the flexibility of Python. This allows for the simulation of larger molecular systems and longer reaction times that would be computationally prohibitive in pure Python.

License
This project is licensed under the MIT License - see the LICENSE file for details.
