# Unisparse Toolbox Function Inventory

This file lists function declarations found in the source used by the toolbox project.

## Public Functions (callable by file name)

### Core UniSparse (folder: Unisparse)
- unisparse
- unisparse_univreg
- unilasso_objective_given_eta_loo
- uniMCP_objective_given_eta_loo
- uniSCAD_objective_given_eta_loo
- unibridge_objective_given_eta_loo

### Utilities (folder: supp funs)
- split_data
- compute_sparse_metrics

### Optimizer (folder: RMPSH)
- RMPSH

### Data Generation (folder: Data generation)
- Generate_data_scenario_1
- Generate_data_scenario_counterexample
- Generate_data_scenario_homecourt

### Other Methods (folder: other methods)
- fit_lasso
- fit_adap_lasso
- fit_mcp
- fit_mcp_LLA
- fit_scad
- fit_scad_LLA
- mcp_penalty
- mcp_derivative
- mcp_objective
- scad_penalty
- scad_derivative
- scad_objective

## Local/Internal Helper Functions

These are declared inside other files and are typically not called directly from outside.

### In Unisparse/unisparse.m
- cv_for_objective
- unisparse_cv_single
- cv_run_local

### In scenarios/scenario_1.m
- generate_scenario1_data
- summarize_unisparse_methods

## Count Summary
- Public function files: 24
- Local/internal helper functions: 5
- Total function declarations found: 29
