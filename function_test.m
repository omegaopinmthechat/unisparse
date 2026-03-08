clc; clear;
addpath('./Unisparse/','./Data generation/');

rng(2);
n = 120; p = 10;
[X, y, beta0_true, beta_true, Sigma] = Generate_data_scenario_homecourt(n, p);

rmps_options.DisplayUpdate = 0;
rmps_options.PrintSolution = 0;
rmps_options.MaxRuns       = 3;
rmps_options.TolFun2       = 1e-6;
rmps_options.cutoff        = 1e-6;

results = unisparse(X, y, [1e-3, 1e1], 5, 'all', [], [], [], rmps_options, 3.7, 3.0);

save('unisparse_cv_results.mat', 'results', '-v7.3');
fprintf('Done.\n');