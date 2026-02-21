%% ================== Griewank (10D) using RMPSH ==================

clear; clc;

rng(1);   % reproducibility

% ------------------- Problem dimension -------------------
M = 10;

% ------------------- Bounds -------------------
lb = -600 * ones(M,1);
ub =  600 * ones(M,1);

% ------------------- Starting point -------------------
x0 = 200*(2*rand(M,1)-1);   % Wide uniform

% ------------------- Griewank function -------------------
griewank = @(x) ...
    1 + sum(x.^2)/4000 - prod(cos(x ./ sqrt((1:M)')));

fprintf('\nGriewank at start = %.6f\n', griewank(x0));

% ------------------- RMPSH parameters -------------------
MaxTime  = 60;        % seconds
MaxRuns  = 200;
MaxIter  = 5000;
sInitial = 1;
rho      = 2;
rho2     = 2;
TolFun1  = 1e-6;
TolFun2  = 1e-20;
phi      = 1e-20;
lambda   = 1e-20;
DisplayUpdate = 1;
DisplayEvery  = 2;
PrintSolution = 1;

% ------------------- Run RMPSH -------------------
[x_opt, fval, comp_time] = RMPSH( ...
    griewank, x0, lb, ub, ...
    MaxTime, MaxRuns, MaxIter, ...
    sInitial, rho, rho2, ...
    TolFun1, TolFun2, ...
    phi, lambda, DisplayUpdate, DisplayEvery, PrintSolution);

% ------------------- True optimum -------------------
x_true = zeros(M,1);
f_true = 0;

fprintf('\n========== GRIEWANK RESULTS ==========\n');
fprintf('Final f(x)      = %.10e\n', fval);
fprintf('True minimum    = %.10e\n', f_true);
fprintf('Distance to 0   = %.3e\n', norm(x_opt - x_true));
fprintf('Computation time = %.2f sec\n', comp_time);
