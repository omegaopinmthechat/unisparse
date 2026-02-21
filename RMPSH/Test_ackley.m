%% ================== Ackley (10D) using RMPSH ==================

clear; clc;

rng(1);   % reproducibility

% ------------------- Problem dimension -------------------
M = 10;

% ------------------- Bounds -------------------
lb = -33 * ones(M,1);
ub =  33 * ones(M,1);

% ------------------- Starting point -------------------
x0 = 20*(2*rand(M,1)-1);   % Uniform in (-20,20)

% ------------------- Ackley function -------------------
ackley = @(x) ...
    -20*exp(-0.2*sqrt(mean(x.^2))) ...
    -exp(mean(cos(2*pi*x))) ...
    +20 + exp(1);

fprintf('\nAckley at start = %.6f\n', ackley(x0));

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
    ackley, x0, lb, ub, ...
    MaxTime, MaxRuns, MaxIter, ...
    sInitial, rho, rho2, ...
    TolFun1, TolFun2, ...
    phi, lambda, DisplayUpdate, DisplayEvery, PrintSolution);

% ------------------- True optimum -------------------
x_true = zeros(M,1);
f_true = 0;

fprintf('\n========== ACKLEY RESULTS ==========\n');
fprintf('Final f(x)      = %.10e\n', fval);
fprintf('True minimum    = %.10e\n', f_true);
fprintf('Distance to 0   = %.3e\n', norm(x_opt - x_true));
fprintf('Computation time = %.2f sec\n', comp_time);
