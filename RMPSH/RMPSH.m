function [x_opt, fval, comp_time] = RMPSH(objFun, x0, lb, ub, options)
% RMPSH: Recursive Modified Pattern Search on Hyper-Rectangles
% Parallel-friendly version with options-struct interface.
%
% % Required inputs:
%   objFun  - function handle
%   x0      - initial point
%   lb, ub  - bounds
%
% % Optional (via struct "options"):
% MaxTime - maximum allowed execution time in seconds 
% MaxRuns - maximum number of runs (outer loops) 
% MaxIter - maximum iterations per run (inner loop) 
% sInitial - initial step-size (epsilon) 
% rho - step decay rate for runs 2+ 
% rho2 - step decay rate for run 1 
% TolFun1 - tolerance for step decay (controls epsilon updates) 
% TolFun2 - tolerance for run restart stopping 
% phi - minimum allowed step-size 
% cutoff - sparsity threshold 
% DisplayUpdate - 1 = show iteration updates; 0 = silent 
% DisplayEvery - display every that many seconds (if DisplayUpdate = 1) 
% PrintSolution - 1 = display final solution; 0 = don’t show 

% % Outputs: 
% x_opt - final solution (in original coordinates) 
% fval - objective value at x_opt 
% comp_time - total computation time (seconds)
tic;

% ================= DEFAULT OPTIONS (MSCOR-STYLE) =================
defaults.MaxTime       = 3600;
defaults.MaxRuns       = 1000;
defaults.MaxIter       = 10000;
defaults.sInitial      = 1;
defaults.rho           = 2;
defaults.rho2          = 2;
defaults.TolFun1       = 1e-6;
defaults.TolFun2       = 1e-20;
defaults.phi           = 1e-20;
defaults.cutoff        = 1e-20;
defaults.DisplayUpdate = 1;
defaults.DisplayEvery  = 2;
defaults.PrintSolution = 0;

% ================= MERGE USER OPTIONS =================
if nargin < 5 || isempty(options)
    options = defaults;
else
    fn = fieldnames(defaults);
    for k = 1:numel(fn)
        if ~isfield(options, fn{k})
            options.(fn{k}) = defaults.(fn{k});
        end
    end
end

% ================= EXTRACT OPTIONS =================
MaxTime       = options.MaxTime;
MaxRuns       = options.MaxRuns;
MaxIter       = options.MaxIter;
sInitial      = options.sInitial;
rho           = options.rho;
rho2          = options.rho2;
TolFun1       = options.TolFun1;
TolFun2       = options.TolFun2;
phi           = options.phi;
cutoff        = options.cutoff;
DisplayUpdate = options.DisplayUpdate;
DisplayEvery  = options.DisplayEvery;
PrintSolution = options.PrintSolution;

if DisplayUpdate == 1
    fprintf('===================== RMPSH Starts =====================\n');
end
% ================= BASIC CHECKS =================
x0 = x0(:); lb = lb(:); ub = ub(:);
M  = length(x0);

if length(lb) ~= M || length(ub) ~= M
    error('Dimension mismatch: x0, lb, ub must have same length.');
end
if any(ub <= lb)
    error('Each component must satisfy ub > lb.');
end

% ================= UNIT-BOX TRANSFORMATION =================
transformation      = @(theta)(theta .* (ub - lb) + lb);
anti_transformation = @(x)((x - lb) ./ (ub - lb));

theta0 = anti_transformation(x0);

ValAtInitialPoint = objFun(x0);

if DisplayUpdate == 1
    fprintf('\n=> Obj. fun. value at initial point: %g\n', ValAtInitialPoint);
    fprintf('=> Lets BLACKBOX it using RMPSH !!!\n\n');
end

Theta_array     = zeros(M, MaxRuns);
Loop_solution   = zeros(MaxRuns,1);
array_of_values = zeros(MaxIter,1);

last_toc  = 0;
break_now = 0;

% ================= MAIN LOOP =================
for iii = 1:MaxRuns
    
    epsilon = sInitial;
    
    if iii == 1
        epsilon_decreasing_factor = rho2;
        theta = theta0;
    else
        epsilon_decreasing_factor = rho;
        theta = Theta_array(:,iii-1);
    end
    
    for i = 1:MaxIter
        
        if toc > MaxTime
            break_now = 1;
            fprintf('=> RMPSH terminated after %.2f seconds.\n', MaxTime);
            break;
        end
        
        current_lh = objFun(transformation(theta));
        
        % ---- Time display ----
        toc_now = toc;
        if DisplayUpdate == 1 && (toc_now - last_toc > DisplayEvery)
            fprintf('=> Run: %d, Iter: %d, Obj: %g, log10(step): %.2f\n', ...
                iii, i, current_lh, log10(epsilon));
            last_toc = toc_now;
        end
        
        total_lh           = zeros(2*M,1);
        matrix_update_at_h = zeros(2*M,M);
        
        for index = 1:(2*M)
            
            location_number = mod(index-1,2*M)+1;
            theta_local     = theta.';
            epsilon_temp    = ((-1)^location_number)*epsilon;
            change_loc      = ceil(location_number/2);
            
            possibility      = theta_local;
            value_at_pos     = possibility(change_loc);
            possibility_temp = value_at_pos + epsilon_temp;
            
            % ---- Boundary control ----
            if (value_at_pos == 0 && epsilon_temp < 0)
                possibility_temp = 0;
            elseif (possibility_temp > 1 && value_at_pos < 1 - phi)
                ff = log(epsilon_temp / (1 - value_at_pos)) / log(epsilon_decreasing_factor);
                f  = ceil(ff);
                epsilon_temp     = epsilon_temp / (epsilon_decreasing_factor^f);
                possibility_temp = value_at_pos + epsilon_temp;
            elseif (possibility_temp < 0 && value_at_pos > phi)
                ff = log(-epsilon_temp / value_at_pos) / log(epsilon_decreasing_factor);
                f  = ceil(ff);
                epsilon_temp     = epsilon_temp / (epsilon_decreasing_factor^f);
                possibility_temp = value_at_pos + epsilon_temp;
            end
            
            if (possibility_temp < 0 || possibility_temp > 1)
                total_lh(index)              = current_lh;
                matrix_update_at_h(index,:) = theta_local;
            else
                if possibility_temp < cutoff
                    possibility_temp = 0;
                end
                possibility(change_loc) = possibility_temp;
                x_candidate = transformation(possibility.');
                total_lh(index) = objFun(x_candidate);
                matrix_update_at_h(index,:) = possibility;
            end
        end
        
        [candidate, I] = min(total_lh);
        if candidate < current_lh
            theta = matrix_update_at_h(I,:).';
        end
        
        array_of_values(i) = min(candidate, current_lh);
        
        if i > 1 && abs(array_of_values(i)-array_of_values(i-1)) < TolFun1
            if epsilon > phi
                epsilon = epsilon / epsilon_decreasing_factor;
            else
                break;
            end
        end
    end
    
    Theta_array(:,iii) = theta;
    Loop_solution(iii) = objFun(transformation(theta));
    
    if iii > 1
        if norm(Theta_array(:,iii-1)-Theta_array(:,iii)) < TolFun2
            break;
        end
    end
    
    if break_now == 1
        break;
    end
end

x_opt     = transformation(theta);
fval      = objFun(x_opt);
comp_time = toc;

if PrintSolution == 1
    fprintf('\n=> Final RMPSH solution:\n');
    disp(x_opt');
end

if DisplayUpdate == 1
    fprintf('\n=> Obj. fun. value at RMPSH minima: %g\n', fval);
    fprintf('=> Total time taken: %.4f secs.\n', comp_time);
    fprintf('xxxxxxxxxxxxxxxxxxxxxx RMPSH ends xxxxxxxxxxxxxxxxxxxxxxxxxx\n');
end
end
