% INIT_PROJECT  Add project folders to MATLAB path and propagate to workers
% Usage: run this once from the project root before running demos

repoRoot = pwd;
folders = {'Unisparse', 'supp funs', 'other methods', 'Data generation', 'RMPSH'};

added = {};
for k = 1:numel(folders)
    pth = fullfile(repoRoot, folders{k});
    if exist(pth, 'dir')
        addpath(pth);
        added{end+1} = pth; %#ok<AGROW>
    end
end

% Ensure a parallel pool exists and propagate paths to workers
try
    pool = gcp('nocreate');
    if isempty(pool)
        pool = parpool('Processes');
    end

    % Add the same folders on all workers
    for k = 1:numel(added)
        pctRunOnAll(sprintf('addpath(''%s'')', added{k}));
    end
catch
    % non-fatal; continue without worker propagation
end

fprintf('init_project: added %d folders to path.\n', numel(added));
for k = 1:numel(added)
    fprintf(' - %s\n', added{k});
end
