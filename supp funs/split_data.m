function data = split_data(X, y, nfolds)
% split_data
% -------------------------------------------------
% If nfolds = 1:
%     80/20 train-test split
% If nfolds > 1:
%     K-fold cross-validation with balanced folds
%
% Output struct (indexes only to avoid copying data):
%     data.train_idx{f} , data.test_idx{f}
% -------------------------------------------------

[n, ~] = size(X);
idx = randperm(n);

if nfolds == 1
    % ---------------- 80/20 split ----------------
    ntrain = floor(0.8 * n);
    train_idx = idx(1:ntrain);
    test_idx  = idx(ntrain+1:end);
    data.train_idx{1} = train_idx;
    data.test_idx{1}  = test_idx;
    % Backwards-compatible copies for existing callers
    data.train{1}.X = X(train_idx,:);
    data.train{1}.y = y(train_idx);
    data.test{1}.X  = X(test_idx,:);
    data.test{1}.y  = y(test_idx);
    
else
    % ---------------- Proper K-fold split ----------------
    % Divide indices into K equal or near-equal folds
    fold_sizes = repmat(floor(n/nfolds), 1, nfolds);
    remainder  = n - sum(fold_sizes);
    
    % distribute leftover observations
    fold_sizes(1:remainder) = fold_sizes(1:remainder) + 1;

    % build each fold
    start = 1;
    for f = 1:nfolds
        stop = start + fold_sizes(f) - 1;
        test_idx = idx(start:stop);

        train_idx = setdiff(idx, test_idx);

        data.train_idx{f} = train_idx;
        data.test_idx{f}  = test_idx;
        % Backwards-compatible copies for existing callers
        data.train{f}.X = X(train_idx,:);
        data.train{f}.y = y(train_idx);
        data.test{f}.X  = X(test_idx,:);
        data.test{f}.y  = y(test_idx);

        start = stop + 1;
    end
end

end
