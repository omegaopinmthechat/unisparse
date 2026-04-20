function [X, y, true_support] = Generate_data_twoclass(n, p, n_shifted, rho, shift)
% Generate two-class data with AR(1) covariance and mean shift signal.
%
%   n         - total samples (200)
%   p         - features (500)
%   n_shifted - features shifted in y=1 class (20)
%   rho       - AR(1) correlation (0.8)
%   shift     - mean shift in y=1 class (0.5)

    if nargin < 3, n_shifted = 20;  end
    if nargin < 4, rho       = 0.8; end
    if nargin < 5, shift     = 0.5; end

    % AR(1) covariance: Sigma_jk = rho^|j-k|
    % [R, C] = meshgrid(1:p, 1:p);
    % Sigma  = rho .^ abs(R - C);          % p x p

    % faster AR(1) for laptop
    col    = rho .^ (0:p-1)';
    Sigma  = toeplitz(col);

    % Cholesky factor for sampling
    L = chol(Sigma + 1e-10*eye(p), 'lower');

    % Split samples evenly between classes
    n1 = floor(n/2);   % y = 1
    n0 = n - n1;       % y = 0

    % Sample class 0: X ~ N(0, Sigma)
    X0 = randn(n0, p) * L';

    % Sample class 1: first n_shifted features shifted by +shift
    mu1      = zeros(1, p);
    mu1(1:n_shifted) = shift;
    X1 = randn(n1, p) * L' + mu1;

    % Stack and create labels
    X = [X0; X1];
    y = [zeros(n0,1); ones(n1,1)];

    % Shuffle rows
    idx = randperm(n);
    X   = X(idx, :);
    y   = y(idx);

    % Standardize features (fit on full X — or fit on training only in CV)
    X = X - mean(X, 1);
    s = std(X, 0, 1);  s(s==0) = 1;
    X = X ./ s;

    % Ground truth: first n_shifted features are the signal
    true_support = false(p, 1);
    true_support(1:n_shifted) = true;
end