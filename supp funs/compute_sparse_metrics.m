function metrics = compute_sparse_metrics(beta_hat_whole, beta_true_whole, ...
                                           yhat_full, yfull, tol)
% compute_sparse_metrics
% ---------------------------------------------------------
% Returns a SINGLE row vector of metrics:
% [TPR, FPR, MCC, Beta_RMSE, Beta_MAD, Full_MSE, FDR]
%
% Coefficients include intercept:
%   beta_whole = [beta0 ; beta(1:p)]
%
% Support metrics are computed ONLY on beta(1:p).
% ---------------------------------------------------------

% ---------- Safety ----------
beta_hat_whole  = beta_hat_whole(:);
beta_true_whole = beta_true_whole(:);

if length(beta_hat_whole) ~= length(beta_true_whole)
    error('beta_hat_whole and beta_true_whole must have same length.');
end

if length(beta_hat_whole) < 2
    error('beta vectors must contain at least intercept + 1 slope.');
end

% ---------- Separate intercept and slopes ----------
beta_hat  = beta_hat_whole(2:end);
beta_true = beta_true_whole(2:end);

% ---------- Support via tolerance (slopes only) ----------
supp_true = abs(beta_true) > tol;
supp_hat  = abs(beta_hat)  > tol;

TP = sum( supp_hat &  supp_true);
FP = sum( supp_hat & ~supp_true);
TN = sum(~supp_hat & ~supp_true);
FN = sum(~supp_hat &  supp_true);

% ---------- TPR / FPR ----------
if (TP + FN) > 0
    TPR = TP / (TP + FN);
else
    TPR = 0;
end

if (FP + TN) > 0
    FPR = FP / (FP + TN);
else
    FPR = 0;
end

% ---------- FDR ----------
if (TP + FP) > 0
    FDR = FP / (TP + FP);
else
    FDR = 0;
end

% ---------- MCC ----------
den = sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN));
if den > 0
    MCC = (TP*TN - FP*FN) / den;
else
    MCC = 0;
end

% ---------- Estimation errors (slopes only) ----------
Beta_RMSE = sqrt(mean((beta_hat - beta_true).^2));
Beta_MAD  = mean(abs(beta_hat - beta_true));

% ---------- Prediction errors ----------
Full_MSE = mean((yfull - yhat_full).^2);

% ---------- SINGLE OUTPUT VECTOR ----------
metrics = [TPR, FPR, MCC, Beta_RMSE, Beta_MAD, Full_MSE, FDR];

end
