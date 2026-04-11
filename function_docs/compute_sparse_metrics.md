# compute_sparse_metrics

## What this function does
Computes sparse-recovery and prediction metrics.

## Signature
~~~matlab
metrics = compute_sparse_metrics(beta_hat_whole, beta_true_whole, yhat_full, yfull, tol)
~~~

## Typical use case
- Measure support recovery quality.
- Compare models using both prediction and sparsity metrics.

## Mathematical form
$$
TPR=\frac{TP}{TP+FN},\quad FPR=\frac{FP}{FP+TN},\quad FDR=\frac{FP}{TP+FP}
$$
$$
MCC=\frac{TP\cdot TN-FP\cdot FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}
$$
$$
Beta\text{-}RMSE=\sqrt{\frac{1}{p}\sum_j(\hat{\beta}_j-\beta_j)^2},\quad
Full\text{-}MSE=\frac{1}{n}\sum_i(y_i-\hat{y}_i)^2
$$

## Parameters
- `beta_hat_whole`: Estimated coefficients [beta0; beta].
- `beta_true_whole`: Ground-truth coefficients [beta0; beta].
- `yhat_full`: Predicted response values.
- `yfull`: Observed response values.
- `tol`: Tolerance threshold for support detection.

## Returns
- Row vector:
`[TPR, FPR, MCC, Beta_RMSE, Beta_MAD, Full_MSE, FDR]`

## Example
~~~matlab
metrics = compute_sparse_metrics(beta_hat, beta_true, yhat, y, 1e-4);
~~~

## Practical notes
- Support metrics exclude the intercept.
- Choose tol according to coefficient scale.
