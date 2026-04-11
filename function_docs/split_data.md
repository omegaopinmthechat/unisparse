# split_data

## What this function does
Builds train/test indices for CV.

## Signature
~~~matlab
data = split_data(X, y, nfolds)
~~~

## Typical use case
- Generate CV folds for training and evaluation.
- Use nfolds = 1 for a fast holdout split.

## Mathematical form
If $nfolds=1$, use an 80/20 split:
$$
n_{train}=\lfloor 0.8n \rfloor,\quad n_{test}=n-n_{train}
$$
If $nfolds>1$, create near-balanced K folds (size difference at most 1).

## Parameters
- `X`: Design matrix of size n x p.
- `y`: Response vector of size n x 1.
- `nfolds`: Number of folds used for data splitting.

## Returns
- `data.train_idx{f}`, `data.test_idx{f}`.
- Backward-compatible data copies in `data.train{f}` and `data.test{f}`.

## Example
~~~matlab
data = split_data(X, y, 5);
train_idx = data.train_idx{1};
~~~

## Practical notes
- Splits are randomized; set rng(seed) for reproducibility.
- Fold sizes are balanced when nfolds > 1.
