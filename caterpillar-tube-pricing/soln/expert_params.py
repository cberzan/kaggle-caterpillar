import numpy as np


def base_get_indices(X):
    return np.ones(len(X), dtype=bool)


base_params = {
    'objective': 'reg:linear',
    'silent': 1,
    'num_rounds': 1000,
    'gamma': 0.0,
    'eta': 0.02,
    'max_depth': 8,
    'min_child_weight': 6,
    'subsample': 0.7,
    'colsample_bytree': 0.6,
}


def supplier66_get_indices(X):
    return (X.supplier == 'S-0066')


supplier66_params = {
    'objective': 'reg:linear',
    'silent': 1,
    'num_rounds': 1000,
    'gamma': 0.0,
    'eta': 0.02,
    'max_depth': 8,
    'min_child_weight': 6,
    'subsample': 0.7,
    'colsample_bytree': 0.6,
}
