import numpy as np


def layer1_get_indices(X):
    return np.ones(len(X), dtype=bool)


common_brackets = [
    (1, 2, 5, 10, 25, 50, 100, 250),
    (1, 6, 20),
    (1, 2, 3, 5, 10, 20),
    (1, 2, 5, 10, 25, 50, 100),
    (5, 19, 20),
]


def layer2_get_indices(X):
    return ~X.bracketing_pattern.isin(common_brackets)
