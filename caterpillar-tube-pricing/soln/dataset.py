import numpy as np
import os
import pandas as pd

from soln import dummy


def load_raw_data():
    # The 'data' dir is next to the 'soln' dir.
    base_path = os.path.join(
        os.path.dirname(dummy.__file__), '..', 'data', 'competition_data')

    filenames = [
        'train_set', 'test_set', 'tube',
    ]
    raw = {}
    for filename in filenames:
        raw[filename] = pd.read_csv(os.path.join(base_path, filename + '.csv'))
    return raw


def get_dev_split(raw):
    X_all = raw['train_set'].copy()
    y_all = X_all.pop('cost')

    split_index = 27186
    X_train = X_all[:split_index]
    X_test = X_all[split_index:]
    y_train = y_all[:split_index]
    y_test = y_all[split_index:]

    return X_train, y_train, X_test, y_test


def get_extended_X(X, raw):
    """
    Return extended dataset by joining with other files.
    """
    X_ext = pd.merge(X, raw['tube'], on='tube_assembly_id')
    return X_ext


def log_transform_y(y):
    return np.log(y + 1)


def inverse_log_transform_y(y):
    return np.exp(y) - 1
