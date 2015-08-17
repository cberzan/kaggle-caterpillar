import numpy as np
import os
import pandas as pd

from soln import dummy


def load_raw_data():
    # The 'data' dir is next to the 'soln' dir.
    base_path = os.path.join(
        os.path.dirname(dummy.__file__), '..', 'data', 'competition_data')

    filenames = [
        'train_set', 'test_set', 'tube', 'specs', 'bill_of_materials',
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
    X_ext = X

    # Join X_ext with tube.csv by tube_assembly_id.
    X_ext = pd.merge(X_ext, raw['tube'], on='tube_assembly_id')

    # Convert specs.csv to a DataFrame with columns `tube_assembly_id` and
    # `specs`, where `specs` is a list of strings.
    specs_df = pd.DataFrame()
    specs_df['tube_assembly_id'] = raw['specs']['tube_assembly_id']
    tmp_df = raw['specs'].where(pd.notnull(raw['specs']), None)
    specs = [filter(None, row[1:]) for row in tmp_df.values]
    specs_df['specs'] = specs

    # Join X_ext with specs_df on tube_assembly_id.
    X_ext = pd.merge(X_ext, specs_df, on='tube_assembly_id')

    # Convert bill_of_materials.csv to a DataFrame with columns
    # `tube_assembly_id` and `components`, where `components` is a list of
    # (component_str, quantity) tuples.
    bill = raw['bill_of_materials']
    components_df = pd.DataFrame()
    components_df['tube_assembly_id'] = bill['tube_assembly_id']
    tmp_df = bill.where(pd.notnull(bill), None)
    rows = [filter(None, row[1:]) for row in tmp_df.values]
    components = [list(zip(row[0::2], row[1::2])) for row in rows]
    components_df['components'] = components

    # Join X_ext with components_df on tube_assembly_id.
    X_ext = pd.merge(X_ext, components_df, on='tube_assembly_id')

    return X_ext


def log_transform_y(y):
    return np.log(y + 1)


def inverse_log_transform_y(y):
    return np.exp(y) - 1
