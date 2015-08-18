from collections import Counter
from sklearn.cross_validation import StratifiedKFold
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


def generate_xv_splits(raw):
    """
    Stratified 10-fold cross-validation iterator.

    Yields (X_train, y_train, X_test, y_test) tuples of DataFrames.
    """
    # Collect the list of unique `tube_assembly_id`s, shuffle them, and split
    # them into 90% train / 10% test, stratifying on supplier. If the same
    # tube_assembly_id has multiple suppliers, one is picked arbitrarily.
    taids = raw['train_set'][['tube_assembly_id', 'supplier']].copy()
    taids.drop_duplicates(
        subset='tube_assembly_id', take_last=True, inplace=True)
    taids = taids.reset_index(drop=True)
    np.random.seed(666)
    taids['pos'] = np.random.permutation(len(taids))
    taids.sort('pos', inplace=True)
    taids.pop('pos')
    suppliers = taids.pop('supplier').values
    counter = Counter(suppliers)
    for i, val in enumerate(suppliers):
        if counter[val] < 10:
            suppliers[i] = "other"

    # Stratified split.
    skf = StratifiedKFold(suppliers, n_folds=10, random_state=666)
    for train_is, test_is in skf:
        taids['test_set'] = taids.index.isin(test_is)

        # Split Xy_all according to taids['test_set'].
        Xy_all = raw['train_set'].copy()
        Xy_all = pd.merge(Xy_all, taids, on='tube_assembly_id')
        Xy_train = Xy_all[Xy_all['test_set'] == False]
        Xy_train = Xy_train.reset_index(drop=True)
        Xy_train.pop('test_set')
        X_train = Xy_train
        y_train = X_train.pop('cost')
        Xy_test = Xy_all[Xy_all['test_set'] == True]
        Xy_test = Xy_test.reset_index(drop=True)
        Xy_test.pop('test_set')
        X_test = Xy_test
        y_test = X_test.pop('cost')

        yield X_train, y_train, X_test, y_test


def generate_xv_splits_np(raw, featurizer):
    """
    Split using `generate_xv_splits`, then featurize and convert to ndarrays.

    Yields (X_train_feats, X_train_np, y_train_np, X_test_np, y_test_np)
    tuples, where X_train_feats is a DataFrame and the others are ndarrays.
    """
    for X_train, y_train, X_test, y_test in generate_xv_splits(raw):
        # FIXME: stop duplicating work here.
        X_train_ext = get_extended_X(X_train, raw)
        X_test_ext = get_extended_X(X_test, raw)
        y_train_log = log_transform_y(y_train)
        y_test_log = log_transform_y(y_test)

        featurizer.fit(X_train_ext)
        X_train_feats = featurizer.transform(X_train_ext)
        X_test_feats = featurizer.transform(X_test_ext)

        X_train_np = X_train_feats.astype(np.float).values
        X_test_np = X_test_feats.astype(np.float).values
        y_train_np = y_train_log.values
        y_test_np = y_test_log.values

        yield X_train_feats, X_train_np, y_train_np, X_test_np, y_test_np


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
