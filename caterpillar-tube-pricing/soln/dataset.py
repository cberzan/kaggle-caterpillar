from collections import Counter
from datetime import datetime
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


def log_transform_y(y):
    return np.log(y + 1)


def inverse_log_transform_y(y):
    return np.exp(y) - 1


def get_specs_df(raw):
    """
    Return DataFrame with columns `tube_assembly_id` and `specs`.

    Based on specs.csv. Each value in the `specs` column is a list of strings.
    If a spec occurs more than once for a tube_assembly_id, it will be
    repeated.
    """
    specs_df = pd.DataFrame()
    specs_df['tube_assembly_id'] = raw['specs']['tube_assembly_id']
    tmp_df = raw['specs'].where(pd.notnull(raw['specs']), None)
    specs = [filter(None, row[1:]) for row in tmp_df.values]
    specs_df['specs'] = specs
    return specs_df


def get_components_df(raw):
    """
    Return DataFrame with columns `tube_assembly_id` and `components`.

    Based on bill_of_materials.csv. Each value in the `components` column is a
    list of strings. If a component occurs with quantity N for a
    tube_assembly_id, it will be repeated N times in the list. This is OK,
    because the maximum N is 6 for our dataset.
    """
    bill = raw['bill_of_materials']
    components_df = pd.DataFrame()
    components_df['tube_assembly_id'] = bill['tube_assembly_id']
    tmp_df = bill.where(pd.notnull(bill), None)
    components = []
    for orig_row in (filter(None, row[1:]) for row in tmp_df.values):
        new_row = []
        for component_str, count in zip(orig_row[0::2], orig_row[1::2]):
            assert int(count) == count
            new_row.extend([component_str] * int(count))
        components.append(new_row)
    components_df['components'] = components
    return components_df


def get_quote_age_feature(dataset):
    """
    Return quote_age (quote_date_days_since_1900) Series.
    """
    series = pd.to_datetime(dataset['quote_date']) - datetime(1900, 1, 1)
    return series.astype('timedelta64[D]')


def get_adj_quantity_feature(dataset):
    """
    Return adj_quantity feature (combining min_order_quantity and quantity).
    """
    return dataset[['min_order_quantity', 'quantity']].max(axis=1)


def get_adj_bracketing_feature(dataset):
    """
    Return adj_bracketing feature (whether there really is bracket pricing).
    """
    adj_bracketing = np.zeros(len(dataset), dtype=np.bool)
    grouped = dataset.groupby(
        ['tube_assembly_id', 'supplier', 'quote_date'])
    for t_s_q, indices in grouped.groups.iteritems():
        if len(indices) > 1:
            adj_bracketing[indices] = True
    return adj_bracketing


def get_bracketing_pattern_feature(dataset):
    """
    Return bracketing_pattern feature (sorted tuple of adj_quantities seen).
    """
    grouped = dataset.groupby(
        ['tube_assembly_id', 'supplier', 'quote_date'])
    bracketing_pattern = [None] * len(dataset)
    for t_s_q, indices in grouped.groups.iteritems():
        if len(indices) > 1:
            bracket = tuple(sorted(dataset.adj_quantity[indices].values))
        else:
            bracket = ()
        for index in indices:
            bracketing_pattern[index] = bracket
    return bracketing_pattern


def get_augmented_dataset(orig_set, tube_df, specs_df, components_df):
    """
    Return aug_set with the same rows as orig_set, but more features.
    """
    aug_set = orig_set

    # Log-transform cost.
    if 'cost' in aug_set.columns:
        cost = aug_set.pop('cost')
        aug_set['log_cost'] = log_transform_y(cost)

    # Join with tube, specs, bill_of_materials tables.
    aug_set = pd.merge(aug_set, tube_df, on='tube_assembly_id')
    aug_set = pd.merge(aug_set, specs_df, on='tube_assembly_id')
    aug_set = pd.merge(aug_set, components_df, on='tube_assembly_id')

    # Rename some features.
    renamed_cols = {
        'wall': 'wall_thickness',
        'other': 'num_other',
    }
    aug_set.rename(columns=renamed_cols, inplace=True)

    # Convert some binary features from str to bool.
    bin_cols = {
        'bracket_pricing': 'Yes',
        'end_a_1x': 'Y',
        'end_a_2x': 'Y',
        'end_x_1x': 'Y',
        'end_x_2x': 'Y',
    }
    for col, true_val in bin_cols.iteritems():
        aug_set[col] = (aug_set[col] == true_val)

    # Add computed features.
    aug_set['quote_age'] = get_quote_age_feature(aug_set)
    aug_set['adj_quantity'] = get_adj_quantity_feature(aug_set)
    aug_set['adj_bracketing'] = get_adj_bracketing_feature(aug_set)
    aug_set['bracketing_pattern'] = get_bracketing_pattern_feature(aug_set)

    # TODO:
    # - bend_radius from tube.csv has missing values (9999) for 8 rows;
    #   currently that gets treated as the scalar 9999, which is wrong.
    # - material_id from tube.csv has missing values; currently
    #   OneHotFeaturizer treats missing values as a value `nan` that is
    #   different from 'other' and all the other values. Should we just use
    #   'other' for missing values?
    # - end_a and end_x from tube_csv have missing value 'NONE' and '9999',
    #   which pandas by default treats as two different string values)
    # - handle duplicate specs (e.g. SP-0007); currently ListFeaturizer
    #   just treats them as a single one.
    # - add an `ends` list-valued feature, e.g. [EF-003, EF-006], so that
    #   the info about both ends is merged into a single feature, which
    #   gets converted to num_EF_003, num_EF_006, etc. numerical features.
    # - similarly, add end_1x_count and end_2x count features, treating the
    #   two ends as interchangeable.
    # - features like num_sleeve, etc. based on component types

    return aug_set


def add_dev_fold_column(aug_train_set, num_folds=10):
    """
    Return aug_train_set with added `dev_fold` column.'

    Splits the dataset into `num_folds` folds, stratifying by supplier.
    """
    # Collect the list of unique `tube_assembly_id`s, and shuffle them. If the
    # same tube_assembly_id has multiple suppliers, pick one arbitrarily.
    taids = aug_train_set[['tube_assembly_id', 'supplier']].copy()
    taids.drop_duplicates(
        subset='tube_assembly_id', take_last=True, inplace=True)
    taids = taids.reset_index(drop=True)
    np.random.seed(666)
    taids['pos'] = np.random.permutation(len(taids))
    taids.sort('pos', inplace=True)
    taids.pop('pos')

    # Replace rare suppliers with "other".
    suppliers = taids.pop('supplier').values
    counter = Counter(suppliers)
    for i, val in enumerate(suppliers):
        if counter[val] < num_folds:
            suppliers[i] = "other"

    # Split taids into num_folds folds, stratifying by supplier.
    skf = StratifiedKFold(suppliers, n_folds=num_folds, random_state=666)
    taids['dev_fold'] = -1
    for i, (train_is, test_is) in enumerate(skf):
        taids.loc[test_is, 'dev_fold'] = i

    # Now taids is a DataFrame mapping tube_assembly_id to dev_fold.
    # Merge it with aug_train_set to add the dev_fold column in the latter.
    return aug_train_set.merge(taids, on='tube_assembly_id')


def get_augmented_train_and_test_set():
    """
    Return (aug_train_set, aug_test_set) DataFrames.
    """
    raw = load_raw_data()
    tube_df = raw['tube']
    specs_df = get_specs_df(raw)
    components_df = get_components_df(raw)
    aug_train_set = get_augmented_dataset(
        raw['train_set'], tube_df, specs_df, components_df)
    aug_train_set = add_dev_fold_column(aug_train_set)
    aug_test_set = get_augmented_dataset(
        raw['train_set'], tube_df, specs_df, components_df)
    return aug_train_set, aug_test_set
