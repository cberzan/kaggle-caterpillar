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
        'tube_end_form',
    ]
    raw = {}
    for filename in filenames:
        raw[filename] = pd.read_csv(os.path.join(base_path, filename + '.csv'))
    return raw


def load_raw_components():
    # The 'data' dir is next to the 'soln' dir.
    base_path = os.path.join(
        os.path.dirname(dummy.__file__), '..', 'data', 'competition_data')

    group_names = [
        'adaptor', 'boss', 'elbow', 'float', 'hfl', 'nut', 'other', 'sleeve',
        'straight', 'tee', 'threaded',
    ]
    group_dfs = {}
    comp_types = pd.read_csv(os.path.join(base_path, 'type_component.csv'))
    for group_name in group_names:
        group_dfs[group_name] = pd.read_csv(
            os.path.join(base_path, 'comp_' + group_name + '.csv'))
    return comp_types, group_dfs


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


def get_component_info_df(comp_types, group_dfs):
    """
    Return DataFrame with a row for each component.
    """
    df = pd.DataFrame()
    col_to_default = {
        'component_type_id': 'OTHER',
        'weight': np.nan,
        'orientation': 'No',
        'unique_feature': 'No',
        'end_form_id_1': None,
        'end_form_id_2': None,
        'end_form_id_3': None,
        'end_form_id_4': None,
    }
    for group_name, group_df in group_dfs.iteritems():
        tmp_df = pd.DataFrame()
        tmp_df['component_id'] = group_df.component_id
        tmp_df['component_group_id'] = group_name
        for col_name, default_val in col_to_default.iteritems():
            if col_name in group_df.columns:
                tmp_df[col_name] = group_df[col_name]
            else:
                tmp_df[col_name] = default_val
        df = df.append(tmp_df)
    for col_name in ('orientation', 'unique_feature'):
        df[col_name] = (df[col_name] == 'Yes')
    return df


def get_forming_ends(raw):
    """
    Return set of "forming" end forms from tube_end_form.csv.
    """
    df = raw['tube_end_form']
    return set(df[df.forming == 'Yes'].end_form_id)


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


def get_component_groups_feature(dataset, component_info_df):
    """
    Return component_groups feature (list of component groups).

    Assumes `dataset` already has the `components` column.
    """
    comp_to_group = dict(zip(
        component_info_df.component_id.values,
        component_info_df.component_group_id.values))
    component_groups = []
    for components in dataset.components:
        groups = [comp_to_group[comp] for comp in components]
        component_groups.append(groups)
    return component_groups


def get_component_types_feature(dataset, component_info_df):
    """
    Return component_types feature (list of component types).

    Assumes `dataset` already has the `components` column.
    """
    comp_to_type = dict(zip(
        component_info_df.component_id.values,
        component_info_df.component_type_id.values))
    component_types = []
    for components in dataset.components:
        types = [comp_to_type[comp] for comp in components]
        component_types.append(types)
    return component_types


def get_unique_feature_count_feature(dataset, component_info_df):
    """
    Return unique_feature_count feature.

    (number of components with unique_feature=True)

    Assumes `dataset` already has the `components` column.
    """
    comp_to_uniqfeat = dict(zip(
        component_info_df.component_id.values,
        component_info_df.unique_feature.astype(np.int).values))
    unique_feature_count = []
    for components in dataset.components:
        count = sum(comp_to_uniqfeat[comp] for comp in components)
        unique_feature_count.append(count)
    return unique_feature_count


def get_orientation_count_feature(dataset, component_info_df):
    """
    Return orientation_count feature.

    (number of components with orientation=True)

    Assumes `dataset` already has the `components` column.
    """
    comp_to_ori = dict(zip(
        component_info_df.component_id.values,
        component_info_df.orientation.astype(np.int).values))
    orientation_count = []
    for components in dataset.components:
        count = sum(comp_to_ori[comp] for comp in components)
        orientation_count.append(count)
    return orientation_count


def get_total_component_weight_feature(dataset, component_info_df):
    """
    Return total_component_weight feature.

    (sum of weights of individual components)

    Assumes `dataset` already has the `components` column.
    """
    comp_to_weight = dict(zip(
        component_info_df.component_id.values,
        component_info_df.weight.values))
    # Treat missing weights as zeros.
    for comp in comp_to_weight:
        if np.isnan(comp_to_weight[comp]):
            comp_to_weight[comp] = 0.0
    total_comp_weight = []
    for components in dataset.components:
        total_weight = sum(comp_to_weight[comp] for comp in components)
        total_comp_weight.append(total_weight)
    return total_comp_weight


def get_component_end_forms_feature(dataset, component_info_df):
    """
    Return component_end_forms feature.

    This feature is a list of `end_form_id_{1,2,3,4}` from components in groups
    'threaded' and 'adaptor'.

    Assumes `dataset` already has the `components` column.
    """
    comp_end_forms = zip(
        component_info_df.component_id.values,
        component_info_df.end_form_id_1.values,
        component_info_df.end_form_id_2.values,
        component_info_df.end_form_id_3.values,
        component_info_df.end_form_id_4.values,
    )
    comp_to_end_forms = dict([
        # Annoying: Need to filter out both np.nan and None values.
        (tup[0], filter(lambda x: isinstance(x, str), tup[1:]))
        for tup in comp_end_forms
    ])
    component_end_forms = []
    for components in dataset.components:
        end_forms = []
        for comp in components:
            end_forms.extend(comp_to_end_forms[comp])
        component_end_forms.append(end_forms)
    return component_end_forms


def get_ends_features(dataset, forming_ends):
    """
    Return a dict of end-related features.
    """
    feats = {}

    # Combine the two end forms into a single list-valued feature.
    feats['ends'] = list(zip(dataset.end_a.values, dataset.end_x.values))

    # Combine the two end dimensions into a single numeric feature.
    feats['end_1x_count'] = (
        dataset.end_a_1x.astype(np.int) + dataset.end_x_1x.astype(np.int))
    feats['end_2x_count'] = (
        dataset.end_a_2x.astype(np.int) + dataset.end_x_2x.astype(np.int))

    # end_a_forming, end_x_forming, end_forming_count features.
    feats['end_a_forming'] = dataset.end_a.isin(forming_ends)
    feats['end_x_forming'] = dataset.end_x.isin(forming_ends)
    feats['end_forming_count'] = (
        feats['end_a_forming'].astype(np.int) +
        feats['end_x_forming'].astype(np.int))

    return feats


def get_augmented_dataset(
        orig_set, tube_df, specs_df, components_df, component_info_df,
        forming_ends):
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
    aug_set['component_groups'] = get_component_groups_feature(
        aug_set, component_info_df)
    aug_set['component_types'] = get_component_types_feature(
        aug_set, component_info_df)
    aug_set['unique_feature_count'] = get_unique_feature_count_feature(
        aug_set, component_info_df)
    aug_set['orientation_count'] = get_orientation_count_feature(
        aug_set, component_info_df)
    aug_set['total_component_weight'] = get_total_component_weight_feature(
        aug_set, component_info_df)
    aug_set['component_end_forms'] = get_component_end_forms_feature(
        aug_set, component_info_df)
    end_feats = get_ends_features(aug_set, forming_ends)
    for feat_name, feat_col in end_feats.iteritems():
        aug_set[feat_name] = feat_col

    # TODO:
    # - bend_radius from tube.csv has missing values (9999) for 8 rows;
    #   currently that gets treated as the scalar 9999, which is wrong.
    # - material_id from tube.csv has missing values; currently
    #   OneHotFeaturizer treats missing values as a value `nan` that is
    #   different from 'other' and all the other values. Should we just use
    #   'other' for missing values?
    # - end_a and end_x from tube_csv have missing value 'NONE' and '9999',
    #   which pandas by default treats as two different string values)

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
    taids.set_index('pos', inplace=True)
    taids.sort_index(inplace=True)

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
    comp_types, group_dfs = load_raw_components()
    tube_df = raw['tube']
    specs_df = get_specs_df(raw)
    components_df = get_components_df(raw)
    component_info_df = get_component_info_df(comp_types, group_dfs)
    forming_ends = get_forming_ends(raw)
    aug_train_set = get_augmented_dataset(
        raw['train_set'], tube_df, specs_df, components_df,
        component_info_df, forming_ends)
    aug_train_set = add_dev_fold_column(aug_train_set)
    raw['test_set'].pop('id')
    aug_test_set = get_augmented_dataset(
        raw['test_set'], tube_df, specs_df, components_df,
        component_info_df, forming_ends)
    return aug_train_set, aug_test_set


def generate_xv_splits(aug_train_set, num_folds=10):
    """
    Yield (X_train, y_train, X_test, y_test) tuples of DataFrames.
    """
    for fold in xrange(num_folds):
        Xy_train = aug_train_set[aug_train_set.dev_fold != fold]
        Xy_train = Xy_train.reset_index(drop=True)
        Xy_train.pop('dev_fold')
        X_train = Xy_train
        y_train = X_train.pop('log_cost')

        Xy_test = aug_train_set[aug_train_set.dev_fold == fold]
        Xy_test = Xy_test.reset_index(drop=True)
        Xy_test.pop('dev_fold')
        X_test = Xy_test
        y_test = X_test.pop('log_cost')

        yield X_train, y_train, X_test, y_test


class CategoricalToNumeric(object):
    """
    Convert categorical or list features into numeric features.

    A categorical feature taking values like 'a', 'b', 'c' is converted into
    binary features ['col_name a', 'col_name b', 'col_name c'].

    A list feature taking values like ['a', 'a'], ['a', b'], ['c'] is converted
    into numeric features ['col_name a', 'col_name b', 'col_name c'],
    representing how many times each value appeared.
    """

    def __init__(self, col_name, multiple=False, min_seen_count=10):
        self.col_name = col_name
        self.multiple = multiple
        self._counter = None
        self.val_to_int = None
        self.int_to_val = None
        self.min_seen_count = min_seen_count

    def fit(self, dataset):
        # Build a map from string feature values to unique integers.
        # Assumes 'other' does not occur as a value.
        self.val_to_int = {'XXX_other': 0}
        self.int_to_val = ['XXX_other']
        next_index = 1
        self.build_counter(dataset[self.col_name])
        for val, count in self._counter.iteritems():
            assert val not in self.val_to_int
            if count >= self.min_seen_count:
                self.val_to_int[val] = next_index
                self.int_to_val.append(val)
                next_index += 1
        assert len(self.val_to_int) == next_index
        assert len(self.int_to_val) == next_index

    def build_counter(self, series):
        self._counter = Counter()
        for val in series:
            if self.multiple:
                # val is a list of categorical values.
                self._counter.update(val)
            else:
                # val is a single categorical value.
                self._counter[val] += 1

    def transform(self, dataset):
        feats = np.zeros((len(dataset), len(self.val_to_int)))
        for i, orig_val in enumerate(dataset[self.col_name]):
            if self.multiple:
                # orig_val is a list of categorical values.
                list_of_vals = orig_val
            else:
                # orig_val is a single categorical value.
                list_of_vals = [orig_val]
            for val in list_of_vals:
                if val in self.val_to_int:
                    feats[i, self.val_to_int[val]] += 1
                else:
                    feats[i, self.val_to_int['XXX_other']] += 1
        feat_names = [
            '{} {}'.format(self.col_name, val)
            for val in self.int_to_val]
        return pd.DataFrame(
            feats, index=dataset.index, columns=feat_names)


class AllCategoricalsFeaturizer(object):
    """
    Convert all categoricals to numeric features.

    Output is a DataFrame.
    """
    def __init__(self, keep_orig_feats=False):
        self.keep_orig_feats = keep_orig_feats
        self.featurizers = [
            CategoricalToNumeric('supplier'),
            CategoricalToNumeric('material_id'),
            CategoricalToNumeric('end_a'),
            CategoricalToNumeric('end_x'),
            CategoricalToNumeric('specs', multiple=True),
            CategoricalToNumeric('components', multiple=True),
            CategoricalToNumeric('bracketing_pattern'),
            CategoricalToNumeric('ends', multiple=True),
            CategoricalToNumeric('component_groups', multiple=True),
            CategoricalToNumeric('component_types', multiple=True),
            CategoricalToNumeric('component_end_forms', multiple=True),
        ]
        self.features_to_remove = [
            'tube_assembly_id',
            'quote_date',
        ]

    def fit(self, dataset):
        for featurizer in self.featurizers:
            featurizer.fit(dataset)

    def transform(self, dataset, include_taid=False):
        result = dataset.copy(deep=False)
        if not self.keep_orig_feats:
            for col_name in self.features_to_remove:
                result.pop(col_name)

        dfs = [result]
        for featurizer in self.featurizers:
            dfs.append(featurizer.transform(dataset))
            if not self.keep_orig_feats:
                result.pop(featurizer.col_name)

        return pd.concat(dfs, axis=1)


def featurize_and_to_numpy(featurizer, X_train, y_train, X_test, y_test):
    """
    Featurize the given datasets, and convert to numpy arrays.
    """
    featurizer.fit(X_train)
    X_train_feats = featurizer.transform(X_train)
    X_test_feats = featurizer.transform(X_test)

    X_train_np = X_train_feats.astype(np.float).values
    y_train_np = y_train.values
    X_test_np = X_test_feats.astype(np.float).values
    if y_test is None:
        y_test_np = None
    else:
        y_test_np = y_test.values

    return X_train_np, y_train_np, X_test_np, y_test_np
