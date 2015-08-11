from collections import Counter
import numpy as np
import pandas as pd


class OneHotFeaturizer(object):
    """
    One-hot encoder for a categorical feature.

    Behaves like a composition of sklearn's LabelEncoder (with support for an
    "other" label) and OneHotEncoder.
    """
    def __init__(self, col_name, min_seen_count=10):
        self.col_name = col_name
        self.val_to_int = None
        self.int_to_val = None
        self.min_seen_count = min_seen_count

    def fit(self, dataset):
        """
        Build the featurizer using the given dataset (DataFrame).
        """
        # Build a map from string feature values to unique integers.
        # Assumes 'other' does not occur as a value.
        self.val_to_int = {'other': 0}
        self.int_to_val = ['other']
        next_index = 1
        counter = Counter(dataset[self.col_name])
        for val, count in counter.iteritems():
            assert val not in self.val_to_int
            if count >= self.min_seen_count:
                self.val_to_int[val] = next_index
                self.int_to_val.append(val)
                next_index += 1
        assert len(self.val_to_int) == next_index
        assert len(self.int_to_val) == next_index

    def transform(self, dataset):
        """
        Featurize the given dataset (DataFrame) and return result as DataFrame.
        """
        feats = np.zeros(
            (len(dataset), len(self.val_to_int)), dtype=bool)
        for i, val in enumerate(dataset[self.col_name]):
            if val in self.val_to_int:
                feats[i][self.val_to_int[val]] = True
            else:
                feats[i][self.val_to_int['other']] = True
        feat_names = [
            '{} {}'.format(self.col_name, val)
            for val in self.int_to_val]
        return pd.DataFrame(
            feats, index=dataset.index, columns=feat_names)


class CustomFeaturizer(object):
    """
    Combined featurizer for all features.
    """
    def __init__(self):
        self.featurizers = [
            OneHotFeaturizer('supplier'),
            OneHotFeaturizer('material_id'),
        ]

    def fit(self, dataset):
        for featurizer in self.featurizers:
            featurizer.fit(dataset)

    def transform(self, dataset):
        # Accumulate columns from all the featurizers.
        dfs = [
            featurizer.transform(dataset)
            for featurizer in self.featurizers]
        result = pd.concat(dfs, axis=1)

        # Add features without modification.
        orig_cols = [
            'annual_usage', 'min_order_quantity', 'quantity', 'diameter',
            'length', 'num_bends', 'bend_radius', 'num_boss', 'num_bracket',
        ]
        for col in orig_cols:
            result[col] = dataset[col]

        # Rename some features.
        renamed_cols = {
            'wall': 'wall_thickness',
            'other': 'num_other',
        }
        for col, new_col in renamed_cols.iteritems():
            result[new_col] = dataset[col]

        # Add some binary features.
        bin_cols = {
            'bracket_pricing': 'Yes',
            'end_a_1x': 'Y',
            'end_a_2x': 'Y',
            'end_x_1x': 'Y',
            'end_x_2x': 'Y',
        }
        for col, true_val in bin_cols.iteritems():
            result[col] = (dataset[col] == true_val)

        # TODO: Columns not used:
        #
        # From train_set.csv:
        # - quote_date
        #
        # From tube.csv:
        # - end_a
        # - end_x

        # TODO:
        # - material_id from tube.csv has missing values; currently
        #   OneHotFeaturizer treats missing values as a value `nan` that is
        #   different from 'other' and all the other values. Should we just use
        #   'other' for missing values?

        return result
