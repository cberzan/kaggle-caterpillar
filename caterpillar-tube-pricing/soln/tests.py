from soln.dataset import get_dev_split
from soln.dataset import get_extended_X
from soln.dataset import load_raw_data

import nose
import sys


global raw
raw = None


def setup_module():
    print >>sys.stderr, "load_raw_data..."
    global raw
    raw = load_raw_data()


class TestDataset(object):
    def test_dev_split_no_overlap(self):
        X_train, y_train, X_test, y_test = get_dev_split(raw)

        # All instances assigned to either test or train.
        nose.tools.assert_equal(len(X_train), len(y_train))
        nose.tools.assert_equal(len(X_test), len(y_test))
        nose.tools.assert_equal(
            len(X_train) + len(X_test), len(raw['train_set']))

        # Test and train contain no `tube_assembly_id`s in common.
        train_ids = set(X_train['tube_assembly_id'])
        test_ids = set(X_test['tube_assembly_id'])
        nose.tools.assert_false(train_ids.intersection(test_ids))

        # Split is roughly 90% / 10%.
        test_frac = 1.0 * len(X_test) / len(raw['train_set'])
        print test_frac
        nose.tools.assert_almost_equals(test_frac, 0.1, delta=0.05)

    def test_csv_merge(self):
        X_train, y_train, X_test, y_test = get_dev_split(raw)
        X_train_ext = get_extended_X(X_train, raw)

        # Check join with tube.csv.
        taid = 'TA-00034'
        df = X_train_ext[X_train['tube_assembly_id'] == taid]
        ref_diams = raw['tube'][raw['tube']['tube_assembly_id'] == taid]
        nose.tools.assert_equal(len(df['diameter'].unique()), 1)
        nose.tools.assert_equal(len(ref_diams), 1)
        nose.tools.assert_equal(
            df['diameter'].unique()[0],
            ref_diams['diameter'].values[0])

        # Check join with specs.csv.
        taid = 'TA-00207'
        ref_specs = ['SP-0063', 'SP-0070', 'SP-0080']
        df = X_train_ext[X_train['tube_assembly_id'] == taid]
        assert len(df['specs']) >= 1
        for val in df['specs']:
            nose.tools.assert_equal(val, ref_specs)

        # Check join with bill_of_materials.csv.
        taid = 'TA-00249'
        ref_components = [
            ('C-1536', 2.0),
            ('C-1642', 1.0),
            ('C-1649', 1.0),
        ]
        df = X_train_ext[X_train['tube_assembly_id'] == taid]
        assert len(df['components']) >= 1
        for val in df['components']:
            nose.tools.assert_equal(val, ref_components)
