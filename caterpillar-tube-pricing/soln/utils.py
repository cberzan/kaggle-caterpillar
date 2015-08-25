from collections import Counter
from collections import defaultdict
from sklearn import tree
from sklearn.externals.six import StringIO
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
import pydot


def print_feature_importances(X_train, regressor):
    assert len(X_train.columns) == len(regressor.feature_importances_)
    feat_imps = list(zip(X_train.columns, regressor.feature_importances_))
    feat_imps.sort(key=lambda (feat, imp): imp, reverse=True)
    for feat, imp in feat_imps:
        print feat, imp

    return feat_imps


def dump_decision_tree(filename, X_train, tree_reg, **kwargs):
    dot_data = StringIO()
    tree.export_graphviz(
        tree_reg, out_file=dot_data, feature_names=X_train.columns, **kwargs)
    graph = pydot.graph_from_dot_data(dot_data.getvalue())
    graph.write_pdf(filename)


def print_brackets(df, count):
    df = df.set_index('tube_assembly_id')

    taid_to_quantities = defaultdict(list)
    for taid, quantity in df['quantity'].iteritems():
        taid_to_quantities[taid].append(quantity)

    quantities_ctr = Counter()
    for taid in taid_to_quantities:
        taid_to_quantities[taid].sort()
        quantities_ctr[tuple(taid_to_quantities[taid])] += 1

    cum_frac = 0.0
    for brackets, count in quantities_ctr.most_common(count):
        frac = 1.0 * count / len(taid_to_quantities)
        cum_frac += frac
        print "brackets {}: count {} frac {} cum_frac {}".format(
            brackets, count, frac, cum_frac)


def eval_regressor(regressor, X_train_np, y_train_np, X_test_np, y_test_np):
    """
    Evaluate regressor and return (train_rmsle, test_rmsle).

    Assumes the X were featurized and the y were log-transformed.
    """
    regressor.fit(X_train_np, y_train_np)
    y_train_pred = regressor.predict(X_train_np)
    train_rmsle = np.sqrt(mean_squared_error(y_train_np, y_train_pred))
    y_test_pred = regressor.predict(X_test_np)
    test_rmsle = np.sqrt(mean_squared_error(y_test_np, y_test_pred))
    return train_rmsle, test_rmsle


def count_components(aug_set, component_info_df):
    """
    Return DataFrame with columns ['component_id', 'count'].

    'count' is the number of `tube_assembly_id`s in `aug_set` that have the
    given component.
    """
    # Collect tube_assembly_id -> components mapping. If same tube_assembly_id
    # has multiple values for components, pick one arbitrarily.
    df = aug_set[['tube_assembly_id', 'components']].copy()
    df.drop_duplicates(subset='tube_assembly_id', inplace=True)
    df.set_index('tube_assembly_id', inplace=True)

    # Count `tube_assembly_id`s that have each component, ignoring duplicates.
    cid_to_count = {cid: 0 for cid in component_info_df.component_id.values}
    for taid, cids in df.components.iteritems():
        for cid in np.unique(cids):
            cid_to_count[cid] += 1

    series = pd.Series(cid_to_count, name='count')
    series.index.name = 'component_id'
    df = series.reset_index()
    return df
