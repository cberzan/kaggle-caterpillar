from collections import Counter
from collections import defaultdict
from sklearn import tree
from sklearn.externals.six import StringIO
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
import pydot
import xgboost as xgb


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


def train_model(params, get_indices, featurizer, X_train, y_train):
    # Select subset of train set according to get_indices.
    train_is = get_indices(X_train)
    X_train = X_train[train_is].reset_index(drop=True)
    y_train = y_train[train_is].reset_index(drop=True)

    # Featurize and convert to DMatrix.
    featurizer.fit(X_train)
    X_train_feats = featurizer.transform(X_train)
    X_train_np = X_train_feats.astype(np.float).values
    y_train_np = y_train.values
    xgtrain = xgb.DMatrix(X_train_np, label=y_train_np)

    # Train model.
    params = params.copy()
    num_rounds = params.pop('num_rounds')
    model = xgb.train(params.items(), xgtrain, num_rounds)

    return {
        'train_is': train_is,
        'X_train': X_train,
        'X_train_feats': X_train_feats,
        'y_train': y_train,
        'model': model,
    }


def eval_model(model, get_indices, featurizer, X_eval, y_eval):
    # Select subset of eval set according to get_indices.
    eval_is = get_indices(X_eval)
    X_eval = X_eval[eval_is].reset_index(drop=True)
    y_eval = y_eval[eval_is].reset_index(drop=True)

    # Featurize and convert to DMatrix.
    X_eval_feats = featurizer.transform(X_eval)
    X_eval_np = X_eval_feats.astype(np.float).values
    y_eval_np = y_eval.values
    xgeval = xgb.DMatrix(X_eval_np, label=y_eval_np)

    # Get predictions and compute RMSLE.
    y_eval_pred = model.predict(xgeval)
    rmsle = np.sqrt(mean_squared_error(y_eval_np, y_eval_pred))

    return {
        'eval_is': eval_is,
        'X_eval': X_eval,
        'X_eval_feats': X_eval_feats,
        'y_eval': y_eval,
        'y_eval_pred': y_eval_pred,
        'rmsle': rmsle,
    }
