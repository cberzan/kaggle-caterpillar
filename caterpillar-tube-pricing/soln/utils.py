from collections import Counter
from collections import defaultdict
from sklearn import tree
from sklearn.externals.six import StringIO
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
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


def get_fixed_and_var_cost(quantities, costs):
    """
    Return (fixed_cost, var_cost, r2).
    """
    xs = 1.0 / quantities
    ys = costs

    reg = LinearRegression(fit_intercept=True)
    Xs = xs.reshape(len(xs), 1)
    reg.fit(Xs, ys)
    fixed_cost = reg.coef_[0]
    var_cost = reg.intercept_
    r2 = reg.score(Xs, ys)
    return fixed_cost, var_cost, r2
