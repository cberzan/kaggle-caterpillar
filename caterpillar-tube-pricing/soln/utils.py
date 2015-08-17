from sklearn import tree
from sklearn.externals.six import StringIO
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
