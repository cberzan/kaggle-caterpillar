from collections import Counter
from collections import defaultdict
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
