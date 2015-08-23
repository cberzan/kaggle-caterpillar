from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd

from soln.dataset import inverse_log_transform_y


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


def get_var_cost_only(quantities, costs, fixed_cost):
    """
    Return var_cost.
    """
    xs = 1.0 / quantities
    ys = costs
    zs = ys - fixed_cost * xs
    var_cost = np.mean(zs)
    return var_cost


brapa = (1, 2, 5, 10, 25, 50, 100, 250)
fc_vals = [15.704975, 19.043385, 20.295284, 23.633726]


def generate_bracket_csv(aug_train_set):
    df = aug_train_set[aug_train_set.bracketing_pattern == brapa]
    grouped = df.groupby('tube_assembly_id')
    taids = []
    fixed_costs = []
    var_costs = []
    for taid, indices in grouped.groups.iteritems():
        quantities = df.quantity[indices].values
        costs = inverse_log_transform_y(df.log_cost[indices].values)
        fixed_cost, var_cost, r2 = get_fixed_and_var_cost(quantities, costs)
        if r2 < 0.9999:
            print "{} has bad r2".format(taid)
        taids.append(taid)
        fixed_costs.append(fixed_cost)
        var_costs.append(var_cost)
    fixed_costs = np.array(fixed_costs)

    fc_class = -1 * np.ones(len(taids), dtype=np.int)
    adj_fixed_costs = np.zeros(len(taids))
    for i, fc_val in enumerate(fc_vals):
        indices = np.abs(fixed_costs - fc_val) < 0.1
        fc_class[indices] = i
        adj_fixed_costs[indices] = fc_val
    assert np.all(np.unique(fc_class) == [0, 1, 2, 3])

    adj_var_costs = np.zeros(len(taids))
    for i, taid in enumerate(taids):
        indices = grouped.groups[taid]
        quantities = df.quantity[indices].values
        costs = inverse_log_transform_y(df.log_cost[indices].values)
        fixed_cost = adj_fixed_costs[i]
        adj_var_costs[i] = get_var_cost_only(quantities, costs, fixed_cost)
        assert np.abs(adj_var_costs[i] - var_costs[i]) < 0.01

    df = pd.DataFrame({
        'tube_assembly_id': taids,
        'fixed_cost_class': fc_class,
        'fixed_cost': adj_fixed_costs,
        'var_cost': adj_var_costs,
    })
    df.to_csv('bracket.csv', index=False, columns=[
        'tube_assembly_id', 'fixed_cost_class', 'fixed_cost', 'var_cost'])
