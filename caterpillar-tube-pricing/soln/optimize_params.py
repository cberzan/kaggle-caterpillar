from datetime import datetime
from sklearn.metrics import mean_squared_error
from time import time
import cPickle as pickle
import csv
import hyperopt
import numpy as np
import os
import xgboost as xgb


def xv_eval_params_dummy(folds, params, num_folds=10):
    """
    Dummy version of xv_eval_params, for testing.
    """
    return {
        'params': params,
        'train_wall_times': [0] * num_folds,
        'test_wall_times': [0] * num_folds,
        'train_rmsles': [0] * num_folds,
        'test_rmsles': [0] * num_folds,
    }


def xv_eval_params(folds, params, num_folds=10):
    """
    Fit and evaluate model, and return results as a dict.
    """
    params_copy = params.copy()
    num_rounds = params_copy.pop('num_rounds')

    train_wall_times = []
    test_wall_times = []
    train_rmsles = []
    test_rmsles = []

    for i in xrange(num_folds):
        print "fold", i

        X_train_np = folds[i]['X_train']
        y_train_np = folds[i]['y_train']
        X_test_np = folds[i]['X_test']
        y_test_np = folds[i]['y_test']
        xgtrain = xgb.DMatrix(X_train_np, label=y_train_np)
        xgtest = xgb.DMatrix(X_test_np)

        train_wall_time = time()
        model = xgb.train(params_copy.items(), xgtrain, num_rounds)
        train_wall_time = time() - train_wall_time

        y_train_pred = model.predict(xgtrain)
        train_rmsle = np.sqrt(mean_squared_error(y_train_np, y_train_pred))

        test_wall_time = time()
        y_test_pred = model.predict(xgtest)
        test_rmsle = np.sqrt(mean_squared_error(y_test_np, y_test_pred))
        test_wall_time = time() - test_wall_time

        train_wall_times.append(train_wall_time)
        test_wall_times.append(test_wall_time)
        train_rmsles.append(train_rmsle)
        test_rmsles.append(test_rmsle)

    return {
        'params': params,
        'train_wall_times': train_wall_times,
        'test_wall_times': test_wall_times,
        'train_rmsles': train_rmsles,
        'test_rmsles': test_rmsles,
    }


def summarize_results(raw_results):
    """
    Post-process the results from xv_eval_params.
    """
    results = {
        'finish_time': datetime.now().isoformat(),
        'train_wall_time_avg': np.mean(raw_results['train_wall_times']),
        'train_wall_time_std': np.std(raw_results['train_wall_times']),
        'test_wall_time_avg': np.mean(raw_results['test_wall_times']),
        'test_wall_time_std': np.std(raw_results['test_wall_times']),
        'train_rmsle_avg': np.mean(raw_results['train_rmsles']),
        'train_rmsle_std': np.std(raw_results['train_rmsles']),
        'test_rmsle_avg': np.mean(raw_results['test_rmsles']),
        'test_rmsle_std': np.std(raw_results['test_rmsles']),
    }
    results.update(raw_results['params'])
    return results


class HyperoptWrapper(object):
    def __init__(self, csv_path, num_folds=10):
        self.csv_path = csv_path
        self.csv_file = open(self.csv_path, 'w')
        self.csv_writer = csv.writer(self.csv_file)
        self.first_trial = True

        base_path = 'folds'
        self.folds = []
        for i in xrange(num_folds):
            self.folds.append(np.load(
                os.path.join(base_path, 'fold{}.npz'.format(i))))

    def objective(self, params):
        print "eval objective with params:", params

        # raw_results = xv_eval_params_dummy(self.folds, params)
        raw_results = xv_eval_params(self.folds, params)
        results = summarize_results(raw_results)
        results['loss'] = results['test_rmsle_avg']
        results['loss_variance'] = results['test_rmsle_std'] ** 2
        results['status'] = hyperopt.STATUS_OK

        self.save_trial(results)

        return results

    def save_trial(self, results):
        if self.first_trial:
            self.csv_cols = list(sorted(results.keys()))
            self.csv_writer.writerow(self.csv_cols)
            self.first_trial = False

        assert list(sorted(results.keys())) == self.csv_cols
        csv_vals = [results[key] for key in self.csv_cols]
        self.csv_writer.writerow(csv_vals)
        self.csv_file.flush()


if __name__ == "__main__":
    base_path = 'opt'
    run_id = datetime.now().isoformat()
    csv_path = os.path.join(base_path, 'hyperopt-{}.csv'.format(run_id))
    trials_path = os.path.join(base_path, 'hyperopt-{}.pickle'.format(run_id))
    wrapper = HyperoptWrapper(csv_path)
    hp = hyperopt.hp
    space = {
        'objective': 'reg:linear',
        'silent': 1,
        'num_rounds': 10,
        'eta': hp.quniform('eta', 0.01, 1, 0.01),
        'min_child_weight': 6,
        'subsample': 0.7,
        'colsample_bytree': 0.6,
        'max_depth': 8,

        # 'min_child_weight': hp.quniform('min_child_weight', 0, 20, 1),
        # 'max_depth': hp.quniform('max_depth', 1, 15, 1),
        # 'subsample': hp.quniform('subsample', 0.5, 1.0, 0.1),
        # 'colsample_bytree': hp.quniform('colsample_bytree', 0.1, 1.0, 0.1),
    }
    trials = hyperopt.Trials()
    best = hyperopt.fmin(
        wrapper.objective,
        space=space,
        algo=hyperopt.tpe.suggest,
        max_evals=10,
        trials=trials)
    with open(trials_path, 'w') as f:
        pickle.dump(trials, f)
