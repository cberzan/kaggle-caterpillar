from soln import expert_params
from soln.dataset import AllCategoricalsFeaturizer
from soln.dataset import generate_xv_splits

from sklearn.metrics import mean_squared_error
import cPickle as pickle
import numpy as np
import os
import pandas as pd
import xgboost as xgb


def get_expert_path(expert_name, fold_id):
    return os.path.join('experts', expert_name, str(fold_id))


def train_and_save_expert(expert_name, aug_train_set, folds=True):
    get_indices = getattr(expert_params, expert_name + '_get_indices')
    params = getattr(expert_params, expert_name + '_params')

    if folds:
        folds_gen = enumerate(generate_xv_splits(aug_train_set))
    else:
        X_train = aug_train_set.copy(deep=False)
        y_train = X_train.pop('log_cost')
        folds_gen = [('all', (X_train, y_train, None, None))]

    print "Training {}...".format(expert_name)
    for fold_id, split in folds_gen:
        print "fold {}...".format(fold_id)
        path = get_expert_path(expert_name, fold_id)
        if os.path.exists(path):
            print "  -> skipping because {} exists".format(path)
        else:
            # Select subset of train set according to get_indices.
            X_train, y_train, X_test, y_test = split
            train_is = get_indices(X_train)
            X_train = X_train[train_is].reset_index(drop=True)
            y_train = y_train[train_is].reset_index(drop=True)

            # Featurize and train model.
            featurizer = AllCategoricalsFeaturizer()
            featurizer.fit(X_train)
            X_train_feats = featurizer.transform(X_train)
            X_train_np = X_train_feats.astype(np.float).values
            y_train_np = y_train.values
            xgtrain = xgb.DMatrix(X_train_np, label=y_train_np)
            model = xgb.train(params.items(), xgtrain, params['num_rounds'])

            print "  -> saving to {}".format(path)
            os.makedirs(path)
            with open(os.path.join(path, 'featurizer'), 'w') as f:
                pickle.dump(featurizer, f)
            model.save_model(os.path.join(path, 'model'))


def load_expert(expert_name, fold_id):
    get_indices = getattr(expert_params, expert_name + '_get_indices')
    params = getattr(expert_params, expert_name + '_params')
    path = get_expert_path(expert_name, fold_id)
    with open(os.path.join(path, 'featurizer')) as f:
        featurizer = pickle.load(f)
    model = xgb.Booster()
    model.load_model(os.path.join(path, 'model'))
    return {
        'get_indices': get_indices,
        'params': params,
        'featurizer': featurizer,
        'model': model,
    }


def xv_eval_experts(expert_names, get_indices, aug_train_set):
    rmsles = []
    for fold_id, split in enumerate(generate_xv_splits(aug_train_set)):
        # Select subset of test set according to get_indices.
        X_train, y_train, X_test, y_test = split
        test_is = get_indices(X_test)
        X_test = X_test[test_is].reset_index(drop=True)
        y_test = y_test[test_is].reset_index(drop=True)

        # Get predictions from experts and compute RMSLE.
        y_test_pred = get_predictions(
            fold_id, expert_names, get_indices, X_test)
        rmsle = np.sqrt(mean_squared_error(y_test.values, y_test_pred))
        rmsles.append(rmsle)

    return {
        'rmsles': rmsles,
        'rmsle_avg': np.mean(rmsles),
        'rmsle_std': np.std(rmsles),
    }


def get_predictions(fold_id, expert_names, get_indices, X_eval):
    y_eval_pred = pd.Series(np.zeros(len(X_eval)))
    have_y_eval_pred = pd.Series(np.zeros(len(X_eval), dtype=bool))

    # Get layered predictions from each expert.
    for expert_name in expert_names:
        expert = load_expert(expert_name, fold_id)

        # Select subset according to expert's indices.
        eval_is = expert['get_indices'](X_eval)
        X_eval_sub = X_eval[eval_is].reset_index(drop=True)

        # Featurize and get predictions.
        X_eval_sub_feats = expert['featurizer'].transform(X_eval_sub)
        X_eval_sub_np = X_eval_sub_feats.astype(np.float).values
        xgeval = xgb.DMatrix(X_eval_sub_np)
        y_eval_sub = expert['model'].predict(xgeval)

        # Merge predictions into what we have from layers above.
        y_eval_pred[eval_is] = y_eval_sub
        have_y_eval_pred[eval_is] = True

    assert have_y_eval_pred.all()
    return y_eval_pred.values
