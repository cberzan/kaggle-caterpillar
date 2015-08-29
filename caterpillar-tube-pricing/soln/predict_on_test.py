from sklearn.metrics import mean_squared_error
from time import time
import numpy as np
import pandas as pd

from soln.dataset import AllCategoricalsFeaturizer
from soln.dataset import get_augmented_train_and_test_set
from soln.dataset import inverse_log_transform_y
from soln.layers import layer1_get_indices
from soln.layers import layer2_get_indices
from soln.utils import eval_model
from soln.utils import get_predictions
from soln.utils import load_model_from_disk
from soln.utils import train_model


def train_or_load_layer(
        params, get_indices, featurizer, X_train, y_train, filename,
        from_disk):
    timer = time()

    if from_disk:
        print "Loading {} from disk...".format(filename)
        # This still involves fitting the featurizer...
        layer = load_model_from_disk(
            params, get_indices, featurizer, X_train, y_train, filename)
    else:
        print "Training {}...".format(filename)
        layer = train_model(params, get_indices, featurizer, X_train, y_train)
        layer['model'].save_model(filename)

    timer = time() - timer
    print "    {} seconds elapsed".format(timer)

    return layer


if __name__ == "__main__":
    print "Loading augmented dataset..."
    timer = time()
    aug_train_set, aug_test_set = get_augmented_train_and_test_set()
    aug_train_set.pop('dev_fold')
    X_train = aug_train_set
    y_train = X_train.pop('log_cost')
    X_test = aug_test_set
    timer = time() - timer
    print "    {} seconds elapsed".format(timer)

    layer1_params = {
        'objective': 'reg:linear',
        'silent': 1,
        'num_rounds': 10000,
        'gamma': 0.0,

        'eta': 0.02,
        'min_child_weight': 6,
        'max_depth': 8,
        'subsample': 0.7,
        'colsample_bytree': 0.6,
    }
    layer1_featurizer = AllCategoricalsFeaturizer()
    layer1 = train_or_load_layer(
        layer1_params, layer1_get_indices, layer1_featurizer,
        X_train, y_train, 'layer1.model', from_disk=False)

    layer2_params = {
        'objective': 'reg:linear',
        'silent': 1,
        'num_rounds': 10000,
        'gamma': 0.0,

        'eta': 0.02,
        'min_child_weight': 6,
        'max_depth': 8,
        'subsample': 0.7,
        'colsample_bytree': 0.6,
    }
    layer2_featurizer = AllCategoricalsFeaturizer()
    layer2 = train_or_load_layer(
        layer2_params, layer2_get_indices, layer2_featurizer,
        X_train, y_train, 'layer2.model', from_disk=False)

    print "Results on train set:"
    results_1_on_1 = eval_model(
        layer1['model'], layer1_get_indices, layer1_featurizer,
        X_train, y_train)
    print "Layer 1 on all indices: RMSLE {}".format(
        results_1_on_1['rmsle'])
    results_1_on_2 = eval_model(
        layer1['model'], layer2_get_indices, layer1_featurizer,
        X_train, y_train)
    print "Layer 1 on layer 2 indices: RMSLE {}".format(
        results_1_on_2['rmsle'])
    results_2_on_2 = eval_model(
        layer2['model'], layer2_get_indices, layer2_featurizer,
        X_train, y_train)
    print "Layer 2 on layer 2 indices: RMSLE {}".format(
        results_2_on_2['rmsle'])
    y_train_pred = pd.Series(results_1_on_1['y_eval_pred'], copy=True)
    y_train_pred[results_2_on_2['eval_is']] = results_2_on_2['y_eval_pred']
    rmsle = np.sqrt(mean_squared_error(y_train.values, y_train_pred.values))
    print "Layer 1+2 on all indices: RMSLE {}".format(rmsle)

    print "Predicting..."
    timer = time()
    layer1_pred = get_predictions(
        layer1['model'], layer1_get_indices, layer1_featurizer, X_test)
    layer2_pred = get_predictions(
        layer2['model'], layer2_get_indices, layer2_featurizer, X_test)
    y_test_pred = pd.Series(layer1_pred['y_eval_pred'], copy=True)
    y_test_pred[layer2_pred['eval_is']] = layer2_pred['y_eval_pred']
    timer = time() - timer
    print "    {} seconds elapsed".format(timer)

    print "Writing output..."
    timer = time()
    df = pd.DataFrame()
    df['cost'] = inverse_log_transform_y(y_train_pred)
    df['id'] = df.index + 1
    df.to_csv("train_pred.csv", index=False, columns=['id', 'cost'])
    df = pd.DataFrame()
    df['cost'] = inverse_log_transform_y(y_test_pred)
    df['id'] = df.index + 1
    df.to_csv("test_pred.csv", index=False, columns=['id', 'cost'])
    timer = time() - timer
    print "    {} seconds elapsed".format(timer)

    print "Done!"
