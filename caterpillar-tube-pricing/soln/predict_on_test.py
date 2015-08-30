from sklearn.metrics import mean_squared_error
from time import time
import numpy as np
import pandas as pd

from soln.dataset import get_augmented_train_and_test_set
from soln.dataset import inverse_log_transform_y
from soln.expert_params import base_get_indices
from soln.experts import get_predictions
from soln.experts import train_and_save_expert


if __name__ == "__main__":
    print "Loading augmented dataset..."
    timer = time()
    aug_train_set, aug_test_set = get_augmented_train_and_test_set()
    aug_train_set.pop('dev_fold')
    timer = time() - timer
    print "    {} seconds elapsed".format(timer)

    print "Training experts..."
    timer = time()
    expert_names = ['base', 'supplier41', 'supplier72', 'supplier54']
    for expert_name in expert_names:
        train_and_save_expert(expert_name, aug_train_set, folds=False)
    timer = time() - timer
    print "    {} seconds elapsed".format(timer)

    X_train = aug_train_set
    y_train = X_train.pop('log_cost')
    X_test = aug_test_set

    print "Predicting..."
    timer = time()
    y_train_pred = get_predictions(
        'all', expert_names, base_get_indices, aug_train_set)
    train_rmsle = np.sqrt(mean_squared_error(y_train.values, y_train_pred))
    print "train RMSLE", train_rmsle
    y_test_pred = get_predictions(
        'all', expert_names, base_get_indices, aug_test_set)
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
