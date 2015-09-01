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

    print "Selecting bag..."
    all_taids = np.unique(aug_train_set.tube_assembly_id.values)
    print "aug_train_set has {} rows and {} unique taids".format(
        len(aug_train_set), len(all_taids))
    n_bag_taids = 0.9 * len(all_taids)
    # get_augmented_train_and_test_set() fixes the RNG seed, so to get diverse
    # bags we need to reset the RNG...
    prng = np.random.RandomState()
    print prng.rand()
    bag_taids = prng.choice(all_taids, size=n_bag_taids, replace=False)
    unique_bag_taids = np.unique(bag_taids)
    bag_is = aug_train_set.tube_assembly_id.isin(bag_taids)
    orig_aug_train_set = aug_train_set
    aug_train_set = aug_train_set[bag_is].reset_index(drop=True)
    print "bag has {} rows ({} of all) and {} ({} of all) unique taids".format(
        len(aug_train_set), 1.0 * len(aug_train_set) / len(orig_aug_train_set),
        len(unique_bag_taids), 1.0 * len(unique_bag_taids) / len(all_taids))

    print "Training single model for bag..."
    timer = time()
    train_and_save_expert('base', aug_train_set, folds=False)
    timer = time() - timer
    print "    {} seconds elapsed".format(timer)

    X_train = aug_train_set
    y_train = X_train.pop('log_cost')
    X_test = aug_test_set

    print "Predicting..."
    timer = time()
    y_train_pred = get_predictions(
        'all', ['base'], base_get_indices, aug_train_set)
    train_rmsle = np.sqrt(mean_squared_error(y_train.values, y_train_pred))
    print "train RMSLE", train_rmsle
    y_test_pred = get_predictions(
        'all', ['base'], base_get_indices, aug_test_set)
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
