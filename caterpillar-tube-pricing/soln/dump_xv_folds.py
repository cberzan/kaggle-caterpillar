from soln.dataset import AllCategoricalsFeaturizer
from soln.dataset import featurize_and_to_numpy
from soln.dataset import generate_xv_splits
from soln.dataset import get_augmented_train_and_test_set
from soln.layers import layer1_get_indices
from soln.layers import layer2_get_indices

import numpy as np
import os


if __name__ == "__main__":
    get_indices = layer1_get_indices
    # get_indices = layer2_get_indices

    print "Loading augmented dataset..."
    aug_train_set, aug_test_set = get_augmented_train_and_test_set()

    print "Dumping xv folds using {}...".format(get_indices.__name__)
    featurizer = AllCategoricalsFeaturizer()
    base_path = 'folds'
    for i, split in enumerate(generate_xv_splits(aug_train_set)):
        print i

        # Select subset of train and test set according to get_indices.
        X_train, y_train, X_test, y_test = split
        train_is = get_indices(X_train)
        X_train = X_train[train_is].reset_index(drop=True)
        y_train = y_train[train_is].reset_index(drop=True)
        test_is = get_indices(X_test)
        X_test = X_test[test_is].reset_index(drop=True)
        y_test = y_test[test_is].reset_index(drop=True)

        split_np = featurize_and_to_numpy(
            featurizer, X_train, y_train, X_test, y_test)
        X_train_np, y_train_np, X_test_np, y_test_np = split_np
        np.savez_compressed(
            os.path.join(base_path, 'fold{}.npz'.format(i)),
            X_train=X_train_np,
            y_train=y_train_np,
            X_test=X_test_np,
            y_test=y_test_np)

    print "Wrote feature matrices to {}.".format(base_path)
