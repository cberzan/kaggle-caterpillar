from soln.dataset import AllCategoricalsFeaturizer
from soln.dataset import featurize_and_to_numpy
from soln.dataset import generate_xv_splits
from soln.dataset import get_augmented_train_and_test_set

import numpy as np
import os


if __name__ == "__main__":
    print "Loading augmented dataset..."
    aug_train_set, aug_test_set = get_augmented_train_and_test_set()

    print "Dumping xv folds..."
    featurizer = AllCategoricalsFeaturizer()
    base_path = 'folds'
    for i, split in enumerate(generate_xv_splits(aug_train_set)):
        print i
        split_np = featurize_and_to_numpy(featurizer, *split)
        X_train_np, y_train_np, X_test_np, y_test_np = split_np
        np.savez_compressed(
            os.path.join(base_path, 'fold{}.npz'.format(i)),
            X_train=X_train_np,
            y_train=y_train_np,
            X_test=X_test_np,
            y_test=y_test_np)

    print "Wrote feature matrices to {}.".format(base_path)
