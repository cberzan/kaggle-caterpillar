from sklearn.metrics import mean_squared_error
from time import time
import numpy as np
import pandas as pd
import xgboost as xgb

from soln.dataset import AllCategoricalsFeaturizer
from soln.dataset import featurize_and_to_numpy
from soln.dataset import get_augmented_train_and_test_set
from soln.dataset import inverse_log_transform_y


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

    print "Featurizing..."
    timer = time()
    featurizer = AllCategoricalsFeaturizer()
    X_train_np, y_train_np, X_test_np, _ = featurize_and_to_numpy(
        featurizer, X_train, y_train, X_test, None)
    xgtrain = xgb.DMatrix(X_train_np, label=y_train_np)
    xgtest = xgb.DMatrix(X_test_np)
    timer = time() - timer
    print "    {} seconds elapsed".format(timer)

    print "Fitting..."
    timer = time()
    params = {
        'objective': 'reg:linear',
        'eta': 0.02,
        'min_child_weight': 6,
        'subsample': 0.7,
        'colsample_bytree': 0.6,
        'scale_pos_weight': 0.8,  # undocumented?!
        'silent': 1,
        'max_depth': 8,
        'max_delta_step': 2,
    }
    num_rounds = 5000
    model = xgb.train(params.items(), xgtrain, num_rounds)
    timer = time() - timer
    print "    {} seconds elapsed".format(timer)

    print "Predicting..."
    timer = time()
    y_train_pred = model.predict(xgtrain)
    y_test_pred = model.predict(xgtest)
    train_rmsle = np.sqrt(mean_squared_error(y_train_np, y_train_pred))
    print "train RMSLE {}".format(train_rmsle)
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
