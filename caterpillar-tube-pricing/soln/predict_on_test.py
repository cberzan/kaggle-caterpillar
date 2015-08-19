# from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd

from soln.dataset import AllCategoricalsFeaturizer
from soln.dataset import featurize_and_to_numpy
from soln.dataset import get_augmented_train_and_test_set
from soln.dataset import inverse_log_transform_y


if __name__ == "__main__":
    print "Loading augmented dataset..."
    aug_train_set, aug_test_set = get_augmented_train_and_test_set()

    X_train = aug_train_set
    y_train = X_train.pop('log_cost')
    X_test = aug_test_set

    print "Featurizing..."
    featurizer = AllCategoricalsFeaturizer()
    X_train_np, y_train_np, X_test_np, _ = featurize_and_to_numpy(
        featurizer, X_train, y_train, X_test, None)

    regressor = RandomForestRegressor(n_estimators=20)
    # regressor = RandomForestRegressor(n_estimators=100)
    # regressor = DummyRegressor(strategy='mean')
    print "Fitting..."
    regressor.fit(X_train_np, y_train_np)
    print "Predicting..."
    y_train_pred = regressor.predict(X_train_np)
    y_test_pred = regressor.predict(X_test_np)
    train_rmsle = np.sqrt(mean_squared_error(y_train_np, y_train_pred))
    print "train RMSLE {}".format(train_rmsle)

    print "Writing output..."
    df = pd.DataFrame()
    df['cost'] = inverse_log_transform_y(y_train_pred)
    df['id'] = df.index + 1
    df.to_csv("train_pred.csv", index=False, columns=['id', 'cost'])
    df = pd.DataFrame()
    df['cost'] = inverse_log_transform_y(y_test_pred)
    df['id'] = df.index + 1
    df.to_csv("test_pred.csv", index=False, columns=['id', 'cost'])

    print "Done!"
