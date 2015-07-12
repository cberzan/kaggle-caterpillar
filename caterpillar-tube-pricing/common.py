from sklearn.cross_validation import KFold
import numpy as np


def calc_loss(true_prices, predicted_prices):
    """
    Root Mean Squared Logarithmic Error loss function.

    Lower is better; zero is perfect.
    """
    assert len(true_prices) == len(predicted_prices)
    square_log_errs = (
        np.log(predicted_prices + 1) - np.log(true_prices + 1)) ** 2
    return np.sqrt(np.mean(square_log_errs))


def cross_validation_eval(learner, train_set, label_col, folds=10):
    """
    Evaluate the given learner by cross validation.

    Returns the loss value for each fold.
    """
    perm = np.random.permutation(np.arange(len(train_set)))
    shuffled_train_set = train_set.iloc[perm]
    losses = []
    kf = KFold(len(train_set), n_folds=folds)
    for train_is, test_is in kf:
        train_subset = shuffled_train_set.iloc[train_is]
        test_subset = shuffled_train_set.iloc[test_is]
        true_vals = test_subset[label_col]
        test_subset = test_subset.drop(label_col, 1)
        learner.reset()
        learner.learn(train_subset)
        pred_vals = learner.predict(test_subset)
        loss = calc_loss(true_vals, pred_vals)
        losses.append(loss)
    return np.array(losses)
