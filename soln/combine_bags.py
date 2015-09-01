import numpy as np
import os
import pandas as pd


if __name__ == "__main__":
    num_test_inst = 30235

    # Compute average prediction.
    test_y_pred = np.zeros(num_test_inst)
    num_bags = 9
    for bag in xrange(num_bags):
        df = pd.read_csv(os.path.join('bags', str(bag), 'test_pred.csv'))
        bag_y_pred = df.cost.values
        test_y_pred += bag_y_pred
    test_y_pred /= num_bags

    # Write output.
    df = pd.DataFrame()
    df['cost'] = test_y_pred
    df['id'] = df.index + 1
    df.to_csv("test_pred.csv", index=False, columns=['id', 'cost'])
