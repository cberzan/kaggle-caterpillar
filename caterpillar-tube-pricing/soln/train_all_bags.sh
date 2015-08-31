#!/bin/bash

set -e

mkdir bags
for bag in $(seq 0 8); do
    echo "----------- bag $bag ----------"
    time python -m soln.train_bag
    mkdir bags/$bag
    mv experts train_pred.csv test_pred.csv bags/$bag
done
