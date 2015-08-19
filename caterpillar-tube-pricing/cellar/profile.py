from soln.dataset import load_raw_data
from soln.dataset import generate_xv_splits_np
from soln.featurizer import CustomFeaturizer


if __name__ == "__main__":
    raw = load_raw_data()
    featurizer = CustomFeaturizer()
    for xv_split in generate_xv_splits_np(raw, featurizer):
        print [thing.shape for thing in xv_split]
