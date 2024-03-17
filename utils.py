from sklearn.model_selection import KFold
import numpy as np

def get_split_mont(fold, cross_vali_num, seed=12345):
    # this is seeded, will be identical each time
    kf = KFold(n_splits=cross_vali_num, shuffle=True, random_state=seed)
    all_keys = np.arange(0, 65)#(0,24)
    splits = kf.split(all_keys)
    for i, (train_idx, test_idx) in enumerate(splits):
        train_keys = all_keys[train_idx]
        test_keys = all_keys[test_idx]
        if i == fold:
            break
    return train_keys, test_keys