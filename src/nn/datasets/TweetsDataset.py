import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from ...constants import LEMMAS_PATH, DUPLICATED_PATH, LABELS


class TweetsDataset(Dataset):
    def __init__(self, X, y=None, is_test=False):
        super(TweetsDataset, self).__init__()
        self._X = np.array(X, dtype=np.float32)
        if not is_test:
            self._y = np.array(y, dtype=np.float32)

        self._is_test = is_test

    def __len__(self):
        return len(self._X)

    def __getitem__(self, idx):
        if not self._is_test:
            return self._X[idx], self._y[idx]
        else:
            return self._X[idx]
