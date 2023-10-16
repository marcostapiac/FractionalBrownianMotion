import numpy as np
import torch


class PredictiveLSTMDataset(torch.utils.data.Dataset):
    """ Custom dataset for univariate time-series """

    def __init__(self, data: np.ndarray, lookback: int):
        super().__init__()
        X, y = [], []
        for i in range(data.shape[1] - lookback):
            feature = data[:, i:i + lookback]
            target = data[:, i + 1:i + lookback + 1]
            X.append(feature)
            y.append(target)
        X = np.array(X)
        y = np.array(y)
        newshape = (np.prod(X.shape[:2]), lookback, 1)
        X, y = torch.tensor(X.reshape(newshape)), torch.tensor(y.reshape(newshape))
        self.X = X
        self.y = y

    def __len__(self):
        """ Return number of features """
        return self.X.shape[0]

    def __getitem__(self, index):
        """ Return a single time-series feature-target """
        return self.X[index, :], self.y[index, :]
