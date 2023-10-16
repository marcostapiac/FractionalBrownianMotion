import numpy as np
import torch


class DiscriminativeLSTMDataset(torch.utils.data.Dataset):
    """ Custom dataset for univariate time-series """

    def __init__(self, org_data: np.ndarray, synth_data: np.ndarray, labels: list) -> None:
        super().__init__()
        X, y = [], []
        for dp in org_data:
            X.append(dp)
            y.append(labels[0])  # Real
        for dp in synth_data:
            X.append(dp)
            y.append(labels[1])  # Fake
        X = np.array(X)
        y = np.array(y)
        X, y = torch.tensor(X.reshape((X.shape[0], X.shape[1], 1))), torch.tensor(y.reshape((y.shape[0], 1)))
        self.X = X
        self.y = y

    def __len__(self):
        """ Return number of features """
        return self.X.shape[0]

    def __getitem__(self, index):
        """ Return a single time-series feature-target """
        return self.X[index, :], self.y[index, :]
