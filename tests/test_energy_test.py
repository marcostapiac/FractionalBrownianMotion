import numpy as np

from utils.math_functions import energy_statistic


def test_energy_statistic():
    x = np.array([[1, 2, 10], [4, 5, 10]]).reshape(2, 3)
    y = np.array([[10, 20, 10.], [40, 50, 10.]]).reshape(2, 3)
    res = energy_statistic(x, y)
    assert (np.abs(res - 54.5428806) < 1e-6)
