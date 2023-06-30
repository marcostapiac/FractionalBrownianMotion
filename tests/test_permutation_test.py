import numpy as np

from utils.math_functions import permutation_test, energy_statistic, MMD_statistic


def test_permutation_test_2():
    dim = 64
    num = 100
    data1 = np.array([np.random.randn(dim) for _ in range(num)]).reshape(num, dim)
    data2 = np.array([0.1 * np.random.randn(dim) for _ in range(num)]).reshape(num, dim)
    p = permutation_test(data1, data2, 1000, energy_statistic)
    assert (p > 0.05)


def test_permutation_test_1():
    dim = 64
    num = 100
    data1 = np.array([np.random.randn(dim) for _ in range(num)]).reshape(num, dim)
    data2 = np.array([np.random.randn(dim) for _ in range(num)]).reshape(num, dim)
    p = permutation_test(data1, data2, 1000, MMD_statistic)
    assert (p > 0.05)


if __name__ == "__main__":
    test_permutation_test_1()
    test_permutation_test_2()
