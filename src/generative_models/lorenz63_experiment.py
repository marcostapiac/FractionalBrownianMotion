import numpy as np


def simulate_lorenz63(a: float, b: float, r: float, T: int):
    v10, v20, v30 = 0., 0., 0.
    Vs = np.zeros(shape=(3, T + 1))
    Vs[:, 0] = np.array([v10, v20, v30]).reshape(3, 1)
    for i in range(1, T + 1):
        Vs[0, i] = Vs[0, i - 1] + a * (Vs[1, i - 1] - Vs[0, i - 1])
        Vs[1, i] = -a * Vs[0, i - 1] - Vs[0, i - 1] * Vs[2, i - 1]
        Vs[2, i] = Vs[2, i - 1] * (1. - b) + Vs[0, i - 1] * Vs[1, i - 1] - b * (r + a)
    return Vs[:, 1:]


def observation_model(latents: np.ndarray, T: int) -> np.array:
    obs = np.zeros(shape=(3, T))
    for i in range(T):
        obs[:, i] = 0.1 * np.eye(latents.shape[0]) @ latents[:, i] + np.random.multivariate_normal(mean=[0., 0., 0.],
                                                                                                   cov=np.eye(
                                                                                                       latents.shape[
                                                                                                           0]))
