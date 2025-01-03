import numpy as np

from src.classes.ClassFractionalBrownianNoise import FractionalBrownianNoise
from utils.plotting_functions import plt, plot


def plotting_paths(save=True):
    """
    Function to plot sample paths of LSTM_fBm
    :param save:
    :return:
    """
    N_paths = 3
    N = 2 ** 10
    Hs = np.linspace(0.4, 0.7, N_paths + 1, endpoint=False)[1:]
    paths = []
    deltaT = 1e-3  # for Circulant
    T = deltaT * N  # for Circulant
    for i in range(N_paths):
        fbn = FractionalBrownianNoise(Hs[i])
        Z = np.power(deltaT, Hs[i]) * fbn.circulant_simulation(
            N)  # If using Circulant/CRMD need to scale by deltaT/(N ** Hs[i])
        X = np.cumsum(Z)
        paths.append(X)
    time_ax = np.arange(0, T, step=deltaT)
    plot(time_ax, paths, label_args=np.array(["$H = " + str(round(h, 3)) + "$" for h in Hs]),
         title="Fractional Brownian Motion Sample Paths", xlabel="Time", ylabel="Position")
    if save:
        plt.savefig("../../pngs/SamplefBmPathsCirculant.png", bbox_inches="tight", transparent=False)
        plt.show()
        plt.close()
    else:
        plt.show()
        plt.close()


if __name__ == "__main__":
    plotting_paths(True)
