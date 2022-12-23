from utils.plotting_functions import plot_path, plt
from utils.math_functions import np
from ClassProcess import FractionalBrownianNoise


def plotting_paths(save=True):
    N_paths = 3
    N = 2 ** 9
    Hs = np.linspace(0.35, 0.8, N_paths + 1, endpoint=False)[1:]
    paths = []
    for i in range(N_paths):
        fbn = FractionalBrownianNoise(Hs[i])
        Z = fbn.hosking_simulation(N)
        X = np.cumsum(Z)
        paths.append(X)
    plot_path(np.arange(0, N), paths, label_args=Hs, title="Fractional Brownian Motion Sample Paths")
    if save:
        plt.savefig("SamplefBmPathsHosking.png", bbox_inches="tight")
        plt.show()
        plt.close()
    else:
        plt.show()
        plt.close()


if __name__ == "__main__":
    plotting_paths(False)
