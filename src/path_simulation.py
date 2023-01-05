from utils.plotting_functions import plot_process, plt
from utils.math_functions import np
from ClassFractionalBrownianNoise import FractionalBrownianNoise


def plotting_paths(save=True):
    N_paths = 3
    N = 2 ** 10
    Hs = np.linspace(0.35, 0.8, N_paths + 1, endpoint=False)[1:]
    paths = []
    for i in range(N_paths):
        fbn = FractionalBrownianNoise(Hs[i])
        Z = fbn.davies_and_harte_simulation(N) # If using CRMD need to scale by (N ** Hs[i])
        X = np.cumsum(Z)
        paths.append(X)
    time_ax = np.linspace(0, N, num=N)
    plot_process(time_ax, paths, label_args=Hs, title="Fractional Brownian Motion Sample Paths")
    if save:
        plt.savefig("SamplefBmPathsCirculant.png", bbox_inches="tight")
        plt.show()
        plt.close()
    else:
        plt.show()
        plt.close()


if __name__ == "__main__":
    plotting_paths(True)
