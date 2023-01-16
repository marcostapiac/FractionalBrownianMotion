from utils.plotting_functions import plot_fBm_process, plt
from utils.math_functions import np
from src.ClassFractionalBrownianNoise import FractionalBrownianNoise


def plotting_paths(save=True):
    N_paths = 5
    N = 2 ** 10
    Hs = np.linspace(0.3, 0.9, N_paths + 1, endpoint=False)[1:]
    paths = []
    T = 10. # for Circulant
    deltaT = T/N # for Circulan
    for i in range(N_paths):
        fbn = FractionalBrownianNoise(Hs[i])
        Z = (deltaT**Hs[i])*fbn.davies_and_harte_simulation(N) # If using Circulant/CRMD need to scale by deltaT/(N ** Hs[i])
        X = np.cumsum(Z)
        paths.append(X)
    time_ax = np.arange(0, T, step=deltaT)
    plot_fBm_process(time_ax, paths, isLatex=True, label_args=Hs, title="Fractional Brownian Motion Sample Paths")
    if save:
        plt.savefig("SamplefBmPathsCirculant.png", bbox_inches="tight" , transparent=True)
        plt.show()
        plt.close()
    else:
        plt.show()
        plt.close()


if __name__ == "__main__":
    plotting_paths(True)
