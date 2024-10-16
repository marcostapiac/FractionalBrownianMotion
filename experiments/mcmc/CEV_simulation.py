import numpy as np

from src.classes.ClassFractionalCEV import FractionalCEV
from utils.plotting_functions import plot_subplots, plt


def simulate(muU=1., muX=2., gamma=1., N=2 ** 11, T=10., H=0.8, X0=1., U0=0., saveFig=False, transparent=False):
    sigmaX = np.sqrt(muX * gamma / 0.55)
    alpha = gamma * muX
    deltaT = T / N
    X = []
    m = FractionalCEV(muU=muU, muX=muX, sigmaX=sigmaX, alpha=alpha, X0=X0, U0=U0)
    Xs, Us = m.euler_simulation(H=H, N=N, deltaT=deltaT)
    X.append(Xs)
    plot_subplots(np.arange(0, T + deltaT, step=deltaT), data=np.array([Xs, Us]), label_args=np.array([None, None]),
                  xlabels=np.array(["Time", "Time"]),
                  ylabels=np.array(["Volatility", "Log Price"]),
                  globalTitle="Fractional CEV with $\mu_{U}, \gamma, \mu_{X}, \sigma_{X}, H = " + str(muU) + " ," + str(
                      gamma) + " ," + str(muX) + " ," + str(round(sigmaX, 3)) + " ," + str(H) + "$",
                  saveTransparent=False)
    if saveFig:
        plt.savefig("../pngs/SamplefCEVLongMem.png", bbox_inches="tight", transparent=transparent)
        plt.show()
        plt.close()
    else:
        plt.show()


simulate(saveFig=True, transparent=False)
