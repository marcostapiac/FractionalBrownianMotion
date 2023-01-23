from src.ClassFractionalCEV import FractionalCEV
from utils.plotting_functions import plot, plot_subplots, plt
from utils.math_functions import np
from tqdm import tqdm


def simulate(muU=1., muX=2., gamma=1., N=2 ** 11, T=10., H=0.8, X0=1., U0=0., save=False, transparent=False):
    sigmaX = np.sqrt(muX * gamma / 0.55)
    deltaT = T / N
    X = []
    m = FractionalCEV(muU=muU, muX=muX, sigmaX=sigmaX, gamma=gamma, X0=X0, U0=U0)
    Xs, Us = m.euler_simulation(H=H, N=N, deltaT=deltaT)
    X.append(Xs)
    plot_subplots(np.arange(0, T + deltaT, step=deltaT), [Xs, Us], [None, None], ["Time", "Time"],
                  ["Volatility", "Log Price"],
                  "Fractional CEV with $\mu_{U}, \gamma, \mu_{X}, \sigma_{X}, H = " + str(muU)+ " ,"+ str(gamma)+ " ,"+ str(muX)+ " ,"+ str(round(sigmaX,3))+ " ,"+ str(H)+ "$")
    if save:
        plt.savefig("SamplefCEVLongMem.png", bbox_inches="tight", transparent=transparent)
        plt.show()
        plt.close()
    else:
        plt.show()


simulate(save=True, transparent=False)
