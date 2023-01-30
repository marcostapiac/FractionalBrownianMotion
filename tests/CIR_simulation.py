from tqdm import tqdm

from src.ClassFractionalCIR import FractionalCIR
from utils.math_functions import np
from utils.plotting_functions import plot_subplots, plt


def simulate(muU=1., muX=.5, gamma=1., N=2 ** 11, T=10, H=0.8, X0=1., U0=0., saveFig=False):
    sigmaX = np.sqrt(muX * gamma / 0.55)
    print(2. * muX * gamma / np.power(sigmaX, 2) - 0.5)
    deltaT = T / N
    X = []
    for _ in tqdm(range(1)):
        m = FractionalCIR(muU=muU, muX=muX, sigmaX=sigmaX, gamma=gamma, X0=X0, U0=U0)
        Xs, Us = m.euler_simulation(H=H, N=N, deltaT=deltaT)
        X.append(Xs)
    # plot(np.arange(0, T + deltaT, step=deltaT), X, label_args=[None for _ in range(1)], xlabel="Time",
    #     ylabel="Vol Path", title="Individual Simulation Paths")

    plot_subplots(np.arange(0, T + deltaT, step=deltaT), [Xs, Us], [None, None], ["Time", "Time"],
                  ["Volatility", "Log Price"],
                  "Project Model Simulation")
    if saveFig:
        plt.savefig("../pngs/SamplefCIRLongMem.png", bbox_inches="tight")
        plt.show()
        plt.close()
    else:
        plt.show()


simulate(saveFig=True)
