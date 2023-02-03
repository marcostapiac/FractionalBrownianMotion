import pandas as pd

from src.CEV_multivar_posteriors import fBn_covariance_matrix
from src.ClassFractionalCEV import FractionalCEV
from utils.math_functions import np
from utils.plotting_functions import plot_subplots, plt


def store_data(Xs=None, Us=None, muU=1., muX=1., gamma=1., X0=1., U0=0., H=0.8, N=2 ** 11, T=1e-3 * 2 ** 11):
    sigmaX = np.sqrt(muX * gamma / 0.55)
    alpha = gamma/sigmaX
    deltaT = T / N
    if Xs is None:
        m = FractionalCEV(muU=muU, muX=muX, sigmaX=sigmaX, alpha=alpha, X0=X0, U0=U0)
        Xs, Us = m.euler_simulation(H=H, N=N, deltaT=deltaT)  # Simulated data
    df = pd.DataFrame.from_dict(data={'Log-Price': Us, 'Volatility': Xs})
    plot_subplots(np.arange(0, T + deltaT, step=deltaT), [Xs, Us], [None, None], ["Time", "Time"],
                  ["Volatility", "Log Price"],
                  "Project Model Simulation")
    plt.show()
    df.to_csv('../data/raw_data_simpleObsModel_{}_{}.csv'.format(int(np.log2(N)), int(10 * H)), index=False)


def precompute_fBn_covariance(deltaT=1e-3, N=2 ** 11, H=0.8):
    df = pd.DataFrame(data=np.power(deltaT, 2 * H) * fBn_covariance_matrix(H=H, N=N))
    df.to_csv("../data/fBn_covariance_{}_{}.csv".format(int(np.log2(N)), int(10 * H)), index=False)
