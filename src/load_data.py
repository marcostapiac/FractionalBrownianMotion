import pandas as pd

from utils.math_functions import np
from utils.plotting_functions import plot_subplots, plt


def load_data(N, H, T, isSimple=False):
    if isSimple:
        df = pd.read_csv("../data/raw_data_simpleObsModel_{}_{}.csv".format(int(np.log2(N)), int(10 * H)))
    else:
        df = pd.read_csv("../data/raw_data_{}_{}.csv".format(int(np.log2(N)), int(10 * H)))
    Xs, Us = df["Volatility"].to_numpy(), df["Log-Price"].to_numpy()
    deltaT = T / N
    plot_subplots(np.arange(0, T + deltaT, step=deltaT), [Xs, Us], [None, None], ["Time", "Time"],
                  ["Volatility", "Log Price"], "Project Model Simulation")
    plt.show()
    return df["Volatility"].to_numpy(), df["Log-Price"].to_numpy()


def load_fBn_covariance(N, H):
    df = pd.read_csv("../data/fBn_covariance_{}_{}.csv".format(int(np.log2(N)), int(10 * H)))
    return np.vstack(df.to_numpy())
