from utils.math_functions import np
from src.ClassFractionalCEV import FractionalCEV
from tqdm import tqdm
from utils.plotting_functions import plot_subplots, plot, plt
from src.ClassParticleFilter import FractionalParticleFilter


def test(muU=1., muX=1., gamma=1., X0=1., U0=0., H=0.8, N=2 ** 8, T=5.0, nParticles=100, save=False):
    sigmaX = np.sqrt(muX * gamma / 0.55)
    deltaT = T / N
    m = FractionalCEV(muU=muU, muX=muX, sigmaX=sigmaX, gamma=gamma, X0=X0, U0=U0)
    Xs, Us = m.euler_simulation(H=H, N=N, deltaT=deltaT)  # Simulated data
    L = Us.size
    """ Ideal scenario where true model parameters are known and data is simulated from model """
    pf = FractionalParticleFilter(nParticles=nParticles, muU=muU, muX=muX,
                               sigmaX=sigmaX, gamma=gamma,
                               X0=X0,
                               U0=U0, deltaT=deltaT, H=H, N=N)
    plot_subplots(np.arange(0, T + deltaT, step=deltaT), [Xs, Us], [None, None], ["Time", "Time"],
                  ["Volatility", "Log Price"],
                  "Project Model Simulation")
    plt.show()
    plt.close()
    predLogPrice = [X0]
    predVol = [X0]
    for j in tqdm(range(1, L)):
        # The index now represents actual time
        pf.run_filter(observation=Us[j], deltaT=deltaT, index=j)
        # Use MC approximations using normalised weights
        predLogPrice.append(pf.get_obs_mean_posterior())  # Log price forecast for index j+1
        predVol.append(pf.get_vol_mean_posterior(vol=Xs[j]))  # Vol estimate for index j
        pf.move_after_resample(deltaT=deltaT, rho=0.2)
    plot(np.arange(0., T + deltaT, step=deltaT), [Us, predLogPrice], ["True Log Price", "Optimal Forecast"], "Time",
         "Log Price",
         "Price Forecasting")
    if save:
        plt.savefig("PfTestEasyObsModelLogPrice.png", bbox_inches="tight", transparent=True)
        plt.show()
        plt.close()
    else:
        plt.show()
        plt.close()
    plot(np.arange(0, T + deltaT, step=deltaT), [Xs, predVol], ["True Vol", "Optimal Estimate"], "Time", "Volatility",
         "Volatility Smoothing")
    if save:
        plt.savefig("PfTestEasyObsModelVol.png", bbox_inches="tight", transparent=True)
        plt.show()
        plt.close()
    else:
        plt.show()
        plt.close()


test(save=True)
