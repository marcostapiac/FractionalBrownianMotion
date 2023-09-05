import numpy as np
from src.load_data import load_data
from tqdm import tqdm

from src.classes.ClassFractionalCEV import FractionalCEV
from src.classes.ClassParticleFilter import FractionalParticleFilter
from utils.plotting_functions import plot_subplots, plt


def box_plots(S=100, muU=1., muX=1., gamma=1., X0=1., U0=0., H=0.8, N=2 ** 7, T=1e-3 * 2 ** 7, nParticles=100,
              loadData=False, saveFig=True):
    """ Generate data """
    sigmaX = np.sqrt(muX * gamma / 0.55)
    alpha = gamma / sigmaX
    deltaT = T / N
    model = FractionalCEV(muU=muU, muX=muX, sigmaX=sigmaX, alpha=alpha, X0=X0, U0=U0)
    if not loadData:
        Xs, Us = model.euler_simulation(H=H, N=N, deltaT=deltaT)  # Simulated data
        plot_subplots(np.arange(0, T + deltaT, step=deltaT), [Xs, Us], [None, None], ["Time", "Time"],
                      ["Volatility", "Log Price"],
                      "Project Model Simulation")
        plt.show()
    else:
        Xs, Us = load_data(N=N, H=H, T=T)
    L = Us.size
    """ Ideal scenario where true model parameters are known and data is simulated from model """
    pf = FractionalParticleFilter(nParticles=nParticles, model=model, deltaT=deltaT, H=H, N=N)
    rhos = [0.01, 0.1, 0.2, 0.5, 0.7, 0.9]
    logLikelihoods = np.empty(shape=(S, len(rhos)))
    for i in (range(len(rhos))):
        for j in range(S):
            pf.__init__(nParticles=nParticles, model=model, deltaT=deltaT, H=H, N=N)
            for m in tqdm(range(1, L)):
                pf.run_filter(observation=Us[m], deltaT=deltaT, index=m)
                pf.move_after_resample(currObs=Us[:m + 1], deltaT=deltaT, rho=rhos[i])
            logLikelihoods[j, i] = pf.logLEstimate
    plot_and_save_boxplot(logLikelihoods, dataLabels=rhos, ylabel="Log Likelihood", xlabel="$\\rho$",
                          title_plot="Box and Whisker for Log Likelihood Estimate", toSave=saveFig,
                          saveName="../pngs/BoxWhiskerVarianceHighParticles.png")


box_plots(loadData=True)
