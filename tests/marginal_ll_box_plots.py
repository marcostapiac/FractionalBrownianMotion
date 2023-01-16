from utils.math_functions import np
from src.ClassFractionalCEV import FractionalCEV
from src.ClassFractionalCIR import FractionalCIR
from utils.plotting_functions import plot_subplots, plt, boxplot
from src.ClassParticleFilter import FractionalParticleFilter
from tqdm import tqdm


def box_plots(S=100, muU=1., muX=1., gamma=1., X0=1., U0=0., H=0.8, N=2 ** 7, T=1.0, nParticles=100):
    """ Generate data """
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
    rhos = [0.01, 0.1, 0.2, 0.5, 0.7, 0.9]
    logLikelihoods = np.empty(shape=(S, len(rhos)))
    for i in (range(len(rhos))):
        for j in range(S):
            pf.__init__(nParticles=nParticles, muU=muU, muX=muX,
                        sigmaX=sigmaX, gamma=gamma,
                        X0=X0,
                        U0=U0, deltaT=deltaT, H=H, N=N)
            for m in tqdm(range(1, L)):
                pf.run_filter(observation=Us[m], deltaT=deltaT, index=m)
                pf.move_after_resample(deltaT=deltaT, rho=rhos[i])
            logLikelihoods[j, i] = pf.logLEstimate
    boxplot(logLikelihoods, dataLabels=rhos, ylabel="Log Likelihood", xlabel="$\\rho$",
            plottitle="Box and Whisker for Log Likelihood Estimate")


box_plots()
