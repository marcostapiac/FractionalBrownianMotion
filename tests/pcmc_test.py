from utils.math_functions import np
from src.ClassFractionalCIR import FractionalCIR
from src.ClassFractionalCEV import FractionalCEV
from utils.plotting_functions import plot_subplots, plt, plot
from src.priors import prior
from src.posteriors import posteriors
from tqdm import tqdm


def test_no_H(S=5000, muU=1., muX=1., gamma=1., X0=1., U0=0., H=0.8, N=2 ** 9, T=10., rng=np.random.default_rng()):
    sigmaX = np.sqrt(muX * gamma / 0.55)
    deltaT = T / N
    m = FractionalCEV(muU=muU, gamma=gamma, muX=muX, sigmaX=sigmaX,  X0=X0, U0=U0)
    Xs, Us = m.euler_simulation(H=H, N=N, deltaT=deltaT)  # Simulated data
    # plot_subplots(np.arange(0, T + deltaT, step=deltaT), [Xs, Us], [None, None], ["Time", "Time"],
    #              ["Volatility", "Log Price"], "Project Model Simulation")
    # plt.show()
    muUParams, gammaParams, muXParams, sigmaXParams = (0., 1.), (gamma), (muX), (2.1, sigmaX / 1.1)
    theta = prior(muUParams=muUParams, gammaParams=gammaParams, muXParams=muXParams, sigmaXParams=sigmaXParams)
    theta[1] = gamma
    theta[2] = muX
    theta[-1] = H
    Thetas = [theta]
    m = FractionalCEV(muU=theta[0], gamma=theta[1], muX=theta[2], sigmaX=theta[3], X0=X0, U0=U0)
    #Xs = m.state_simulation(H=theta[4], N=N, deltaT=deltaT)
    for _ in tqdm(range(S)):
        theta = posteriors(muUParams=muUParams, gammaParams=gammaParams, muXParams=muXParams,
                           sigmaXParams=sigmaXParams, deltaT=deltaT, observations=Us, latents=Xs, theta=theta,
                           rng=rng)
        theta = np.append(theta, H)
        theta[1] = gamma
        theta[2] = muX
        theta[3] = sigmaX
        Thetas.append(theta)
        m.__init__(muU=theta[0], gamma=theta[1], muX=theta[2], sigmaX=theta[3], X0=X0, U0=U0)
        #Xs = m.state_simulation(H=theta[4], N=N, deltaT=deltaT)
    Thetas = np.array(Thetas).reshape((S + 1, 5))
    print(theta, muU, gamma, muU, sigmaX, H)
    plot(np.arange(0, S + 1, step=1), [Thetas[:, 0]], ["Observation Mean"], "Gibbs Iteration", "Observation Mean",
         "Gibbs Sampler")
    print(np.mean(Thetas[:,0])-np.std(Thetas[:,0]),np.mean(Thetas[:,0]), np.mean(Thetas[:,0])+np.std(Thetas[:,0]))
    plt.show()

    # TODO: Return final theta and log likelihood of data


test_no_H()
