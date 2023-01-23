from utils.math_functions import np
from src.ClassFractionalCIR import FractionalCIR
from src.ClassFractionalCEV import FractionalCEV
from utils.plotting_functions import plt, plot, plot_subplots
from src.priors import prior
from src.CEV_multivar_posteriors import posteriors, generate_V_matrix
from tqdm import tqdm


def test_no_H(S=10000, muU=1., muX=.1, gamma=1., X0=1., U0=0., H=0.8, N=2 ** 11, T=1e-6*2**11, rng=np.random.default_rng()):
    sigmaX = np.sqrt(muX * gamma / 0.55)
    print(sigmaX)
    deltaT = T / N
    m = FractionalCEV(muU=muU, gamma=gamma, muX=muX, sigmaX=sigmaX, X0=X0, U0=U0)
    Xs, Us = m.euler_simulation(H=H, N=N, deltaT=deltaT)  # Simulated data
    plot_subplots(np.arange(0, T + deltaT, step=deltaT), [Xs, Us], [None, None], ["Time", "Time"],
                  ["Volatility", "Log Price"], "Project Model Simulation")
    plt.show()
    muUParams, gammaParams, muXParams, sigmaXParams = (muU, 1.), (gamma), (muX), (2.1, np.power(sigmaX,2) * 1.1)
    theta = prior(muUParams=muUParams, gammaParams=gammaParams, muXParams=muXParams, sigmaXParams=sigmaXParams)
    theta[4] = H
    Thetas = [theta]
    V_mat = generate_V_matrix(vols=Xs, sigmaX=sigmaX, deltaT=deltaT, H=H, N=N)
    for i in tqdm(range(S)):
        theta = posteriors(muUParams=muUParams, gammaParams=gammaParams, muXParams=muXParams,
                           sigmaXParams=sigmaXParams, deltaT=deltaT, observations=Us, latents=Xs, theta=theta, V=V_mat,
                           rng=rng)
        Thetas.append(theta)
        if max(Xs) > 30:
            plot(np.arange(0, T + deltaT, step=deltaT), [Xs], ["Gibbs Iter: " + str(i)], "Time","Volatility", "Vols")
            plt.show()
            plt.close()
    Thetas = np.array(Thetas).reshape((S + 1, 5))
    burnOut = 0
    stdDev = 2.
    plot(np.arange(0, S + 1, step=1), [Thetas[:, 0]], ["Observation Mean"], "Gibbs Iteration", "Observation Mean",
         "Gibbs Sampler")
    plot(np.arange(0, S + 1, step=1), [Thetas[:, 1]], ["Volatility Mean Reversion"], "Gibbs Iteration",
         "Volatility Mean Reversion",
         "Gibbs Sampler")
    plot(np.arange(0, S + 1, step=1), [Thetas[:, 2]], ["Volatility Mean"], "Gibbs Iteration", "Volatility Mean",
         "Gibbs Sampler")
    plot(np.arange(0, S + 1, step=1), [Thetas[:, 3]], ["Volatility Std Parameter"], "Gibbs Iteration",
         "Volatility Std Parameter",
         "Gibbs Sampler")
    plot(np.arange(0, S + 1, step=1), [Thetas[:, 4]], ["Hurst Parameter"], "Gibbs Iteration", "Hurst Parameter",
         "Gibbs Sampler")
    print("muU: ", np.mean(Thetas[burnOut:, 0]) - stdDev*np.std(Thetas[burnOut:, 0]), np.mean(Thetas[burnOut:, 0]),
          np.mean(Thetas[burnOut:, 0]) + stdDev*np.std(Thetas[burnOut:, 0]))
    print("gamma: ", np.mean(Thetas[burnOut:, 1]) - stdDev*np.std(Thetas[burnOut:, 1]), np.mean(Thetas[burnOut:, 1]),
          np.mean(Thetas[burnOut:, 1]) + stdDev*np.std(Thetas[burnOut:, 1]))
    print("muX: ", np.mean(Thetas[burnOut:, 2]) - stdDev*np.std(Thetas[burnOut:, 2]), np.mean(Thetas[burnOut:, 2]),
          np.mean(Thetas[burnOut:, 2]) + stdDev*np.std(Thetas[burnOut:, 2]))
    print("sigmaX: ", np.mean(Thetas[burnOut:, 3]) - stdDev*np.std(Thetas[burnOut:, 3]), np.mean(Thetas[burnOut:, 3]),
          np.mean(Thetas[burnOut:, 3]) + stdDev*np.std(Thetas[burnOut:, 3]))

    plt.show()

    # TODO: Return final theta and log likelihood of data


test_no_H()
