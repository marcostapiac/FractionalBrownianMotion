from tqdm import tqdm

from src.CEV_multivar_posteriors_repara_Zs import posteriors, fBn_covariance_matrix
from src.ClassFractionalCEV_reparam import FractionalCEV
from src.load_data import load_data
from src.priors import prior
from utils.math_functions import np
from utils.plotting_functions import plt, plot_subplots, gibbs_histogram_plot, plot_parameter_traces


def test_no_H(S=20000, muU=1., muX=1., gamma=1., X0=1., U0=0., H=0.8, N=2 ** 6, T=1e-3 * 2 ** 6,
              rng=np.random.default_rng(), loadData=False):
    alpha = gamma*muX
    sigmaX = np.sqrt(alpha / 0.55)
    deltaT = T / N
    if not loadData:
        m = FractionalCEV(muU=muU, gamma=gamma, alpha=alpha, sigmaX=sigmaX, X0=X0, U0=U0)
        Xs, Us = m.euler_simulation(H=H, N=N, deltaT=deltaT)  # Simulated data
        plot_subplots(np.arange(0, T + deltaT, step=deltaT), [Xs, Us], [None, None], ["Time", "Time"],
                      ["Volatility", "Log Price"], "Project Model Simulation")
        plt.show()
        SH = np.power(deltaT, 2 * H) * fBn_covariance_matrix(N=N, H=H)
        invfBnCovMat = np.linalg.inv(SH)
    else:
        Xs, Us = load_data(N=N, H=H, T=T)
        SH = np.power(deltaT, 2 * H) * fBn_covariance_matrix(N=N, H=H)
        invfBnCovMat = np.linalg.inv(SH)
    # TODO: MAP Optimisation for param initialisation
    muUParams, gammaParams, alphaParams, sigmaXParams = (0., 3.), (gamma, 0.001), (alpha, 0.001), (2.1, np.power(1., 2) * 3.1)
    theta = prior(muUParams=muUParams, gammaParams=gammaParams, muXParams=alphaParams, sigmaXParams=sigmaXParams)
    theta[4] = H
    Thetas = [theta]
    m = FractionalCEV(muU=theta[0], gamma=theta[1], alpha=theta[2], sigmaX=theta[3], X0=X0, U0=U0)
    Xs = m.state_simulation(H=theta[4], N=N, deltaT=deltaT)
    Zs = m.lamperti(Xs)
    for _ in tqdm(range(S)):
        theta = posteriors(muUParams=muUParams, gammaParams=gammaParams, alphaParams=alphaParams, transformedLatents=Zs,
                           sigmaXParams=sigmaXParams, deltaT=deltaT, observations=Us, latents=Xs, theta=theta, invfBnCovMat=invfBnCovMat,
                           rng=rng)
        Thetas.append(theta)
        if max(Xs) > 10:
            print(theta)
        m.__init__(muU=theta[0], gamma=theta[1], alpha=theta[2], sigmaX=theta[3], X0=X0, U0=U0)
        Xs = m.state_simulation(H=theta[4], N=N, deltaT=deltaT)
        Zs = m.lamperti(Xs)
    Thetas = np.array(Thetas).reshape((S + 1, 5))
    burnOut = int(S/10)
    plot_parameter_traces(S=S, Thetas=Thetas)
    Thetas[:,3] *= Thetas[:,3]
    gibbs_histogram_plot(Thetas, burnOut, plottitle="Partially Observed MCMC Histogram", trueVals=[muU, gamma, muX, sigmaX**2],
                         priorParams=[muUParams, gammaParams, alphaParams, sigmaXParams])
    plt.show()


test_no_H(loadData=False)
