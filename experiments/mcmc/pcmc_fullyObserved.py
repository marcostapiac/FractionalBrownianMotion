import numpy as np
from src.load_data import load_data
from tqdm import tqdm

from src.classes.ClassFractionalCEV import FractionalCEV
from utils.distributions.CEV_multivar_posteriors import posteriors, generate_V_matrix, fBn_covariance_matrix
from utils.distributions.priors import prior
from utils.plotting_functions import plt, plot_subplots, gibbs_histogram_plot


def test_no_H(S=20000, muU=1., muX=1., gamma=1., X0=1., U0=0., H=0.8, N=2 ** 10, T=1e-3 * 2 ** 10,
              rng=np.random.default_rng(), loadData=False):
    sigmaX = np.sqrt(muX * gamma / 0.55)
    alpha = gamma / sigmaX
    deltaT = T / N
    if not loadData:
        m = FractionalCEV(muU=muU, alpha=alpha, muX=muX, sigmaX=sigmaX, X0=X0, U0=U0)
        Xs, Us = m.euler_simulation(H=H, N=N, deltaT=deltaT)  # Simulated data
        plot_subplots(np.arange(0, T + deltaT, step=deltaT), data=np.array([Xs, Us]), label_args=np.array([None, None]),
                      xlabels=np.array(["Time", "Time"]),
                      ylabels=np.array(["Volatility", "Log Price"]), globalTitle="Project Model Simulation",
                      saveTransparent=False)
        SH = np.power(deltaT, 2 * H) * fBn_covariance_matrix(N=N, H=H)
        invfBnCovMat = np.linalg.inv(SH)
        V_mat_dash = generate_V_matrix(vols=Xs, sigmaX=1., deltaT=deltaT, H=H, N=N, invfBnCovMat=invfBnCovMat)
    else:
        Xs, Us = load_data(N=N, H=H, T=T)
        SH = np.power(deltaT, 2 * H) * fBn_covariance_matrix(N=N, H=H)
        invfBnCovMat = np.linalg.inv(SH)
        V_mat_dash = generate_V_matrix(vols=Xs, sigmaX=1., deltaT=deltaT, H=H, N=N, invfBnCovMat=invfBnCovMat)
    # TODO: MAP Optimisation for param initialisation
    muUParams, gammaParams, muXParams, sigmaXParams = (0., 3.), (0., 1.), (0., .5), (2.1, np.power(1., 2) * 1.1)
    theta = prior(muUParams=muUParams, gammaParams=gammaParams, muXParams=muXParams, sigmaXParams=sigmaXParams)
    theta[4] = H
    Thetas = [theta]
    for _ in tqdm(range(S)):
        V_mat = V_mat_dash * np.power(theta[3], 2)
        theta = posteriors(muUParams=muUParams, gammaParams=gammaParams, muXParams=muXParams,
                           sigmaXParams=sigmaXParams, deltaT=deltaT, observations=Us, latents=Xs, theta=theta, V=V_mat,
                           rng=rng)
        Thetas.append(theta)
    Thetas = np.array(Thetas).reshape((S + 1, 5))
    burnOut = int(S / 10)
    Thetas[:, 3] *= Thetas[:, 3]
    gibbs_histogram_plot(Thetas, burnOut, plottitle="Fully Observed MCMC Histogram",
                         trueVals=[muU, gamma, muX, sigmaX ** 2],
                         priorParams=[muUParams, gammaParams, muXParams, sigmaXParams])
    plt.show()


test_no_H(loadData=True)
