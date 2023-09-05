import numpy as np
from src.load_data import load_data
from tqdm import tqdm

from src.classes.ClassFractionalCEV import FractionalCEV
from utils.distributions.CEV_multivar_posteriors_Zs import posteriors, generate_V_matrix, fBn_covariance_matrix
from utils.distributions.priors_Zs import prior
from utils.plotting_functions import plt, gibbs_histogram_plot, plot_subplots, plot_parameter_traces


def test_no_H(S=250000, muU=1., muX=1., gamma=1., X0=1., U0=0., H=0.8, N=2 ** 7, T=1e-3 * 2 ** 7,
              rng=np.random.default_rng(), loadData=False):
    sigmaX = np.sqrt(muX * gamma / 0.55)
    alpha = gamma / sigmaX
    deltaT = T / N
    m = FractionalCEV(muU=muU, alpha=alpha, muX=muX, sigmaX=sigmaX, X0=X0, U0=U0)
    if not loadData:
        Xs, Us = m.euler_simulation(H=H, N=N, deltaT=deltaT)  # Simulated data
        plot_subplots(np.arange(0, T + deltaT, step=deltaT), [Xs, Us], [None, None], ["Time", "Time"],
                      ["Volatility", "Log Price"],
                      "Project Model Simulation")
        plt.show()
        SH = np.power(deltaT, 2 * H) * fBn_covariance_matrix(N=N, H=H)
        invfBnCovMat = np.linalg.inv(SH)
        V_mat_dash = generate_V_matrix(vols=Xs, sigmaX=1., deltaT=deltaT, H=H, N=N, invfBnCovMat=invfBnCovMat)
    else:
        Xs, Us = load_data(N=N, H=H, T=T)
        SH = np.power(deltaT, 2 * H) * fBn_covariance_matrix(N=N, H=H)
        invfBnCovMat = np.linalg.inv(SH)
        V_mat_dash = generate_V_matrix(vols=Xs, sigmaX=1., deltaT=deltaT, H=H, N=N, invfBnCovMat=invfBnCovMat)

    Zs = m.lamperti(Xs)
    muUParams, alphaParams, muXParams, sigmaXParams = (0., 3.), (0., 2.), (0., 2.), (2.1, np.power(1., 2) * 3.1)
    theta = prior(muUParams=muUParams, alphaParams=alphaParams, muXParams=muXParams, sigmaXParams=sigmaXParams)
    theta[4] = H
    Thetas = [theta]
    sigmaXAcc = 0.
    for _ in tqdm(range(S)):
        theta, sigmaXAcc = posteriors(muUParams=muUParams, alphaParams=alphaParams,
                                      muXParams=muXParams,
                                      sigmaXParams=sigmaXParams, deltaT=deltaT, observations=Us,
                                      latents=Xs,
                                      transformedLatents=Zs, X0=X0,
                                      theta=theta, invfBnCovMat=invfBnCovMat, V=V_mat_dash,
                                      rng=rng, sigmaXAcc=sigmaXAcc)
        Thetas.append(theta)
    Thetas = np.array(Thetas).reshape((S + 1, 5))
    burnOut = int(S / 10)
    print("Alpha, ObsMean, SigmaX Acceptance Rates: " + str(
        sigmaXAcc / S))
    plot_parameter_traces(S=S, thetas=Thetas)
    # plot_autocorrfns(Thetas=Thetas)
    Thetas[:, 3] *= Thetas[:, 3]
    gibbs_histogram_plot(Thetas, burnOut, plottitle="Fully Observed MCMC Histogram",
                         trueVals=[muU, alpha, muX, sigmaX ** 2],
                         priorParams=[muUParams, alphaParams, muXParams, sigmaXParams])
    plt.show()


test_no_H(loadData=True)
