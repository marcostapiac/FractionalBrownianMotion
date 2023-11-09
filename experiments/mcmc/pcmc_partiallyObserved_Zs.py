import numpy as np
from tqdm import tqdm

from src.classes.ClassFractionalCEV import FractionalCEV
from utils.distributions.CEV_multivar_posteriors_Zs import posteriors, fBn_covariance_matrix, latents_MH
from utils.distributions.priors_Zs import prior
from utils.plotting_functions import plt, gibbs_histogram_plot, plot_subplots, plot_histogram, plot_parameter_traces


def test_no_H(S=1000000, muU=1., muX=1., gamma=1., X0=1., U0=0., H=0.8, N=2 ** 7, T=1e-3 * 2 ** 7,
              rng=np.random.default_rng()):
    sigmaX = np.sqrt(muX * gamma / 0.55)
    alpha = gamma / sigmaX
    deltaT = T / N
    U0 = X0 + np.sqrt(deltaT) * rng.normal()
    m = FractionalCEV(muU=muU, alpha=alpha, muX=muX, sigmaX=sigmaX, X0=X0, U0=U0)
    Xs, Us = m.euler_simulation(H=H, N=N, deltaT=deltaT)  # Simulated data
    plot_subplots(np.arange(0, T + deltaT, step=deltaT), np.array([Xs, Us]), np.array([None, None]),
                  np.array(["Time", "Time"]),
                  np.array(["Volatility", "Log Price"]),
                  "Project Model Simulation")
    plt.show()
    SH = np.power(deltaT, 2 * H) * fBn_covariance_matrix(N=N, H=H)
    invfBnCovMat = np.linalg.inv(SH)

    muUParams, alphaParams, muXParams, sigmaXParams = (0., 3.), (0., 2.), (0., 2.), (2.1, np.power(1., 2) * 3.1)
    theta = prior(muUParams=muUParams, alphaParams=alphaParams, muXParams=muXParams, sigmaXParams=sigmaXParams)
    theta[4] = H
    Thetas = np.reshape([theta], (1, 5))
    m = FractionalCEV(muU=theta[0], alpha=theta[1], muX=theta[2], sigmaX=theta[3], X0=X0, U0=U0)
    Xs = m.state_simulation(H=theta[4], N=N, deltaT=deltaT)
    Zs = m.lamperti(Xs)
    sigmaXAcc = 0.
    for i in tqdm(range(S)):
        theta, sigmaXAcc = posteriors(muUParams=muUParams, alphaParams=alphaParams,
                                      muXParams=muXParams,
                                      sigmaXParams=sigmaXParams, deltaT=deltaT, observations=Us,
                                      latents=Xs,
                                      transformedLatents=Zs, X0=X0,
                                      theta=theta, invfBnCovMat=invfBnCovMat,
                                      rng=rng, sigmaXAcc=sigmaXAcc)
        Thetas = np.vstack([Thetas, theta])
        if i > 0 and i % 10000 == 1:
            fig, ax, binVals = plot_histogram(Thetas[int(i / 8):, 3], xlabel="Volatility Variance", ylabel="PDF",
                                              plottitle="Histogram of Volatility Variance")
            ax.axvline(sigmaX, label="True Parameter Value $ " + str(round(sigmaX, 3)) + " $", color="blue")
            plt.show()
        m, Xs, Zs = latents_MH(i, generator=m, X0=X0, U0=U0, latents=Xs, transformedLatents=Zs, deltaT=deltaT, N=N,
                               theta=theta, observations=Us,
                               rng=rng, invfBnCovMat=invfBnCovMat)

    burnOut = int(S / 8)
    print("SigmaX Acceptance Rates: " + str(sigmaXAcc / S))
    Thetas[:, 3] *= Thetas[:, 3]
    plot_parameter_traces(S=S, thetas=Thetas)
    gibbs_histogram_plot(Thetas, burnOut, titlePlot="Partially Observed MCMC Histogram",
                         trueVals=np.array([muU, alpha, muX, sigmaX ** 2]),
                         priorParams=np.array([muUParams, alphaParams, muXParams, sigmaXParams]))
    plt.show()


test_no_H()
