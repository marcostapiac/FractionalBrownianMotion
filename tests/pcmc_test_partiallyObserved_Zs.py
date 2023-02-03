from tqdm import tqdm

from src.CEV_multivar_posteriors_Zs import posteriors, fBn_covariance_matrix, generate_V_matrix, latents_MH
from src.ClassFractionalCEV import FractionalCEV
from src.load_data import load_data
from src.store_data import store_data
from src.priors_Zs import prior
from utils.math_functions import np
from utils.plotting_functions import plt, gibbs_histogram_plot, plot_subplots, histogramplot, plot_parameter_traces, \
    plot


def test_no_H(S=300000, muU=1., muX=1., gamma=1., X0=1., U0=0., H=0.8, N=2 ** 7, T=1e-3 * 2 ** 7,
              rng=np.random.default_rng(), loadData=False):
    sigmaX = np.sqrt(muX * gamma / 0.55)
    alpha = gamma / sigmaX
    deltaT = T / N
    U0 = X0 + np.sqrt(deltaT) * rng.normal()
    m = FractionalCEV(muU=muU, alpha=alpha, muX=muX, sigmaX=sigmaX, X0=X0, U0=U0)
    if not loadData:
        Xs, Us = m.euler_simulation(H=H, N=N, deltaT=deltaT)  # Simulated data
        store_data(Xs=Xs, Us=Us, muU=muU, muX=muX, gamma=gamma, X0=X0, U0=U0, H=H, N=N, T=T)
        plot_subplots(np.arange(0, T + deltaT, step=deltaT), [Xs, Us], [None, None], ["Time", "Time"],
                      ["Volatility", "Log Price"],
                      "Project Model Simulation")
        plt.show()
        SH = np.power(deltaT, 2 * H) * fBn_covariance_matrix(N=N, H=H)
        invfBnCovMat = np.linalg.inv(SH)
    else:
        Xs, Us = load_data(N=N, H=H, T=T, isSimple=True)
        SH = np.power(deltaT, 2 * H) * fBn_covariance_matrix(N=N, H=H)
        invfBnCovMat = np.linalg.inv(SH)
    muUParams, alphaParams, muXParams, sigmaXParams = (0., 3.), (0., 2.), (0., 2.), (2.1, np.power(1., 2) * 3.1)
    theta = prior(muUParams=muUParams, alphaParams=alphaParams, muXParams=muXParams, sigmaXParams=sigmaXParams)
    theta[4] = H
    Thetas = np.reshape([theta], (1, 5))
    m = FractionalCEV(muU=theta[0], alpha=theta[1], muX=theta[2], sigmaX=theta[3], X0=X0, U0=U0)
    Xs = m.state_simulation(H=theta[4], N=N, deltaT=deltaT)
    Zs = m.lamperti(Xs)
    print(theta[3])
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
        if i>0 and i % 10000 == 1:
            print(sigmaXAcc/i)
            fig, ax, binVals = histogramplot(Thetas[int(i/8):, 3], xlabel="Volatility Variance",
                                             ylabel="PDF",
                                             plottitle="plottitle")
            ax.axvline(sigmaX, label="True Parameter Value $ " + str(round(sigmaX, 3)) + " $", color="blue")
            plt.show()
        m, Xs, Zs = latents_MH(i, generator=m, X0=X0, U0=U0, latents=Xs, transformedLatents=Zs, deltaT=deltaT, N=N,
                               theta=theta, observations=Us,
                               rng=rng, invfBnCovMat=invfBnCovMat)

    burnOut = int(S / 8)
    print("SigmaX Acceptance Rates: " + str(sigmaXAcc / S))
    plot_parameter_traces(S=S, Thetas=Thetas)
    # plot_autocorrfns(Thetas=Thetas)
    Thetas[:, 3] *= Thetas[:, 3]
    gibbs_histogram_plot(Thetas, burnOut, plottitle="Partially Observed MCMC Histogram",
                         trueVals=[muU, alpha, muX, sigmaX ** 2],
                         priorParams=[muUParams, alphaParams, muXParams, sigmaXParams])
    plt.show()


test_no_H(loadData=True)
