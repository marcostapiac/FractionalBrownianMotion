from tqdm import tqdm

from src.CEV_multivar_posteriors import posteriors, fBn_covariance_matrix
from src.CEV_multivar_posteriors_Zs import posteriors, fBn_covariance_matrix
from src.ClassFractionalCEV import FractionalCEV
from src.load_data import load_data
from src.priors import prior
from src.priors_Zs import prior
from utils.math_functions import np
from utils.plotting_functions import plt, gibbs_histogram_plot, plot_subplots


def test_no_H(S=30000, muU=1., muX=1., gamma=1., X0=1., U0=0., H=0.8, N=2 ** 9, T=1e-3 * 2 ** 9,
              rng=np.random.default_rng(), loadData=False):
    sigmaX = np.sqrt(muX * gamma / 0.55)
    print(sigmaX)
    alpha = gamma / sigmaX
    deltaT = T / N
    m = FractionalCEV(muU=muU, gamma=gamma, muX=muX, sigmaX=sigmaX, X0=X0, U0=U0)
    if not loadData:
        Xs, Us = m.euler_simulation(H=H, N=N, deltaT=deltaT)  # Simulated data
        plot_subplots(np.arange(0, T + deltaT, step=deltaT), [Xs, Us], [None, None], ["Time", "Time"],
                      ["Volatility", "Log Price"],
                      "Project Model Simulation")
        plt.show()
        SH = np.power(deltaT, 2 * H) * fBn_covariance_matrix(N=N, H=H)
        invfBnCovMat = np.linalg.inv(SH)
    else:
        Xs, Us = load_data(N=N, H=H, T=T)
        SH = np.power(deltaT, 2 * H) * fBn_covariance_matrix(N=N, H=H)
        invfBnCovMat = np.linalg.inv(SH)
    muUParams, alphaParams, muXParams, sigmaXParams = (0., 3.), (0., 2.), (0., 2.), (2.1, np.power(1., 2) * 3.1)
    theta = prior(muUParams=muUParams, alphaParams=alphaParams, muXParams=muXParams, sigmaXParams=sigmaXParams)
    theta[4] = H
    Thetas = [theta]
    m = FractionalCEV(muU=theta[0], gamma=theta[1] * theta[3], muX=theta[2], sigmaX=theta[3], X0=X0, U0=U0)
    Xs = m.state_simulation(H=theta[4], N=N, deltaT=deltaT)
    Zs = m.lamperti(Xs)
    alphaAcc, volAcc, sigmaXAcc = 0., 0., 0.
    for _ in tqdm(range(S)):
        theta, alphaAcc, volAcc, sigmaXAcc = posteriors(muUParams=muUParams, alphaParams=alphaParams,
                                                        muXParams=muXParams,
                                                        sigmaXParams=sigmaXParams, deltaT=deltaT, observations=Us,
                                                        latents=Xs,
                                                        transformedLatents=Zs, X0=X0,
                                                        theta=theta, invfBnCovMat=invfBnCovMat,
                                                        rng=rng, alphaAcc=alphaAcc, volAcc=volAcc, sigmaXAcc=sigmaXAcc)
        Thetas.append(theta)
        if max(Xs) > 10:
            print(theta)
        m.__init__(muU=theta[0], gamma=theta[1] * theta[3], muX=theta[2], sigmaX=theta[3], X0=X0, U0=U0)
        Xs = m.state_simulation(H=theta[4], N=N, deltaT=deltaT)
        Zs = m.lamperti(Xs)
    Thetas = np.array(Thetas).reshape((S + 1, 5))
    burnOut = int(S / 10)
    print("Alpha, ObsMean, SigmaX Acceptance Rates: " + str(alphaAcc / S) + ", " + str(volAcc / S) + ", " + str(
        sigmaXAcc / S))
    # plot_parameter_traces(S=S, Thetas=Thetas)
    # plot_autocorrfns(Thetas=Thetas)
    Thetas[:, 3] *= Thetas[:, 3]
    gibbs_histogram_plot(Thetas, burnOut, plottitle="Partially Observed Gibbs Histogram", trueVals=[muU, alpha, muX, sigmaX ** 2],
                         priorParams=[muUParams, alphaParams, muXParams, sigmaXParams])
    plt.show()


test_no_H(loadData=False)
