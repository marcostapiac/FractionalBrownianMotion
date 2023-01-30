from tqdm import tqdm

from src.CEV_multivar_posteriors import posteriors, generate_V_matrix
from src.ClassFractionalCEV import FractionalCEV
from src.load_data import load_data, load_fBn_covariance
from src.priors import prior
from utils.math_functions import np
from utils.plotting_functions import plt, plot_subplots, gibbs_histogram_plot


def test_no_H(S=20000, muU=1., muX=1., gamma=1., X0=1., U0=0., H=0.8, N=2 ** 12, T=1e-3 * 2 ** 12,
              rng=np.random.default_rng(), loadData=False):
    sigmaX = np.sqrt(muX * gamma / 0.55)
    deltaT = T / N
    if not loadData:
        m = FractionalCEV(muU=muU, gamma=gamma, muX=muX, sigmaX=sigmaX, X0=X0, U0=U0)
        Xs, Us = m.euler_simulation(H=H, N=N, deltaT=deltaT)  # Simulated data
        plot_subplots(np.arange(0, T + deltaT, step=deltaT), [Xs, Us], [None, None], ["Time", "Time"],
                      ["Volatility", "Log Price"], "Project Model Simulation")
        plt.show()
        V_mat_dash = generate_V_matrix(vols=Xs, sigmaX=1., deltaT=deltaT, H=H, N=N)
    else:
        Xs, Us = load_data(N=N, H=H, T=T)
        V_mat_dash = generate_V_matrix(vols=Xs, sigmaX=1., deltaT=deltaT, H=H, N=N, SH=load_fBn_covariance(N=N, H=H))
    # TODO: MAP Optimisation for param initialisation
    muUParams, gammaParams, muXParams, sigmaXParams = (0., 3.), (0., 1.), (0., .5), (2.1, np.power(1., 2) * 1.1)
    theta = prior(muUParams=muUParams, gammaParams=gammaParams, muXParams=muXParams, sigmaXParams=sigmaXParams)
    theta[4] = H
    theta[1] = gamma
    theta[3] = sigmaX
    Thetas = [theta]
    for _ in tqdm(range(S)):
        V_mat = V_mat_dash * np.power(theta[3], 2)
        theta = posteriors(muUParams=muUParams, gammaParams=gammaParams, muXParams=muXParams,
                           sigmaXParams=sigmaXParams, deltaT=deltaT, observations=Us, latents=Xs, theta=theta, V=V_mat,
                           rng=rng)
        Thetas.append(theta)
    Thetas = np.array(Thetas).reshape((S + 1, 5))
    burnOut = 1000
    # plot_parameter_traces(S=S, Thetas=Thetas)
    # plot_autocorrfns(Thetas=Thetas)
    gibbs_histogram_plot(Thetas, burnOut, trueVals=[muU, gamma, muX, sigmaX],
                         priorParams=[muUParams, gammaParams, muXParams, sigmaXParams])
    plt.show()


test_no_H(loadData=False)
