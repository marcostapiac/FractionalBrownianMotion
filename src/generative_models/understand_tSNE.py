from tqdm import tqdm

from src.classes.ClassFractionalBrownianNoise import FractionalBrownianNoise
from src.classes.ClassFractionalCEV import FractionalCEV
from utils import config
from utils.math_functions import generate_CEV, generate_fBm, generate_fBn
from utils.plotting_functions import plot_tSNE, plot_diffusion_marginals
import numpy as np

if __name__ == "__main__":
    h, td, S, rng = 0.7, 256, 200000, np.random.default_rng()
    generator = FractionalBrownianNoise(H=h, rng=rng)
    samplesfBn = np.zeros((S, td))
    for i in tqdm(range(S)):
        samplesfBn[i,:] = generator.circulant_simulation(N_samples=td)
    np.save(config.ROOT_DIR + "data/two_hundred_thousand_fBn_samples_H{}_T{}.npy".format(h, td), samplesfBn)

    S = 200000
    muU = 1.
    muX = 2.
    alpha = 1.
    sigmaX = 0.5
    X0 = 1.
    U0 = 0.
    cevGen = FractionalCEV(muU=muU, alpha=alpha, sigmaX=sigmaX, muX=muX, X0=X0, U0=U0, rng=rng)
    samplesCEV = np.zeros((S, td))
    for i in tqdm(range(S)):
        samplesCEV[i, :] = cevGen.state_simulation(H=h, N=td, deltaT=1. / td)[1:]
    np.save(
        config.ROOT_DIR + "data/two_hundred_thousand_CEV_samples_H{}_T{}_alpha{}_sigmaX{}_muX{}.npy".format(
            h, td, alpha, sigmaX, muX), samplesCEV)