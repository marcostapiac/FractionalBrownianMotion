from tqdm import tqdm

from src.ClassFractionalBrownianNoise import FractionalBrownianNoise
from utils.math_functions import np, chi2


def standardise_sample(fBn_sample, invL):
    return np.squeeze(invL @ fBn_sample)


def chiSquared(fBn_sample, invL):
    standard_sample = standardise_sample(fBn_sample, invL)
    ts = np.sum([i ** 2 for i in standard_sample])
    return ts


def chisquared_test(N, H, M):
    ts = []
    fbn = FractionalBrownianNoise(H)
    invL = np.linalg.inv(
        np.linalg.cholesky(np.atleast_2d(
            [[fbn.covariance(i - j) for j in range(N)] for i in range(N)])))
    alpha = 0.05
    crit = chi2.ppf(q=1 - alpha, df=N - 1)  # Upper alpha quantile, and dOf = N - 1
    for _ in tqdm(range(M)):
        Z = fbn.davies_and_harte_simulation(N)
        tss = chiSquared(Z, invL)
        ts.append(tss)
    return np.mean(ts), crit


if __name__ == "__main__":
    print(chisquared_test(N=2 ** 10, H=0.3, M=1000))
