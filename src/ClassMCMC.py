from utils.math_functions import np, multsnorm, snorm
from utils.plotting_functions import plot, plt


class MCMC:
    def __init__(self, likelihoodFnc, proposalLLFnc, proposalFnc, rng=np.random.default_rng()):
        self.rng = rng
        self.likelihood = likelihoodFnc  # TODO: Make sure this is a pointer to a function!
        self.proposal = proposalFnc
        self.proposal_likelihood = proposalLLFnc

    def accept_reject(self, newP, currentP):
        logProbProp = np.log(self.proposal_likelihood(currentP, newP)) - np.log(
            self.proposal_likelihood(newP, currentP))
        logProbLL = np.log(self.likelihood(newP)) - np.log(self.likelihood(currentP))
        logAccProb = logProbProp + logProbLL
        u = self.rng.uniform(0., 1., size=1)
        if u < np.exp(min(0., logAccProb)):
            return newP
        return currentP


class pCnMCMC(MCMC):
    def __init__(self, likelihoodFnc, rng=np.random.default_rng()):
        super().__init__(likelihoodFnc=likelihoodFnc, proposalFnc=self.proposal, proposalLLFnc=self.proposal_likelihood,
                         rng=rng)

    def proposal(self, current, rho=0.254):
        return np.sqrt(
            1 - np.power(rho, 2)) * current + rho * np.atleast_2d(
            self.rng.normal(size=current.shape[0])).T

    @staticmethod
    def proposal_likelihood(x, xd, rho=0.1):
        assert (len(x.shape) > 1 and len(x.shape) == len(xd.shape))
        r2 = np.power(rho, 2)
        return multsnorm.pdf(np.squeeze(x), mean=np.sqrt(1 - r2) * np.squeeze(xd), cov=r2 * np.eye(x.shape[0]))


def ll(x):
    assert (len(x.shape) > 1)
    return multsnorm.pdf(np.squeeze(x), mean=None, cov=np.eye(x.shape[0]))


def proposal(current, rng):
    return current + 0.1 * np.atleast_2d(rng.normal(current.shape[0])).T


def proposalLL(x, xd):
    return multsnorm.pdf(np.squeeze(x), mean=np.squeeze(xd), cov=(0.1 ** 2) * np.eye(xd.shape[0]))


def test():
    #  Doesn't seem to work with my choice of proposal and proposalLL why????
    # m = MCMC(ll, proposalFnc=proposal, proposalLLFnc=proposalLL)
    m = pCnMCMC(ll)
    x = np.ones(shape=(1, 1)) + np.atleast_2d(m.rng.normal(size=1)).T
    Xs = [x[0, 0]]
    T = 10000
    for _ in range(T):
        newP = m.proposal(current=x)
        x = m.accept_reject(newP, x)
        Xs.append(x[0, 0])
    plot(np.linspace(0, T + 1, T + 1), [Xs], label_args=[None], xlabel="Time",
                  ylabel="State Position", title= "pCN MCMC on Standard Normal Distribution")

    plt.show()


#test()
