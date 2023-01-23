from utils.math_functions import np, logsumexp, snorm, gammaDist
from src.ClassFractionalCIR import ProjectModel
from src.ClassParticleFilter import ProjectParticleFilter
from src.ClassMCMC import MCMC
from tqdm import tqdm
from p_tqdm import t_map
from functools import partial
from copy import deepcopy
from utils.plotting_functions import plot_subplots, plot, plt
from src.priors import prior
from src.CIR_posteriors import posteriors


class ParticleMCMC(MCMC):
    def __init__(self, pf: ProjectParticleFilter, prior, proposalLLFnc, proposalFnc,
                 llFnc=None, rng=np.random.default_rng()):
        super().__init__(likelihoodFnc=llFnc, proposalLLFnc=proposalLLFnc, proposalFnc=proposalFnc, rng=rng)
        self.pf = pf
        self.prior = prior
