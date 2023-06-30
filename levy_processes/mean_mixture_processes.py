import numpy as np
from processes import base_processes


class MeanMixtureLevyProcess(base_processes.LevyProcess):
    def __init__(self, mu, mu_W, var_W, subordinator, rng=np.random.default_rng()):
        self.rng = rng
        self.mu = mu
        self.mu_W = mu_W
        self.var_W = var_W
        self.std_W = np.sqrt(var_W)
        self.subordinator = subordinator

    def get_mu_W(self):
        return self.mu_W

    def get_var_W(self):
        return self.var_W

    def simulate_jumps(self, rate=1.0, M=2000, gamma_0=0., truncation=1e-6):
        subordinator_sample = self.subordinator.simulate_jumps(rate, M, gamma_0, truncation=truncation)
        jtimes = subordinator_sample[0]
        jsizes = self.mu_W * subordinator_sample[1] + self.std_W * np.sqrt(subordinator_sample[1]) * self.rng.normal(
            size=subordinator_sample[1].shape)
        return jtimes, jsizes

    def simulate_small_jumps(self, rate=1.0, M=2000, gamma_0=0., truncation=1e-6):
        subordinator_sample = self.subordinator.simulate_jumps(rate, M, gamma_0, truncation=0.)
        jtimes = subordinator_sample[0]
        jumps = subordinator_sample[1]
        sub_jumps = jumps[jumps < truncation]
        jsizes = self.mu_W * sub_jumps + self.std_W * np.sqrt(sub_jumps) * self.rng.normal(
            size=sub_jumps.shape)
        return jtimes, jsizes

    def simulate_path(self, observation_times):
        mean_mix_jtimes, mean_mix_jsizes = self.simulate_jumps()
        integrated_process = self.integrate(observation_times, mean_mix_jtimes, mean_mix_jsizes, drift=self.mu)

        return integrated_process


class NormalTemperedStableProcess(MeanMixtureLevyProcess):

    def __init__(self, alpha, beta, C, mu, mu_W, var_W, rng=np.random.default_rng()):
        self.tsp = base_processes.TemperedStableProcess(alpha=alpha, beta=beta, C=C, rng=rng)
        super().__init__(mu=mu, mu_W=mu_W, var_W=var_W, subordinator=self.tsp, rng=rng)


class NormalGammaProcess(MeanMixtureLevyProcess):

    def __init__(self, beta, C, mu, mu_W, var_W, rng=np.random.default_rng()):
        self.gp = base_processes.GammaProcess(beta=beta, C=C, rng=rng)
        super().__init__(mu=mu, mu_W=mu_W, var_W=var_W, subordinator=self.gp, rng=rng)


class GeneralHyperbolicProcess(MeanMixtureLevyProcess):
    # using GIG process
    def __init__(self, delta, gamma, lambd, mu, mu_W, var_W, rng=np.random.default_rng()):
        self.gig = base_processes.GIGProcess(delta=delta, gamma=gamma, lambd=lambd, rng=rng)
        super().__init__(mu=mu, mu_W=mu_W, var_W=var_W, subordinator=self.gig, rng=rng)
