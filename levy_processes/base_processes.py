import numpy as np
from utils.maths_functions import hankel_squared, gammafnc, incgammal, get_z0_H0, gammainc, gammaincc, \
    gammaincinv, psi, dpsi, g, levy_stable


class LevyProcess:
    """
	Base class for all Levy processes
	"""

    @staticmethod
    def integrate(evaluation_points, t_series, x_series, drift=0.):
        """
		Static method for plotting paths on a discretised time axis
		"""
        W = [x_series[t_series < point].sum() + drift * point for point in evaluation_points]
        return np.array(W).T


class JumpLevyProcess(LevyProcess):
    """
	Specific class for handling pure jump processes
	"""

    def __init__(self, rng=np.random.default_rng()):
        self.rng = rng

    def accept_reject_simulation(self, h_func, thinning_func, rate, M, gamma_0, truncation):
        """
		Simulate jump sizes and times using poisson epochs, a jump function and a thinning function
		"""
        min_jump = np.inf
        x = []
        curr_epoch = gamma_0
        while min_jump >= truncation:
            epoch_seq = self.rng.exponential(scale=rate, size=M)
            epoch_seq[0] += curr_epoch
            epoch_seq = epoch_seq.cumsum()
            curr_epoch = epoch_seq[-1]
            x_seq = h_func(epoch_seq)
            min_jump = x_seq[-1]
            if min_jump < truncation:
                x.append(x_seq[x_seq >= truncation])
            else:
                x.append(x_seq)
            if truncation == 0.:
                break

        x = np.concatenate(x)
        acceptance_seq = thinning_func(x)
        u = self.rng.uniform(low=0., high=1., size=x.size)
        x = x[u < acceptance_seq]
        jtimes = self.rng.uniform(low=0., high=1. / rate, size=x.size)
        return jtimes, x

    def generate_marginal_samples(self, numSamples, tHorizon=1.0):
        return

    def unit_expected_residual_jumps(self, truncation):
        return 0.

    def unit_variance_residual_jumps(self, truncation):
        return 0.

    def small_jump_covariance(self, truncation, case=3):
        if case == 1:
            """ Truncated series with no covariance modelling """
            return 0.0, 0.0
        elif case == 2:
            """ Full Gaussian approximation """
            return self.unit_expected_residual_jumps(truncation), self.unit_variance_residual_jumps(truncation)
        elif case == 3:
            return self.unit_expected_residual_jumps(truncation), 0.0
        else:
            raise ValueError("Case number needs to be an integer between 1 and 3 inclusive")


class GammaProcess(JumpLevyProcess):
    """
	Pure jump Gamma process
	"""

    def __init__(self, beta=None, C=None, rng=np.random.default_rng()):
        self.set_parameters(beta, C)
        super().__init__(rng=rng)

    def set_parameters(self, beta, C):
        """
        Compared to Barndorff-Nielson
        beta = gamma**2/2
        C = ni
        """
        self.beta = beta
        self.C = C

    def get_parameters(self):
        return {"beta": self.beta, "C": self.C}

    def h_func(self, epoch):
        return 1. / (self.beta * (np.exp(epoch / self.C) - 1.))

    def thinning_func(self, x):
        return (1. + self.beta * x) * np.exp(-self.beta * x)

    def simulate_jumps(self, rate=1.0, M=1000, gamma_0=0.0, truncation=1e-6):
        return self.accept_reject_simulation(self.h_func, self.thinning_func, rate, M, gamma_0, truncation)

    def unit_expected_residual_jumps(self, truncation):
        return (self.C / self.beta) * incgammal(1., self.beta * truncation)

    def unit_variance_residual_jumps(self, truncation):
        return (self.C / self.beta ** 2) * incgammal(2., self.beta * truncation)

    def generate_marginal_samples(self, numSamples, tHorizon=1.0):
        return self.rng.gamma(shape=tHorizon * self.C, scale=1 / self.beta, size=numSamples)


class TemperedStableProcess(JumpLevyProcess):

    def __init__(self, alpha=None, beta=None, C=None, rng=np.random.default_rng()):
        self.set_parameters(alpha, beta, C)
        super().__init__(rng=rng)

    def set_parameters(self, alpha, beta, C):
        """
        Compared to Barndorff-Nielson
        alpha = kappa
        beta = gamma**(1/kappa)/2.0
        C  = delta * (2 ** kappa) * kappa * (1 / gammafnc(1 - kappa))
        """
        assert (0. < alpha < 1.)
        assert (C > 0.0 and beta >= 0.0)
        self.alpha = alpha
        self.beta = beta
        self.C = C

    def get_parameters(self):
        return {"alpha": self.alpha, "beta": self.beta, "C": self.C}

    def h_func(self, epoch):
        return np.power((self.alpha / self.C) * epoch, np.divide(-1., self.alpha))

    def thinning_func(self, x):
        return np.exp(-self.beta * x)

    def simulate_jumps(self, rate=1.0, M=1000, gamma_0=0., truncation=1e-6):
        return self.accept_reject_simulation(self.h_func, self.thinning_func, rate, M, gamma_0, truncation)

    def unit_expected_residual_jumps(self, truncation):
        """ Truncation is on jumps, not epochs """
        return (self.C * self.beta ** (self.alpha - 1.)) * incgammal(1. - self.alpha, self.beta * truncation)

    def unit_variance_residual_jumps(self, truncation):
        """ Truncation is on jumps, not epochs """
        return (self.C * self.beta ** (self.alpha - 2.)) * incgammal(2. - self.alpha, self.beta * truncation)

    def generate_marginal_samples(self, numSamples, tHorizon=1.0):
        kappa = self.alpha
        beta = self.beta
        delta = self.C * gammafnc(1 - kappa) / (kappa * (2 ** kappa))
        x = np.array([])
        # while (x.shape[0] < numSamples):
        x = np.append(x, levy_stable.rvs(kappa, beta=1.0, loc=0.0, scale=(tHorizon * delta) ** (1. / kappa),
                                         size=int(numSamples / 2)))
        prob_acc = np.exp(-beta * x)
        u = self.rng.uniform(0., 1., size=prob_acc.size)
        x = np.where(prob_acc > u, x, 0.)
        x = x[x > 0.]
        return x


class GIGProcess(JumpLevyProcess):

    def __init__(self, delta=None, gamma=None, lambd=None, rng=np.random.default_rng()):
        self.set_parameters(delta, gamma, lambd)
        super().__init__(rng=rng)

    def set_parameters(self, delta, gamma, lambd):
        """
        Compared to Barndorff-Nielson
        delta = delta
        gamma = gamma
        lambd  = nu
        """
        self.delta = delta
        self.gamma = gamma
        self.lambd = lambd

    def get_parameters(self):
        return {"delta": self.delta, "gamma": self.gamma, "lambd": self.lambd}

    def generate_marginal_samples(self, numSamples, tHorizon=1.0):
        """ Code is translated from MATLAB Code from:
            Jan Patrick Hartkopf (2022).
            gigrnd (https://www.mathworks.com/matlabcentral/fileexchange/78805-gigrnd),
            MATLAB Central File Exchange.
            Setup - - we sample from the two parameter version of the GIG(alpha, omega) where:
            P, a, b = lambd, gamma_param ** 2, delta ** 2,
        """

        """ Which parameter is scaled by TIME ??? """
        a = (self.gamma) ** 2
        b = (self.delta) ** 2
        lambd = (self.lambd)
        omega = np.sqrt(a * b)
        swap = False
        if lambd < 0:
            lambd = lambd * -1
            swap = True
        alpha = np.sqrt(omega ** 2 + lambd ** 2) - lambd
        x = -psi(1, alpha, lambd)
        if (x >= 0.5) and (x <= 2):
            t = 1
        elif x > 2:
            t = np.sqrt(2 / (alpha + lambd))
        elif x < 0.5:
            t = np.log(4 / (alpha + 2 * lambd))

        x = -psi(-1, alpha, lambd)
        if (x >= 0.5) and (x <= 2):
            s = 1
        elif x > 2:
            s = np.sqrt(4 / (alpha * np.cosh(1) + lambd))
        elif x < 0.5:
            s = min(1 / lambd, np.log(1 + 1 / alpha + np.sqrt(1 / alpha ** 2 + 2 / alpha)))

        eta = -psi(t, alpha, lambd)
        zeta = -dpsi(t, alpha, lambd)
        theta = -psi(-s, alpha, lambd)
        xi = dpsi(-s, alpha, lambd)
        p = 1 / xi
        r = 1 / zeta
        td = t - r * eta
        sd = s - p * theta
        q = td + sd

        X = [0 for _ in range(numSamples)]
        for i in range(numSamples):
            done = False
            while not done:
                U = self.rng.uniform(0., 1., size=1)
                V = self.rng.uniform(0., 1., size=1)
                W = self.rng.uniform(0., 1., size=1)
                if U < (q / (p + q + r)):
                    X[i] = -sd + q * V
                elif U < ((q + r) / (p + q + r)):
                    X[i] = td - r * np.log(V)
                else:
                    X[i] = -sd + p * np.log(V)
                f1 = np.exp(-eta - zeta * (X[i] - t))
                f2 = np.exp(-theta + xi * (X[i] + s))
                if (W * g(X[i], sd, td, f1, f2)) <= np.exp(psi(X[i], alpha, lambd)):
                    done = True
        X = np.exp(X) * (lambd / omega + np.sqrt(1 + (lambd / omega) ** 2))
        if swap:
            X = 1 / X
        X = X / np.sqrt(a / b)
        X = X.reshape((1, X.shape[0]))
        return X[0]

    def simulate_jumps(self, rate=1., M=2000, gamma_0=0., truncation=0.):
        if np.abs(self.lambd) >= 0.5:
            simulator = self.SimpleSimulator(self, rng=self.rng)
            jtimes, jsizes = simulator.simulate_internal_jumps(rate, M, gamma_0, truncation)
        else:
            z0, H0 = get_z0_H0(self.lambd)
            simulator1 = self.__N1(self, z0, H0, rng=self.rng)
            simulator2 = self.__N2(self, z0, H0, rng=self.rng)
            jtimes1, jsizes1 = simulator1.simulate_internal_jumps(rate, M, gamma_0, truncation)
            jtimes2, jsizes2 = simulator2.simulate_internal_jumps(rate, M, gamma_0, truncation)
            jtimes = np.append(jtimes1, jtimes2)
            jsizes = np.append(jsizes1, jsizes2)

        if self.lambd > 0.:
            # simulate gamma component
            simulator = GammaProcess(beta=self.gamma ** 2 / 2., C=self.lambd)
            e_jtimes, e_jsizes = simulator.simulate_jumps(rate=rate, M=M, gamma_0=gamma_0, truncation=truncation)
            jtimes = np.append(jtimes, e_jtimes)
            jsizes = np.append(jsizes, e_jsizes)
        return jtimes, jsizes

    def unit_expected_residual_jumps(self, truncation):
        lambd = self.lambd
        delta = self.delta
        if np.abs(lambd) < 0.5:
            return delta * np.sqrt((2 * truncation) / np.pi)
        else:
            a = np.pi * np.power(2.0, (1.0 - 2.0 * np.abs(lambd)))
            b = gammafnc(np.abs(lambd)) ** 2
            c = 1 / (1 - 2 * np.abs(lambd))
            z1 = (a / b) ** c
            H0 = z1 * hankel_squared(np.abs(lambd), z1)
            return 2 * delta * np.sqrt((2 * truncation) / np.pi) / (np.pi * H0)

    def unit_variance_residual_jumps(self, truncation):
        lambd = self.lambd
        delta = self.delta
        if np.abs(lambd) < 0.5:
            return delta * truncation * np.sqrt((2. * truncation) / np.pi) / 3.
        else:
            a = np.pi * np.power(2.0, (1.0 - 2.0 * np.abs(lambd)))
            b = gammafnc(np.abs(lambd)) ** 2
            c = 1 / (1 - 2 * np.abs(lambd))
            z1 = (a / b) ** c
            H0 = z1 * hankel_squared(np.abs(lambd), z1)
            return 2 * delta * truncation * np.sqrt((2. * truncation) / np.pi) / (3. * np.pi * H0)

    class SimpleSimulator(JumpLevyProcess):
        def __init__(self, outer, rng=np.random.default_rng()):
            self.outer = outer
            super().__init__(rng=rng)
            self.tsp_generator = TemperedStableProcess(alpha=0.5, beta=outer.gamma ** 2 / 2,
                                                       C=outer.delta * gammafnc(0.5) / (np.sqrt(2.) * np.pi),
                                                       rng=outer.rng)

        def __generate_z(self, x):
            return np.sqrt(self.rng.gamma(shape=0.5, scale=(2. * self.outer.delta ** 2) / x))

        def thinning_func(self, z):
            return 2. / (np.pi * z * hankel_squared(np.abs(self.outer.lambd), z))

        def accept_reject_simulation(self, x, z, thinning_func, rate):
            assert (x.shape == z.shape)
            acceptance_seq = thinning_func(z)
            u = self.rng.uniform(low=0.0, high=1.0, size=x.size)
            x_acc = x[u < acceptance_seq]
            times = self.rng.uniform(low=0.0, high=1. / rate, size=x_acc.size)
            return times, x_acc

        def simulate_internal_jumps(self, rate, M, gamma_0, truncation):
            _, x = self.tsp_generator.simulate_jumps(rate, M, gamma_0, truncation=truncation)
            z = self.__generate_z(x)
            jtimes, jsizes = self.accept_reject_simulation(x, z, thinning_func=self.thinning_func, rate=rate)
            return jtimes, jsizes

    class __N1(SimpleSimulator):
        def __init__(self, outer, z0, H0, rng=np.random.default_rng()):
            self.outer = outer  # Outer is GIGProcess object
            super().__init__(outer, rng=rng)
            self.q1 = self.__Q1(outer, z0, H0, rng=rng)
            self.z0 = z0
            self.H0 = H0

        def __generate_z(self, x):
            # sim from truncated square root gamma density
            lambd = self.outer.lambd
            delta = self.outer.delta
            C1 = self.rng.uniform(0., 1., size=x.size)
            l = C1 * gammainc(np.absolute(lambd), (self.z0 ** 2 * x) / (2 * delta ** 2))
            zs = np.sqrt(((2 * delta ** 2) / x) * gammaincinv(np.absolute(lambd), l))
            return zs

        def thinning_func(self, z):
            # thin according to algo 4 step 4
            lambd = self.outer.lambd
            return self.H0 / (hankel_squared(np.abs(lambd), z) *
                              (z ** (2. * np.abs(lambd))) / (self.z0 ** (2 * np.abs(lambd) - 1)))

        def simulate_internal_jumps(self, rate, M, gamma_0, truncation):
            _, x = self.q1.simulate_jumps(rate, M, gamma_0, truncation)
            _, x = self.accept_reject_simulation(x, x, thinning_func=lambda xs: (
                        np.abs(self.outer.lambd) * incgammal(np.abs(self.outer.lambd), (self.z0 ** 2) * xs / (
                        2. * (self.outer.delta ** 2))) / np.power(
                    (self.z0 ** 2) * xs / (2. * self.outer.delta ** 2), np.abs(self.outer.lambd))), rate=rate)
            z = self.__generate_z(x)
            jtimes, jsizes = self.accept_reject_simulation(x, z, thinning_func=self.thinning_func, rate=rate)
            return jtimes, jsizes

        class __Q1(GammaProcess):
            def __init__(self, outer, z0, H0, rng=np.random.default_rng()):
                super().__init__(beta=(outer.gamma ** 2) / 2.,
                                 C=z0 / (np.pi * np.pi * H0 * np.abs(outer.lambd)))
                self.outer = outer  # Outer is GIGProcess Object

    class __N2(SimpleSimulator):
        def __init__(self, outer, z0, H0, rng=np.random.default_rng()):
            self.outer = outer
            super().__init__(outer, rng=rng)
            self.q2 = self.__Q2(outer, z0, H0, rng=rng)
            self.z0 = z0
            self.H0 = H0

        def __generate_z(self, x):
            # sim from truncated square root gamma density (algo 5 step 3)
            delta = self.outer.delta
            z0 = self.z0
            C2 = self.rng.uniform(low=0.0, high=1.0, size=x.size)
            return np.sqrt(
                ((2 * delta ** 2) / x) * gammaincinv(0.5, C2 * (gammaincc(0.5, (z0 ** 2) * x / (2 * delta ** 2)))
                                                     + gammainc(0.5, (z0 ** 2) * x / (2 * delta ** 2))))

        def thinning_func(self, z):
            # thin according to algo 5 step 4
            return self.H0 / (z * hankel_squared(np.abs(self.outer.lambd), z))

        def simulate_internal_jumps(self, rate, M, gamma_0, truncation):
            _, x = self.q2.simulate_jumps(rate, M, gamma_0, truncation)
            _, x = self.accept_reject_simulation(x, x, thinning_func=lambda xs: gammaincc(0.5, (self.z0 ** 2) * xs / (
                    2 * self.outer.delta ** 2)), rate=rate)
            z = self.__generate_z(x)
            jtimes, jsizes = self.accept_reject_simulation(x, z, thinning_func=self.thinning_func, rate=rate)
            return jtimes, jsizes

        class __Q2(TemperedStableProcess):
            def __init__(self, outer, z0, H0, rng=np.random.default_rng()):
                super().__init__(beta=(outer.gamma ** 2) / 2., alpha=0.5,
                                 C=np.sqrt(2 * outer.delta ** 2) * gammafnc(0.5) / ((np.pi ** 2) * H0))
