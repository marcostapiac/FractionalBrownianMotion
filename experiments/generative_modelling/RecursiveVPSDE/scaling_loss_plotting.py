import matplotlib
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('ggplot')
matplotlib.rcParams.update({
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
    'text.latex.preamble': r"\usepackage{amsmath}"
})
if __name__=="__main__":
    diff_times = np.linspace(1e-3, 1, 100000)[:50000]
    beta_min = 1e-4
    beta_max = 20.
    eff_times = beta_min*diff_times+0.5*diff_times**2*(beta_max-beta_min)
    beta_tau = np.exp(-0.5*eff_times)
    beta_2_tau = np.exp(-eff_times)
    sigma_2_tau = 1.-np.exp(-eff_times)
    plt.plot(diff_times, sigma_2_tau, label="$\lambda(\\tau)=\sigma_{\\tau}^{2}$")
    plt.plot(diff_times, (sigma_2_tau**2/beta_2_tau), label="$\lambda(\\tau)=\\frac{\sigma_{\\tau}^{4}}{\\beta_{\\tau}^{2}}$")
    plt.title("Standard vs Proposed Loss Weighting Function")
    plt.legend()
    plt.show()