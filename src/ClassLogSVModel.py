from utils.math_functions import np, pd
from utils.plotting_functions import plt, plot


class LogSVModel:

    def __init__(self, a, b, s, isStationary=True):
        self.a = a
        self.b = b
        self.s = s
        if isStationary:
            self.logVol2t = a / (1. - b)  # Long term variance (no information, only valid for -1<=b<=1)
        else:
            # TODO
            self.logVol2t = a / (1. - b)

    def get_log_vol(self):
        return self.logVol2t

    def __update_log_vol(self):
        self.logVol2t = self.a + self.b * self.logVol2t + self.s * np.random.normal()  # log(sigma_{t}^{2})

    def __next_price(self):
        return np.exp(0.5 * self.logVol2t) * np.random.normal()

    def simulate_log_vols(self, T):
        logVolData = []  # TODO: Ensure long term variance is used OR BWD forecast is used
        for i in range(T):
            self.__update_log_vol()
            logVolData.append(self.get_log_vol())
        return np.array(logVolData)

    def simulate_obs(self, T_horizon):
        if self.get_log_vol() != self.a / (1. - self.b):
            raise RuntimeError("Ensure path is simulated from t=0")
        else:
            obsData = []
            for i in range(T_horizon):
                obsData.append(self.__next_price())
                self.__update_log_vol()
            obsData.append(self.__next_price())
            return pd.DataFrame(data={'price': obsData})

    @staticmethod
    def plot_simulated(obsData, volData, title):
        T_horizon = len(obsData)
        time = np.linspace(0., T_horizon, T_horizon)
        fig, ax = plt.subplots(2, 1)
        plot(time, lines=[obsData], label_args=["Price Process"], xlabel="Time", ylabel="Price Process",
             title=title, fig=fig, ax=ax[0])
        plot(time, lines=[volData], label_args=["Log Volatility Process"], xlabel="Time", ylabel="Volatility Process",
             title=title, fig=fig, ax=ax[1])
        plt.show()
