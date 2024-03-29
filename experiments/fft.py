import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.fft import fftshift
from tftb.processing import WignerVilleDistribution

from src.classes.ClassFractionalBrownianNoise import FractionalBrownianNoise

plt.style.use('ggplot')
matplotlib.rcParams.update({
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
    'text.latex.preamble': r"\usepackage{amsmath}"
})


def estimate_and_plot_H(sample, N, H):
    ks = np.linspace(1, N, num=N, dtype=int)
    freqs = np.pi * ks / N
    Is = (np.abs(np.fft.fft(sample)) ** 2) / N  # fBn_spectral_density(hurst=H, N=2 * N + 1)
    fig, ax = plt.subplots()
    ax.scatter(np.log10(freqs), np.log10(Is), s=2, color="blue", label="Periodogram")
    fit = np.polyfit(np.log10(freqs)[:int(0.05 * N)], np.log10(Is)[:int(0.05 * N)], 1)
    print("Estimated H from exact sample:", 0.5 * (1. - fit[0]))
    ax.plot(np.unique(np.log10(freqs)),
            np.poly1d(fit)(np.unique(np.log10(freqs))),
            label="Straight Line Fit")
    plt.legend()
    plt.show()


def plot_periodogram(sample, N, Nfft):
    f, Pxx_den = scipy.signal.periodogram(sample, fs=1 / N, nfft=Nfft, return_onesided=False, scaling="spectrum")
    plt.scatter(f * N * 2 * np.pi, (Pxx_den))
    # plt.show()


def single_time_fft(t, lag, freq, H):
    return np.sum(
        0.5 * (np.power(np.abs(lag + 1), 2 * H) + np.power(np.abs(-1 + lag), 2 * H) - 2 * np.power(np.abs(lag),
                                                                                                   2 * H)) * np.exp(
            1j * lag * freq))


def avg_fft_uptoT(T, lag, freq, H):
    assert (T > 0)
    pow = 2. * H + 1.
    const = 1. / (2. * T * pow)
    return np.sum((const * (np.power(T, pow) + (
            np.power(np.maximum(T - lag, 0), pow) - np.power(np.maximum(0, lag) - lag, pow)) - (
                                    np.power(lag - np.minimum(T, lag), pow) - np.power(np.maximum(0, lag),
                                                                                       pow))) - 0.5 * np.power(
        np.abs(lag), pow - 1.)) * np.exp(1j * lag * freq))


def check():
    H = 0.7
    T = 1.
    pow = 2 * H + 1.
    TgridSp = 100000
    timegrid = np.linspace(0, T, num=TgridSp)
    lag_grid = np.arange(-1000, 1000 + 1, dtype=int)
    answs = np.zeros((lag_grid.shape[0], 2))
    for i in range(lag_grid.shape[0]):
        lag = lag_grid[i]
        answs[i, 0] = (1. / (2 * T)) * np.sum(np.power(np.abs(timegrid - lag), pow - 1) * (1 / TgridSp))
        answs[i, 1] = (1. / (2. * T * pow)) * ((
                                                       np.power(np.maximum(T - lag, 0), pow) - np.power(
                                                   np.maximum(0, lag) - lag, pow)) - (
                                                       np.power(lag - np.minimum(T, lag), pow) - np.power(
                                                   np.maximum(0, lag),
                                                   pow)))


def manual_timedep_avg_fft(N, H, T):
    freq_grid = np.linspace(-np.pi, np.pi, num=N)
    lag_grid = np.arange(-10000, 10000 + 1, dtype=int)
    psd = np.array([np.abs(avg_fft_uptoT(T=T, lag=lag_grid, freq=freq, H=H)) ** 2 for freq in freq_grid]) * (
            1. / (2. * np.pi * N))
    plt.semilogy(freq_grid, psd)
    plt.title("Average Spectral Density up to Time {}".format(T))
    plt.plot()
    plt.show()
    plt.close()


def manual_timedep_fft(N, H):
    time_grid = np.linspace(0., 1., num=N)
    freq_grid = np.linspace(-np.pi, np.pi, num=N)
    lag_grid = np.arange(-10000, 10000 + 1, dtype=int)
    for _ in range(0):
        t = time_grid[np.random.randint(low=0, high=N)]
        psd = np.array([np.abs(single_time_fft(t=t, lag=lag_grid, freq=freq, H=H)) ** 2 for freq in freq_grid]) * (
                1. / (2. * np.pi * N))
        plt.semilogy(freq_grid, psd)
        plt.title("Time {}".format(t))
        plt.plot()
        plt.show()
        plt.close()
    psd = np.array([np.abs(single_time_fft(t=100000, lag=lag_grid, freq=freq, H=H)) ** 2 for freq in freq_grid]) * (
            1. / (2. * np.pi * N))
    plt.semilogy(freq_grid, psd)
    plt.title("Time {}".format(1))
    plt.plot()
    plt.show()
    plt.close()


def manual_consecutive_ffts(N, H):
    freq_grid = np.linspace(-np.pi, np.pi, num=N)
    lag_grid = np.arange(-1000, 1000 + 1, dtype=int)
    t1 = (N - 1) / N
    t2 = 1.
    fft1 = np.array([single_time_fft(t=t1, lag=lag_grid, freq=freq, H=H) for freq in freq_grid])
    fft2 = np.array([single_time_fft(t=t2, lag=lag_grid, freq=freq, H=H) for freq in freq_grid])
    plt.scatter(freq_grid, np.abs(fft1))
    plt.plot()
    plt.show()
    plt.scatter(freq_grid, np.abs(fft2))
    plt.plot()
    plt.show()
    plt.scatter(freq_grid, np.abs(fft2) - np.abs(fft1))
    plt.plot()
    plt.show()
    plt.scatter(freq_grid, np.abs(fft2 - fft1))
    plt.plot()
    plt.show()


def plot_spectrogram(sample, N):
    nperseg = 30
    frequencies, times, Sxx = scipy.signal.spectrogram(sample, fs=1. / N, nperseg=nperseg, noverlap=nperseg // 2,
                                                       return_onesided=False, scaling="spectrum")
    plt.pcolormesh(times, fftshift(frequencies * 2 * np.pi * N), fftshift(Sxx, axes=0), shading='gouraud')
    plt.ylabel('Frequency')
    plt.xlabel('Time')
    plt.colorbar(label='Power')
    plt.show()
    plt.close()


def plot_WVS(sample, N):
    ts = np.linspace(0, 1., num=N)
    dt = 1. / N
    Sxx, _, freqs = WignerVilleDistribution(sample, timestamps=ts).run()
    f_wvd = np.fft.fftshift(np.fft.fftfreq(Sxx.shape[0], d=2 * (dt)))
    df_wvd = f_wvd[1] - f_wvd[0]  # the frequency step in the WVT
    im = plt.imshow(np.fft.fftshift(Sxx, axes=0), aspect='auto', origin='lower',
                    extent=(ts[0] - dt / 2, ts[-1] + dt / 2,
                            f_wvd[0] - df_wvd / 2, f_wvd[-1] + df_wvd / 2))
    plt.xlabel('time [s]')
    plt.ylabel('frequency [Hz]')
    plt.colorbar(im)
    plt.title('Wigner-Ville Transform')
    plt.show()


if __name__ == "__main__":
    H = 0.7
    N = 2 ** 10
    mkv_blnk = 30
    fbn = FractionalBrownianNoise(H=H, rng=np.random.default_rng())
    sample = fbn.circulant_simulation(N_samples=N).cumsum()
    plot_spectrogram(sample=sample, N=N)
    plot_WVS(sample=sample, N=N)

    """
    plot_periodogram(sample, N=N, Nfft=N)
    plt.plot()
    plt.show()
    plot_periodogram(sample[-mkv_blnk:], N=N, Nfft=N)
    plt.plot()
    plt.show()
    plot_periodogram(sample[-mkv_blnk:], N=N, Nfft=mkv_blnk)
    plt.plot()
    plt.show()
    plot_periodogram(sample[:mkv_blnk], N=N, Nfft=N)
    plt.plot()
    plt.show()
    #    manual_timedep_fft(N=N, H=H)
    # manual_timedep_avg_fft(N=N, H=H, T=10000000)
    # plot_periodogram(sample + np.random.normal(loc=0, scale=1, size=N), N=N, H=H)
    # plot_periodogram(np.random.normal(loc=0, scale=1, size=N), N=N, H=H)
    """
