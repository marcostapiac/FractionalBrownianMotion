import numbers
from types import NoneType
from typing import Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import invgamma as sinvgamma, kstest
from scipy.stats import norm as snorm
from scipy.stats import truncnorm
from sklearn.manifold import TSNE

from utils.math_functions import acf


def plot_subplots(time_ax, lines, label_args, xlabels, ylabels, title, fig=None, ax=None, saveTransparent=False):
    """ Plotting function ot plot multiple traces in same figure but different axis"""
    try:
        assert (len(lines) == len(xlabels) and len(lines) == len(ylabels))
    except AssertionError:
        return RuntimeError("Please add as many x-y axis labels as lines to plot")
    plt.style.use('ggplot')
    matplotlib.rcParams.update({
        'font.family': 'serif',
        'text.usetex': True,
        'pgf.rcfonts': False,
    })
    if (fig and ax) is None:
        L = len(lines)
        fig, ax = plt.subplots(L, 1)
    for i in range(len(lines)):
        ax[i].plot(time_ax, lines[i], label=label_args[i], lw=1.1, marker='.', markersize=1)
        ax[i].set_xlabel(xlabels[i])
        ax[i].set_ylabel(ylabels[i])
        if saveTransparent:
            ax[i].xaxis.label.set_color('white')  # setting up X-axis label color to yellow
            ax[i].yaxis.label.set_color('white')
            ax[i].tick_params(axis='x', colors='white')  # setting up X-axis tick color to red
            ax[i].tick_params(axis='y', colors='white')
            ax[i].grid(visible=True)
            ax[i].legend()
    if saveTransparent:
        fig.patch.set_alpha(0.0)
        fig.suptitle(title, color="white")
    else:
        fig.suptitle(title)
    plt.tight_layout()


def plot(time_ax, lines, label_args, xlabel, ylabel, title, fig=None, ax=None, saveTransparent=False):
    """ Plotting function to plot multiple traces in same figure and axis object"""
    plt.style.use('ggplot')
    matplotlib.rcParams.update({
        'font.family': 'serif',
        'text.usetex': True,
        'pgf.rcfonts': False,
    })
    if (fig and ax) is None:
        fig, ax = plt.subplots()
    for i in range(len(lines)):
        ax.step(time_ax, lines[i], label=label_args[i], lw=1.1, marker='.')
    if saveTransparent:
        fig.patch.set_alpha(0.0)
        ax.xaxis.label.set_color('white')  # setting up X-axis label color to yellow
        ax.yaxis.label.set_color('white')
        ax.tick_params(axis='x', colors='white')  # setting up X-axis tick color to red
        ax.tick_params(axis='y', colors='white')
        ax.set_title(title, color="white")
    else:
        ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    plt.tight_layout()


def plot_fBm_process(time_ax, paths, label_args, xlabel=None, ylabel=None, title="Sample Paths",
                     fig=None,
                     ax=None, saveTransparent=False):
    plt.style.use('ggplot')
    if (fig and ax) is None:
        fig, ax = plt.subplots()
    for i in range(len(paths)):
        ax.step(time_ax, paths[i], label="$H = " + str(round(label_args[i], 3)) + "$", lw=1.2)
    if (xlabel and ylabel) is None:
        ax.set_xlabel("Time")
        ax.set_ylabel("Position")
    else:
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
    if saveTransparent:
        fig.patch.set_alpha(0.0)
        ax.xaxis.label.set_color('white')  # setting up X-axis label color to yellow
        ax.yaxis.label.set_color('white')
        ax.tick_params(axis='x', colors='white')  # setting up X-axis tick color to red
        ax.tick_params(axis='y', colors='white')
        ax.set_title(title, color="white")
    else:
        ax.set_title(title)
    ax.legend()


def qqplot(x, y, xlabel="", ylabel="", plottitle="", quantiles=None, interpolation='nearest', ax=None, rug=False,
           rug_length=0.05, rug_kwargs=None, font_size=14, title_size=14, log=True, **kwargs):
    """Draw a quantile-quantile plot for `x` versus `y`.

    Parameters
    ----------
    x, y : array-like
        One-dimensional numeric arrays.

    ax : matplotlib.axes.Axes, optional
        Axes on which to plot. If not provided, the current axes will be used.

    quantiles : int or array-like, optional
        Quantiles to include in the plot. This can be an array of quantiles, in
        which case only the specified quantiles of `x` and `y` will be plotted.
        If this is an int `n`, then the quantiles will be `n` evenly spaced
        points between 0 and 1. If this is None, then `min(len(x), len(y))`
        evenly spaced quantiles between 0 and 1 will be computed.

    interpolation : {‘linear’, ‘lower’, ‘higher’, ‘midpoint’, ‘nearest’}
        Specify the interpolation method used to find quantiles when `quantiles`
        is an int or None. See the documentation for numpy.quantile().

    rug : bool, optional
        If True, draw a rug plot representing both samples on the horizontal and
        vertical axes. If False, no rug plot is drawn.

    rug_length : float in [0, 1], optional
        Specifies the length of the rug plot lines as a fraction of the total
        vertical or horizontal length.

    rug_kwargs : dict of keyword arguments
        Keyword arguments to pass to matplotlib.axes.Axes.axvline() and
        matplotlib.axes.Axes.axhline() when drawing rug plots.

    kwargs : dict of keyword arguments
        Keyword arguments to pass to matplotlib.axes.Axes.scatter() when drawing
        the q-q plot.
    """
    plt.style.use('ggplot')
    plt.rc('font', size=font_size)
    plt.rc('axes', titlesize=title_size)
    x1 = x
    y1 = y
    if ax is None:
        ax = plt.gca()
    if quantiles is None:
        quantiles = min(len(x1), len(y1))

    # Compute quantiles of the two samples
    if isinstance(quantiles, numbers.Integral):
        quantiles = np.linspace(start=0, stop=1, num=int(quantiles))
    else:
        quantiles = np.atleast_1d(np.sort(quantiles))
    x_quantiles1 = np.quantile(x1, quantiles, interpolation=interpolation)
    y_quantiles1 = np.quantile(y1, quantiles, interpolation=interpolation)
    # Draw the rug plots if requested
    if rug:
        # Default rug plot settings
        rug_x_params = dict(ymin=0, ymax=rug_length, c='gray', alpha=0.5)
        rug_y_params = dict(xmin=0, xmax=rug_length, c='gray', alpha=0.5)

        # Override default setting by any user-specified settings
        if rug_kwargs is not None:
            rug_x_params.update(rug_kwargs)
            rug_y_params.update(rug_kwargs)

        # Draw the rug plots
        for point in x1:
            ax.axvline(point, **rug_x_params)
        for point in y1:
            ax.axhline(point, **rug_y_params)

    # Draw the q-q plot and compare with y = x
    ax.scatter(x_quantiles1, y_quantiles1, c="black", label="Q-Q plot", **kwargs)
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
    ]
    # now plot both limits against each other
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(plottitle)
    ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0, label="Line of Equality")
    if log:
        ax.set_xscale('log')
        ax.set_yscale('log')
    ax.legend()


def histogramplot(rvs, pdf_vals=None, axis=None, num_bins=100, xlabel="", ylabel="", plottitle="", plottlabel="",
                  fig=None, ax=None):
    plt.style.use('ggplot')
    if (fig and ax) is None:
        fig, ax = plt.subplots()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(plottitle)
    binvals, _, _ = plt.hist(rvs, num_bins, density=True, label="Histogram")
    if pdf_vals is not None:
        ax.plot(axis, pdf_vals, label=plottlabel, color="red")
    plt.legend()
    return fig, ax, binvals


def gibbs_histogram_plot(Thetas, burnOut, plottitle, trueVals, priorParams):
    muUPriorParams, alphaPriorParams, muXPriorParams, sigmaXPriorParams = priorParams

    fig, ax, binVals = histogramplot(Thetas[burnOut:, 0], xlabel="Observation Mean", ylabel="PDF",
                                     plottitle=plottitle)
    ax.axvline(trueVals[0], label="True Parameter Value $ " + str(round(trueVals[0], 3)) + " $", color="blue")
    axis = np.linspace(snorm.ppf(0.001, loc=muUPriorParams[0], scale=muUPriorParams[1]),
                       snorm.ppf(0.999, loc=muUPriorParams[0], scale=muUPriorParams[1]), num=1000)
    pdfVals = snorm.pdf(axis, loc=muUPriorParams[0], scale=muUPriorParams[1])
    ax.plot(axis, pdfVals * (np.max(binVals) / np.max(pdfVals)), label="Scaled Prior Distribution", color="orange")
    plt.legend()

    fig, ax, binVals = histogramplot(Thetas[burnOut:, 1], xlabel="Volatility Standardised Mean Reversion", ylabel="PDF",
                                     plottitle=plottitle)
    ax.axvline(trueVals[1], label="True Parameter Value $ " + str(round(trueVals[1], 3)) + " $", color="blue")
    mean, sigma = alphaPriorParams[0], alphaPriorParams[1]
    axis = np.linspace(truncnorm.ppf(q=0.001, a=-mean / sigma, b=np.inf, loc=mean, scale=sigma),
                       truncnorm.ppf(q=0.999, a=-mean / sigma, b=np.inf, loc=mean, scale=sigma), num=1000)
    pdfVals = truncnorm.pdf(axis, a=-mean / sigma, b=np.inf, loc=mean, scale=sigma)
    ax.plot(axis, pdfVals * (np.max(binVals) / np.max(pdfVals)), label="Scaled Prior Distribution",
            color="orange")
    plt.legend()

    fig, ax, binVals = histogramplot(Thetas[burnOut:, 2], xlabel="Volatility Mean", ylabel="PDF",
                                     plottitle=plottitle)
    ax.axvline(trueVals[2], label="True Parameter Value $ " + str(round(trueVals[2], 3)) + " $", color="blue")
    mean, sigma = muXPriorParams[0], muXPriorParams[1]
    axis = np.linspace(truncnorm.ppf(q=0.001, a=-mean / sigma, b=np.inf, loc=mean, scale=sigma),
                       truncnorm.ppf(q=0.999, a=-mean / sigma, b=np.inf, loc=mean, scale=sigma), num=1000)
    pdfVals = truncnorm.pdf(axis, a=-mean / sigma, b=np.inf, loc=mean, scale=sigma)
    ax.plot(axis, pdfVals * np.max(binVals) / np.max(pdfVals), label="Scaled Prior Distribution",
            color="orange")
    plt.legend()

    fig, ax, binVals = histogramplot(Thetas[burnOut:, 3], xlabel="Volatility Variance", ylabel="PDF",
                                     plottitle=plottitle)
    ax.axvline(trueVals[3], label="True Parameter Value $ " + str(round(trueVals[3], 3)) + " $", color="blue")
    alpha0, beta0 = sigmaXPriorParams
    axis = np.linspace(sinvgamma.ppf(0.35, a=alpha0, loc=0., scale=beta0),
                       sinvgamma.ppf(0.65, a=alpha0, loc=0., scale=beta0), num=1000)
    pdfVals = sinvgamma.pdf(axis, a=alpha0, scale=beta0)
    ax.plot(axis, pdfVals * (np.max(binVals) / np.max(pdfVals)), label="Scaled Prior Distribution",
            color="orange")
    plt.legend()


def boxplot(data, xlabel="", ylabel="", plottitle="", dataLabels="", fig=None, ax=None):
    plt.style.use('ggplot')
    if (fig and ax) is None:
        fig, ax = plt.subplots()
    ax.boxplot(data, labels=dataLabels)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(plottitle)
    plt.legend()


def plot_parameter_traces(S, Thetas):
    plot(np.arange(0, S + 1, step=1), [Thetas[:, 0]], ["Observation Mean"], "Algorithm Iteration", "Observation Mean",
         "Metropolis-within-Gibbs Sampler")
    plot(np.arange(0, S + 1, step=1), [Thetas[:, 1]], ["Volatility Standardised Mean Reversion"], "Algorithm Iteration",
         "Volatility Standardised Mean Reversion",
         "Metropolis-within-Gibbs Sampler")
    plot(np.arange(0, S + 1, step=1), [Thetas[:, 2]], ["Volatility Mean"], "Algorithm Iteration", "Volatility Mean",
         "Metropolis-within-Gibbs Sampler")
    plot(np.arange(0, S + 1, step=1), [Thetas[:, 3]], ["Volatility Variance Parameter"], "Algorithm Iteration",
         "Volatility Variance Parameter",
         "Metropolis-within-Gibbs Sampler")


def plot_autocorrfns(Thetas):
    acfObsMean = acf(Thetas[:, 0])
    acfAlpha = acf(Thetas[:, 1])
    acfVolMean = acf(Thetas[:, 2])
    acfVolStd = acf(Thetas[:, 3])
    S = acfVolMean.shape[0]
    plot(np.arange(0, S, step=1), [acfObsMean], ["Observation Mean"], "Lag", "Observation Mean",
         "Autocorrelation Function")
    plot(np.arange(0, S, step=1), [acfAlpha], ["Standardised Volatility Mean Reversion"], "Lag",
         "Standardised Volatility Mean Reversion",
         "Autocorrelation Function")
    plot(np.arange(0, S, step=1), [acfVolMean], ["Volatility Mean"], "Lag", "Volatility Mean",
         "Autocorrelation Function")
    plot(np.arange(0, S, step=1), [acfVolStd], ["Volatility Variance Parameter"], "Lag",
         "Volatility Variance Parameter",
         "Autocorrelation Function")


def plot_loss_epochs(epochs: np.ndarray, train_loss: np.ndarray, val_loss: Union[NoneType, np.array] = None,
                     toSave: bool = False, saveName: Union[NoneType, str] = None) -> None:
    plt.style.use('ggplot')
    matplotlib.rcParams.update({
        'font.family': 'serif',
        'text.usetex': True,
        'pgf.rcfonts': False,
    })
    plt.plot(epochs, train_loss, color="b", label="Train MSE")
    if val_loss is not None: plt.plot(epochs, val_loss, label="Validation MSE")
    plt.xlabel("Epochs")
    plt.ylabel("Loss Function")
    plt.title("Learning Curves")
    plt.legend()
    plt.tight_layout()
    if toSave:
        if saveName is None:
            raise ValueError("If you want to save the image, provide a file name")
        plt.savefig("../pngs/" + str(saveName) + ".png", bbox_inches="tight", transparent=False)
    plt.show()


def plot_tSNE(x: np.ndarray, y: np.ndarray, labels: list[str]) -> None:
    assert (len(labels) == 2)
    x_embed = TSNE().fit_transform(x)
    y_embed = TSNE().fit_transform(y)
    plt.scatter(x_embed[:, 0], x_embed[:, 1], label=labels[0])
    plt.scatter(y_embed[:, 0], y_embed[:, 1], label=labels[1])
    plt.title("t-SNE Plot")
    plt.xlabel("Embedding Dim 1")
    plt.ylabel("Embedding Dim 2")
    plt.tight_layout()
    plt.legend()
    plt.show()


def plot_diffusion_marginals(forward_samples: np.ndarray, reverse_samples: np.ndarray, timeDim: int,
                             diffTime: int) -> None:
    for t in np.arange(start=0, stop=timeDim, step=1):
        forward_t = forward_samples[:, t].flatten()
        reverse_samples_t = reverse_samples[:, t].flatten()
        qqplot(x=forward_t,
               y=reverse_samples_t, xlabel="Fwd Samples at Diff Time {}".format(diffTime),
               ylabel="Reverse Samples at Diff Time {}".format(diffTime),
               plottitle="Marginal Q-Q Plot at Time Dim {}".format(t + 1), log=False)
        print(kstest(forward_t, reverse_samples_t))
        plt.show()
        plt.close()
