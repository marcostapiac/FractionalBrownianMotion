import glob
import multiprocessing as mp
import numbers
import os
from functools import partial
from typing import Union, Optional, Tuple, Mapping

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from PIL import Image
from ml_collections import ConfigDict
from scipy import stats
from scipy.stats import invgamma as sinvgamma
from scipy.stats import norm as snorm
from scipy.stats import truncnorm
from sklearn.manifold import TSNE

from configs import project_config
from configs.project_config import NoneType
from src.classes.ClassFractionalBrownianNoise import FractionalBrownianNoise
from utils.math_functions import acf, reduce_to_fBn, optimise_whittle

plt.style.use('ggplot')
matplotlib.rcParams.update({
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
    'text.latex.preamble': r"\usepackage{amsmath}"
})


def plot_efftimes(lin_ts: np.ndarray, eff_ts: np.ndarray) -> None:
    """
    Function to visualise effective diffusion times
        :param lin_ts: Linear time series
        :param eff_ts: Effective Diffusion Times
        :return: None
    """
    plt.scatter(lin_ts, eff_ts)
    plt.xlabel("Linear Time")
    plt.ylabel("Effective Diffusion Time")
    plt.show()


def plot_subplots(time_ax: np.ndarray, data: np.ndarray, label_args: np.ndarray[Union[NoneType, str]],
                  xlabels: np.ndarray[Union[NoneType, str]], ylabels: np.ndarray[Union[NoneType, str]],
                  globalTitle: str, fig: Union[NoneType, matplotlib.figure.Figure] = None,
                  ax: Union[NoneType, np.ndarray[matplotlib.axes.Axes]] = None, saveTransparent: bool = False):
    """
    Plotting function ot plot multiple traces in same figure but DIFFERENT axis
    :param time_ax: MCMC timeline
    :param data: Data containing MCMC trace values
    :param label_args: Labels for each plot
    :param xlabels: X-axis labels
    :param ylabels: Y-axis labels
    :param globalTitle: Global title
    :param fig: Figure object
    :param ax: Axis object or Array of
    :param saveTransparent: Indicates whether to remove background from figure
    :return: None
    """
    try:
        assert (len(data) == len(xlabels) and len(data) == len(ylabels))
    except AssertionError:
        return RuntimeError("Please add as many x-y axis labels as lines to plot")
    if (fig and ax) is None:
        L = len(data)
        fig, ax = plt.subplots(L, 1)
    for i in range(len(data)):
        ax[i].plot(time_ax, data[i], label=label_args[i], lw=1.1, marker='.', markersize=1)
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
        fig.suptitle(globalTitle, color="white")
    else:
        fig.suptitle(globalTitle)
    plt.tight_layout()
    plt.show()


def plot(time_ax, data, label_args: np.ndarray[str], xlabel: str, ylabel: str, title: str,
         fig: matplotlib.figure.Figure = None, ax: matplotlib.axes.Axes = None, saveTransparent: bool = False):
    """
    Plotting function to plot multiple traces in SAME figure and axis object
        :param time_ax: MCMC timeline
        :param data: Data containing MCMC trace values
        :param label_args: Labels for each plot
        :param xlabel: X-axis label
        :param ylabel: Y-axis label
        :param title: Plot title
        :param fig: Figure object
        :param ax: Axis object
        :param saveTransparent: Indicates whether to remove background from figure
        :return: None

    """
    if (fig and ax) is None:
        fig, ax = plt.subplots()
    for i in range(len(data)):
        ax.step(time_ax, data[i], label=label_args[i], lw=0.01, marker='.')
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


def qqplot(x: np.ndarray, y: np.ndarray, xlabel: str = "", ylabel: str = "", plottitle: str = "",
           quantiles: Union[NoneType, int, np.ndarray] = None, interpolation: str = 'nearest',
           ax: matplotlib.axes.Axes = None, rug: bool = False,
           rug_length: float = 0.05, rug_kwargs: Union[NoneType, Mapping] = None, font_size: int = 14,
           title_size: int = 14, log: bool = True, **kwargs) -> None:
    """Draw a quantile-quantile plot for `x` versus `y`.

    Parameters
    ----------
    x, y : array-like
        One-dimensional numeric arrays.

    xlabel: string
        X-axis label

    ylabel: string
        Y-axis label

    plottitle: str
        Title for entire plot

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
    font_size: int
        Size of font in plot
    title_size: int
        Size of title font in plot
    log: bool
        Flag indicating whether to use log-axis
    rug_kwargs : dict of keyword arguments
        Keyword arguments to pass to matplotlib.axes.Axes.axvline() and
        matplotlib.axes.Axes.axhline() when drawing rug plots.

    kwargs : dict of keyword arguments
        Keyword arguments to pass to matplotlib.axes.Axes.scatter() when drawing
        the q-q plot.
    """
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


def plot_histogram(rvs: np.ndarray, pdf_vals: Union[NoneType, np.ndarray] = None,
                   xlinspace: Union[NoneType, np.ndarray] = None, num_bins: int = 100, xlabel: str = "",
                   ylabel: str = "", plottitle: str = "", plotlabel: str = "",
                   fig: Union[NoneType, matplotlib.figure.Figure] = None,
                   ax: Union[NoneType, matplotlib.axes.Axes] = None) -> Tuple[
    matplotlib.figure.Figure, matplotlib.axes.Axes, np.ndarray]:
    if (fig and ax) is None:
        fig, ax = plt.subplots()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(plottitle)
    binvals, _, _ = plt.hist(rvs, num_bins, density=True, label="Histogram")
    if pdf_vals is not None:
        assert (xlinspace is not None)
        ax.plot(xlinspace, pdf_vals, label=plotlabel, color="blue")
    plt.legend()
    return fig, ax, binvals


def gibbs_histogram_plot(thetas: np.ndarray, burnOut: int, titlePlot: str, trueVals: np.ndarray,
                         priorParams: np.ndarray) -> None:
    muUPriorParams, alphaPriorParams, muXPriorParams, sigmaXPriorParams = priorParams

    fig, ax, binVals = plot_histogram(thetas[burnOut:, 0], xlabel="Observation Mean", ylabel="PDF", plottitle=titlePlot)
    ax.axvline(trueVals[0], label="True Parameter Value $ " + str(round(trueVals[0], 3)) + " $", color="blue")
    axis = np.linspace(snorm.ppf(0.001, loc=muUPriorParams[0], scale=muUPriorParams[1]),
                       snorm.ppf(0.999, loc=muUPriorParams[0], scale=muUPriorParams[1]), num=1000)
    pdfVals = snorm.pdf(axis, loc=muUPriorParams[0], scale=muUPriorParams[1])
    ax.plot(axis, pdfVals * (np.max(binVals) / np.max(pdfVals)), label="Scaled Prior Distribution", color="orange")
    plt.legend()

    fig, ax, binVals = plot_histogram(thetas[burnOut:, 1], xlabel="Volatility Standardised Mean Reversion",
                                      ylabel="PDF", plottitle=titlePlot)
    ax.axvline(trueVals[1], label="True Parameter Value $ " + str(round(trueVals[1], 3)) + " $", color="blue")
    mean, sigma = alphaPriorParams[0], alphaPriorParams[1]
    axis = np.linspace(truncnorm.ppf(q=0.001, a=-mean / sigma, b=np.inf, loc=mean, scale=sigma),
                       truncnorm.ppf(q=0.999, a=-mean / sigma, b=np.inf, loc=mean, scale=sigma), num=1000)
    pdfVals = truncnorm.pdf(axis, a=-mean / sigma, b=np.inf, loc=mean, scale=sigma)
    ax.plot(axis, pdfVals * (np.max(binVals) / np.max(pdfVals)), label="Scaled Prior Distribution",
            color="orange")
    plt.legend()

    fig, ax, binVals = plot_histogram(thetas[burnOut:, 2], xlabel="Volatility Mean", ylabel="PDF", plottitle=titlePlot)
    ax.axvline(trueVals[2], label="True Parameter Value $ " + str(round(trueVals[2], 3)) + " $", color="blue")
    mean, sigma = muXPriorParams[0], muXPriorParams[1]
    axis = np.linspace(truncnorm.ppf(q=0.001, a=-mean / sigma, b=np.inf, loc=mean, scale=sigma),
                       truncnorm.ppf(q=0.999, a=-mean / sigma, b=np.inf, loc=mean, scale=sigma), num=1000)
    pdfVals = truncnorm.pdf(axis, a=-mean / sigma, b=np.inf, loc=mean, scale=sigma)
    ax.plot(axis, pdfVals * np.max(binVals) / np.max(pdfVals), label="Scaled Prior Distribution",
            color="orange")
    plt.legend()

    fig, ax, binVals = plot_histogram(thetas[burnOut:, 3], xlabel="Volatility Variance", ylabel="PDF",
                                      plottitle=titlePlot)
    ax.axvline(trueVals[3], label="True Parameter Value $ " + str(round(trueVals[3], 3)) + " $", color="blue")
    alpha0, beta0 = sigmaXPriorParams
    axis = np.linspace(sinvgamma.ppf(0.35, a=alpha0, loc=0., scale=beta0),
                       sinvgamma.ppf(0.65, a=alpha0, loc=0., scale=beta0), num=1000)
    pdfVals = sinvgamma.pdf(axis, a=alpha0, scale=beta0)
    ax.plot(axis, pdfVals * (np.max(binVals) / np.max(pdfVals)), label="Scaled Prior Distribution",
            color="orange")
    plt.legend()


def plot_and_save_boxplot(data: np.ndarray, dataLabels: list, xlabel: str = "", ylabel: str = "", title_plot: str = "",
                          toSave: bool = False, saveName: str = "",
                          fig: Union[NoneType, matplotlib.figure.Figure] = None,
                          ax: Union[NoneType, matplotlib.axes.Axes] = None) -> None:
    """
    Plot boxplot of data
    :param data: Data
    :param xlabel: X-axis label
    :param ylabel: Y-axis label
    :param title_plot: Title for plot
    :param dataLabels: Legends for EACH boxplot
    :param fig: Figure object
    :param ax: Axis object
    :param toSave: Indicates whether to save figure or not
    :param saveName: Filename for saved figure
    :return: None
    """
    assert (dataLabels == None or (len(data.shape) == 1 and len(dataLabels) == 1) or data.shape[1] == len(dataLabels))
    if (fig and ax) is None:
        fig, ax = plt.subplots()
    fg = ax.boxplot(data)
    if None not in dataLabels: ax.legend([fg["boxes"][i] for i in range(len(fg["boxes"]))], dataLabels,
                                         loc="upper right")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title_plot)
    plt.show()
    if toSave: plt.savefig(saveName, bbox_inches="tight")
    plt.show()
    plt.close()


def plot_parameter_traces(S: int, thetas: np.ndarray) -> None:
    """
    Helper function to plot MCMC evolution of parameters
    :param S: Number of MCMC steps
    :param thetas: Parameters
    :return: None
    """
    plot(np.arange(0, S + 1, step=1), thetas[:, 0], np.array(["Observation Mean"]), "Algorithm Iteration",
         "Observation Mean",
         "Metropolis-within-Gibbs Sampler")
    plot(np.arange(0, S + 1, step=1), thetas[:, 1], np.array(["Volatility Standardised Mean Reversion"]),
         "Algorithm Iteration",
         "Volatility Standardised Mean Reversion",
         "Metropolis-within-Gibbs Sampler")
    plot(np.arange(0, S + 1, step=1), thetas[:, 2], np.array(["Volatility Mean"]), "Algorithm Iteration",
         "Volatility Mean",
         "Metropolis-within-Gibbs Sampler")
    plot(np.arange(0, S + 1, step=1), thetas[:, 3], np.array(["Volatility Variance Parameter"]), "Algorithm Iteration",
         "Volatility Variance Parameter",
         "Metropolis-within-Gibbs Sampler")


def plot_autocorrfns(thetas: np.ndarray) -> None:
    """
    Helper function for autocorrelations
    :param thetas: Parameter values
    :return: None
    """

    acfObsMean = acf(thetas[:, 0])
    acfAlpha = acf(thetas[:, 1])
    acfVolMean = acf(thetas[:, 2])
    acfVolStd = acf(thetas[:, 3])
    S = acfVolMean.shape[0]
    plot(np.arange(0, S, step=1), acfObsMean, np.array(["Observation Mean"]), "Lag", "Observation Mean",
         "Autocorrelation Function")
    plot(np.arange(0, S, step=1), acfAlpha, np.array(["Standardised Volatility Mean Reversion"]), "Lag",
         "Standardised Volatility Mean Reversion",
         "Autocorrelation Function")
    plot(np.arange(0, S, step=1), acfVolMean, np.array(["Volatility Mean"]), "Lag", "Volatility Mean",
         "Autocorrelation Function")
    plot(np.arange(0, S, step=1), acfVolStd, np.array(["Volatility Variance Parameter"]), "Lag",
         "Volatility Variance Parameter",
         "Autocorrelation Function")


def plot_and_save_loss_epochs(epochs: np.ndarray, train_loss: np.ndarray, val_loss: Union[NoneType, np.array] = None,
                              toSave: bool = False, saveName: Union[NoneType, str] = None) -> None:
    """
    Helper function to generate loss curves
    :param epochs: Epoch timeline
    :param train_loss: Training loss for each epoch
    :param val_loss: Validation loss for each epoch
    :param toSave: Indicates if we save figure
    :param saveName: Filename for saved figure
    :return: None
    """
    assert ((toSave and saveName is not None) or (not toSave and saveName is None))
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


def plot_tSNE(x: np.ndarray, labels: list[str], image_path: str, y: Union[NoneType, np.ndarray] = None) -> None:
    """
    Helper function to generate t-SNE plots
    :param x: Data
    :param labels: Labels for plot legend
    :param image_path: Path to save image
    :param y: Optional paramter, data to overlay ontop of previous plot
    :return: None
    """
    assert (len(labels) == 1 or len(labels) == 2)
    try:
        assert (x.shape[0] > 30)
    except AssertionError:
        print("Number of samples must be greater than perplexity value {} in TSNE embedder".format(int(30)))
    x_embed = TSNE().fit_transform(x)
    plt.scatter(x_embed[:, 0], x_embed[:, 1], label=labels[0])
    if y is not None:
        y_embed = TSNE().fit_transform(y)
        plt.scatter(y_embed[:, 0], y_embed[:, 1], label=labels[1])
    plt.title("t-SNE Plot")
    plt.xlabel("$\\textbf{Embedding Dim 1}$")
    plt.ylabel("$\\textbf{Embedding Dim 2}$")
    plt.tight_layout()
    plt.legend()
    plt.savefig(image_path)
    plt.show()


def plot_final_diff_marginals(forward_samples: np.ndarray, reverse_samples: np.ndarray, print_marginals: bool,
                              timeDim: int,
                              image_path: str) -> None:
    """
    Q-Q plot of multidimensional samples
        :param forward_samples: Forward diffsion samples
        :param reverse_samples: Reverse-time diffusion samples
        :param print_marginals: Flag indicating whether to plot and save QQ plots
        :param timeDim: Dimension of each sample
        :param image_path: Path to save image
        :return: None
    """
    for t in np.arange(start=0, stop=timeDim, step=1):
        forward_t = forward_samples[:, t].flatten()
        reverse_samples_t = reverse_samples[:, t].flatten()
        if print_marginals:
            qqplot(x=forward_t,
                   y=reverse_samples_t, xlabel="$\\textbf{Original Data Samples {}}$",
                   ylabel="$\\textbf{Final Reverse Diffusion Samples}$",
                   plottitle="Marginal Q-Q Plot at Time Dim {}".format(t + 1), log=False)
            plt.savefig(image_path + f"_QQ_timeDim{int(t)}")
            plt.show()
            plt.close()


def plot_heatmap(map: np.ndarray, annot: bool, title: str, filepath: str) -> None:
    """
    Helper function to create a heatmap
        :param map: Data to plot
        :param annot: Indicates whether to annotate error on diagram
        :param title: Title for diagram
        :param filepath: Path to save image
        :return: None
    """
    sns.heatmap(map, annot=annot, annot_kws={'size': 15})
    plt.title(title)
    plt.savefig(filepath)
    plt.show()


def plot_diffCov_heatmap(true_cov: np.ndarray, gen_cov: np.ndarray, image_path: str, annot: bool = True,
                         title: str = "") -> None:
    """
    Compute and plot difference between expected and sample covariance matrices
        :param true_cov: Theoretical covariance matrix
        :param gen_cov: Covariance matrix from samples of reverse-time diffusion
        :param image_path: Path to save image
        :param annot: Indicates whether to annotate error on diagram
        :param title: Title of plot
        :return: None
    """
    s = 100 * (gen_cov - true_cov) / true_cov
    print("Average absolute percentage error: ", np.mean(np.abs(s)))
    if title == "": title = "Abs Percentage Error in Covariance Matrices"
    plot_heatmap(map=np.abs(s), annot=annot, title=title, filepath=image_path)


def plot_dataset(forward_samples: np.ndarray, reverse_samples: np.ndarray, image_path: str,
                 labels: Optional[Union[list[str], NoneType]] = None) -> None:
    """
    Scatter plot of 2 dimensional data (in the context of diffusion models)
        :param forward_samples: Original data
        :param reverse_samples: Final reverse-time diffusion samples
        :param image_path: Path to save image
        :param labels: Labels for each plot
        :return: None
    """
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    labels = ["Original Data", "Generated Samples"] if labels is None else labels
    ax.scatter(forward_samples[:, 0], forward_samples[:, 1], alpha=0.6, label=labels[0])
    n = 10000
    ax.scatter(reverse_samples[:n, 0], reverse_samples[:n, 1], alpha=0.3, label=labels[1])
    ax.grid(False)
    ax.set_aspect('equal', adjustable='box')
    strtitle = "Scatter Plot of Final Reverse Diffusion Samples"
    ax.set_title(strtitle)
    ax.set_xlabel("$\\textbf{Time Dim 1}$")
    ax.set_ylabel("$\\textbf{Time Dim 2}$")
    plt.legend()
    plt.savefig(image_path)
    plt.show()


def plot_eigenvalues(expec_cov: np.ndarray, generated_cov: np.ndarray, labels: list[str]) -> None:
    """
    Plot eigenvalues of expected and sample covariance matrix
    :param expec_cov: Theoretical covariance matrix
    :param generated_cov: Covariance matrix for reverse-time diffusion samples
    :param labels: Labels for each plot
    :return: None
    """
    assert (expec_cov.shape == generated_cov.shape and len(labels) == 2)
    eigs1 = np.sort(np.linalg.eigvals(expec_cov))
    eigs2 = np.sort(np.linalg.eigvals(generated_cov))
    Ne = eigs1.shape[0]
    indicators = np.linspace(1, Ne, num=Ne, dtype=int)
    plt.scatter(indicators, eigs1, label=labels[0])
    plt.scatter(indicators, eigs2, label=labels[1])
    plt.legend()
    plt.xlabel("Eigenvalue Indicator")
    plt.ylabel("Eigenvalues")
    plt.title("Plot of Eigenvalues for Covariance Matrices")
    plt.show()


def compare_against_isotropic_Gaussian(forward_samples: np.ndarray, td: int, diffTime: Union[int, float],
                                       rng: np.random.Generator, filepath: str) -> None:
    """
    Generate qualitative comparative plots between forward samples at 'diffTime' and standard isotropic Gaussian
    :param forward_samples: Samples from forward diffusion
    :param td: Dimension of samples
    :param diffTime: Diffusion time index
    :param rng: Random number generator
    :param filepath: Path to save images
    :return: None
    """
    assert (forward_samples.shape[1] == td)
    stdn_samples = rng.normal(size=(forward_samples.shape[0], td))
    labels = ["Forward Samples at time {}".format(diffTime), "Standard Normal Samples"]
    if td == 2: plot_dataset(forward_samples=forward_samples, reverse_samples=stdn_samples, labels=labels,
                             image_path=filepath)
    plot_heatmap(np.cov(forward_samples, rowvar=False), annot=False if td > 16 else True,
                 title="Covariance matrix at forward time {}".format(diffTime), filepath=filepath)


def compare_fBm_to_approximate_fBm(generated_samples: np.ndarray, h: float, td: int, rng: np.random.Generator) -> None:
    """
    Plot tSNE comparing final reverse-time diffusion fBm samples to fBm samples generated from approximate simulation
    methods.
        :param generated_samples: Exact fBm samples
        :param h: Hurst index
        :param td: Dimension of each sample
        :param rng: Random number generator
        :return: None
    """
    generator = FractionalBrownianNoise(H=h, rng=rng)
    S = min(20000, generated_samples.shape[0])
    approx_samples = np.empty((S, td))
    for _ in range(S):
        approx_samples[_, :] = generator.paxon_simulation(
            N_samples=td).cumsum()  # TODO: Are we including initial sample?
    plot_tSNE(generated_samples, y=approx_samples,
              labels=["Reverse Diffusion Samples", "Approximate Samples: Paxon Method"],
              image_path=project_config.ROOT_DIR + "pngs/tSNE_approxfBm_vs_generatedfBm_H{:.3e}_T{}".format(h, td))


def compare_fBm_to_normal(h: float, generated_samples: np.ndarray, td: int, rng: np.random.Generator) -> None:
    """
    Plot tSNE comparing reverse-time diffusion samples to standard normal samples
        :param h: Hurst index.
        :param generated_samples: Exact fBm samples
        :param td: Dimension of each sample
        :param rng: Random number generator
        :return: None
    """
    S = min(20000, generated_samples.shape[0])
    normal_rvs = np.empty((S, td))
    for _ in range(S):
        normal_rvs[_, :] = rng.standard_normal(td)
    plot_tSNE(generated_samples, y=normal_rvs, labels=["Reverse Diffusion Samples", "Standard Normal RVS"],
              image_path=project_config.ROOT_DIR + "pngs/tSNE_normal_vs_generatedfBm_H{:.3e}_T{}".format(h, td))


def plot_errors_ts(diff_time_space: np.ndarray, errors: np.ndarray, plot_title: str, path: str) -> None:
    """
    Plot error over time, averaged over many samples at each point in time.
        :param diff_time_space: Diffusion time space
        :param errors: Score erros
        :param plot_title:  Title for plot
        :param path: Path to save figure
        :return: None
    """
    plt.scatter(diff_time_space, errors, label="MSE")
    plt.xlabel("Diffusion Time")
    plt.ylabel("MSE")
    plt.title(plot_title)
    plt.savefig(path)
    plt.show()


def plot_errors_heatmap(errors: np.ndarray, plot_title: str, path: str, xticks: list, yticks: list) -> None:
    """
    Plot heat map of errors over time and space
        :param errors: Matrix with errors over space and time
        :param plot_title: Title for plot
        :param path: Path to save figure
        :param xticks: Correct Dimension Numbers (0-indexed)
        :param yticks: Correct Reverse-diffusion index (0-indexed)
        :return: None
    """
    ax = sns.heatmap(errors, annot=False, xticklabels=xticks, yticklabels=yticks, annot_kws={'size': 15})
    ax.set_xlabel("Dimension")
    ax.set_ylabel("Reverse-Time Diffusion Time")
    plt.title(plot_title)
    plt.savefig(path)
    plt.show()


def make_gif(frame_folder_path: str, process_str_path: str) -> None:
    """
    Function to take sequence of images and turn into GIF
        :param frame_folder_path: Path where images are stored
        :param process_str_path: String identifying which images to turn into a GIF
        :return: None
    """
    images = []
    try:
        images = [image for image in glob.glob("{}{}*.png".format(frame_folder_path, process_str_path))]
        images = sorted(images, key=lambda x: int(x.replace(".png", "").split("_")[-1]))
        frames = [Image.open(image) for image in images]
    except RuntimeError as e:
        for image in images:
            os.remove(image)
        raise RuntimeError("Error {}".format(e))
    frame_one = frames[0]
    # Proceed to cleaning contents of folder
    for image in images:
        os.remove(image)
    frame_one.save(frame_folder_path + process_str_path + ".gif", format="GIF", append_images=frames, save_all=True,
                   duration=1000, loop=1)


def plot_and_save_diffused_fBm_snapshot(samples: torch.Tensor, cov: torch.Tensor, save_path: str, x_label: str,
                                        y_label: str, plot_title: str) -> None:
    """
    Function to save figure of diffusion samples scatter plot and contours of theoretical marginal
        :param samples: Samples for scatter plot
        :param cov: Covariance matrix of theoretical marginal
        :param save_path: Path to save figure
        :param x_label: X-axis label
        :param y_label: Y-axis label
        :param plot_title: Title for plot
        :return: None
    """
    d1, d2 = samples[:, 0], samples[:, 1]
    expected_dist = stats.multivariate_normal(mean=None, cov=cov)
    t = (np.linspace(stats.norm(loc=0, scale=np.sqrt(cov[0, 0])).ppf(0.001),
                     stats.norm(loc=0, scale=np.sqrt(cov[0, 0])).ppf(0.999), 500))
    h = (np.linspace(stats.norm(loc=0, scale=np.sqrt(cov[1, 1])).ppf(0.001),
                     stats.norm(loc=0, scale=np.sqrt(cov[1, 1])).ppf(0.999), 500))
    z = expected_dist.pdf(np.dstack(np.meshgrid(t, h)))
    fig, ax = plt.subplots()
    ax.contour(t, h, z, levels=100, alpha=0.6)
    ax.scatter(d1, d2)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(plot_title)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def my_pairplot(samples: torch.Tensor, row_idxs: np.ndarray, col_idxs: np.ndarray, cov: torch.Tensor, image_path: str,
                suptitle: str) -> None:
    """
    Function to produce correlation matrix pairplots
        :param samples: Data
        :param row_idxs: Dimensions along row
        :param col_idxs: Dimensions along columns
        :param cov: Covariance matrix
        :param image_path: Save path for image
        :param suptitle: Title for image
        :return: None
    """
    device = samples.device
    fig, ax = plt.subplots(row_idxs.shape[0], col_idxs.shape[0], squeeze=False)
    N = row_idxs.shape[0]
    M = col_idxs.shape[0]
    for idrow in range(N):
        ax[idrow, 0].set_ylabel("Dim {}".format(row_idxs[idrow] + 1))
        row_dim = row_idxs[idrow]
        for idcol in range(idrow, M):
            col_dim = col_idxs[idcol]
            if row_dim == col_dim:
                ax[idrow, idcol].hist(samples[:, row_dim], bins=100, density=True)
                std = torch.sqrt(cov[row_dim, row_dim])
                linspace = np.linspace(stats.norm.ppf(0.001, loc=0, scale=std), stats.norm.ppf(0.999, loc=0, scale=std),
                                       500)
                expected_pdf = stats.norm.pdf(linspace, loc=0, scale=std)
                ax[idrow, idcol].plot(linspace, expected_pdf)
            else:
                dim_pair = torch.Tensor(np.array([row_dim, col_dim])).to(torch.int32).to(device)
                d1, d2 = samples[:, row_dim], samples[:, col_dim]
                paired_cov = torch.index_select(torch.index_select(cov, dim=0, index=dim_pair), dim=1, index=dim_pair)
                expected_dist = stats.multivariate_normal(mean=None, cov=paired_cov)
                t = (np.linspace(stats.norm(loc=0, scale=np.sqrt(paired_cov[0, 0])).ppf(0.001),
                                 stats.norm(loc=0, scale=np.sqrt(paired_cov[0, 0])).ppf(0.999), 500))
                h = (np.linspace(stats.norm(loc=0, scale=np.sqrt(paired_cov[1, 1])).ppf(0.001),
                                 stats.norm(loc=0, scale=np.sqrt(paired_cov[1, 1])).ppf(0.999), 500))
                z = expected_dist.pdf(np.dstack(np.meshgrid(t, h)))
                ax[idrow, idcol].contour(t, h, z, levels=100, alpha=0.6)
                ax[idrow, idcol].scatter(d1, d2, s=2, color="red")
                ax[idrow, idcol].tick_params(which="both", bottom=False, top=False)
        if idrow == (N - 1):
            for j in range(M):
                ax[idrow, j].set_xlabel("Dim {}".format(col_idxs[j] + 1))

    plt.suptitle(suptitle)
    plt.savefig(image_path)
    plt.close()


def hurst_estimation(fBm_samples: np.ndarray, sample_type: str, isfBm: bool, true_hurst: float,
                     show: bool = True) -> pd.DataFrame:
    approx_fBn = reduce_to_fBn(fBm_samples, reduce=isfBm)
    # even_approx_fBn = approx_fBn[:, ::2]  # Every even index
    S = approx_fBn.shape[0]
    with mp.Pool(processes=9) as pool:
        res = pool.starmap(partial(optimise_whittle, data=approx_fBn), [(fidx,) for fidx in range(S)])
    hs = (pd.DataFrame(res, columns=["DF index", "Hurst Estimate"]).set_index("DF index").rename_axis([None], axis=0))
    # with mp.Pool(processes=9) as pool:
    #   even_hs = pool.starmap(partial(optimise_whittle, data=even_approx_fBn), [(fidx,) for fidx in range(S)])
    # my_hs = [np.array(hs), np.array(even_hs)]
    # titles = ["All", "Even"]
    my_hs = [hs.values]
    titles = ["All"]
    for i in range(len(my_hs)):
        mean, std = my_hs[i].mean(), my_hs[i].std()
        print(mean)
        print(std)
        if show:
            fig, ax = plt.subplots()
            ax.axvline(x=true_hurst, color="blue", label="True Hurst")
            plot_histogram(my_hs[i], num_bins=150, xlabel="H", ylabel="density",
                           plottitle="Histogram of {} {} samples' estimated Hurst parameter".format(titles[i],
                                                                                                    sample_type),
                           fig=fig, ax=ax)
            plt.show()
            plt.close()
            # Repeat with constrained axis
            fig, ax = plt.subplots()
            ax.axvline(x=true_hurst, color="blue", label="True Hurst")
            plot_histogram(my_hs[i], num_bins=150, xlabel="H", ylabel="density",
                           plottitle="Constrained hist of {} {} samples' estimated Hurst parameter".format(titles[i],
                                                                                                           sample_type),
                           fig=fig, ax=ax)
            ax.set_xlim(mean - 5 * std, mean + 5 * std)
            plt.show()
            plt.close()
    return hs
