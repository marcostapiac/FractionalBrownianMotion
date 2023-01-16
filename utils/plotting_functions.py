import matplotlib
import matplotlib.pyplot as plt
from utils.math_functions import np
import numbers


def plot_subplots(time_ax, lines, label_args, xlabels, ylabels, title, isLatex=True, fig=None, ax=None):
    """ Plotting function ot plot multiple traces in same figure but different axis"""
    try:
        assert (len(lines) == len(xlabels) and len(lines) == len(ylabels))
    except AssertionError:
        return RuntimeError("Please add as many x-y axis labels as lines to plot")
    if isLatex:
        matplotlib.rcParams.update({
            'font.family': 'serif',
            'text.usetex': True,
            'pgf.rcfonts': False,
        })
    else:
        plt.style.use('ggplot')
    if (fig and ax) is None:
        L = len(lines)
        fig, ax = plt.subplots(L, 1)
    for i in range(len(lines)):
        ax[i].plot(time_ax, lines[i], label=label_args[i], lw=1.1, marker='.', markersize=1)
        ax[i].set_xlabel(xlabels[i])
        ax[i].set_ylabel(ylabels[i])
        ax[i].xaxis.label.set_color('white')  # setting up X-axis label color to yellow
        ax[i].yaxis.label.set_color('white')
        ax[i].tick_params(axis='x', colors='white')  # setting up X-axis tick color to red
        ax[i].tick_params(axis='y', colors='white')
        ax[i].grid(visible=True)
        ax[i].legend()

    fig.patch.set_alpha(0.0)
    fig.suptitle(title, color="white")
    plt.tight_layout()


def plot(time_ax, lines, label_args, xlabel, ylabel, title, isLatex=True, fig=None, ax=None, saveTransparent=False):
    """ Plotting function to plot multiple traces in same figure and axis object"""
    plt.style.use('ggplot')
    if isLatex:
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


def plot_fBm_process(time_ax, paths, label_args, xlabel=None, ylabel=None, title="Sample Paths", isLatex=True,
                     fig=None,
                     ax=None):
    plt.style.use('ggplot')
    if isLatex:
        matplotlib.rcParams.update({
            'font.family': 'serif',
            'text.usetex': True,
            'pgf.rcfonts': False,
        })
    else:
        plt.style.use('ggplot')
    if (fig and ax) is None:
        fig, ax = plt.subplots()
    for i in range(len(paths)):
        ax.step(time_ax, paths[i], label="$H = " + str(round(label_args[i], 3)) + "$", lw=1.2)
    if (xlabel and ylabel) is None:
        ax.set_xlabel("Time", )
        ax.set_ylabel("Position")
    else:
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
    fig.patch.set_alpha(0.0)
    ax.xaxis.label.set_color('white')  # setting up X-axis label color to yellow
    ax.yaxis.label.set_color('white')
    ax.tick_params(axis='x', colors='white')  # setting up X-axis tick color to red
    ax.tick_params(axis='y', colors='white')
    ax.set_title(title, color="white")
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


def histogramplot(rvs, pdf_vals, axis, num_bins=100, xlabel="", ylabel="", plottitle="", plottlabel="", ax=None):
    """ Function to compare generated process with density at t = T_{horizon} """
    plt.style.use('ggplot')
    if ax is None:
        ax = plt.gca()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(plottitle)
    binvals, _, _ = plt.hist(rvs, num_bins, density=True, label="Histogram of Process at $t = T_{horizon}$")
    ax.plot(axis, pdf_vals, label=plottlabel)
    plt.legend()
    plt.show()


def boxplot(data, xlabel="", ylabel="", plottitle="", dataLabels="", ax=None):
    plt.style.use('ggplot')
    if ax is None:
        ax = plt.gca()
    ax.boxplot(data, labels=dataLabels)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(plottitle)
    plt.legend()
    plt.show()
