import ast

import numpy as np
import pandas as pd
import scipy
from matplotlib import pyplot as plt
from ml_collections import ConfigDict
from scipy.stats import ks_2samp

from utils.plotting_functions import plot_histogram, plot_and_save_boxplot


def plot_fBm_results_from_csv(config: ConfigDict) -> None:
    """
    Function to plot quantitative metrics
        :param config: ML experiment metrics
        :return: None
    """
    df = pd.read_csv(config.experiment_path+"_Nepochs{}.csv.gzip".format(config.max_epochs), compression="gzip", index_col=[0])

    # Mean Abs Difference
    # plot_and_save_boxplot(data=df.loc[config.exp_keys[0]].astype(float).to_numpy(), xlabel="1",
    #                      ylabel=config.exp_keys[0], title_plot="Mean Absolute Percentage Difference in Mean Vector",
    #                      dataLabels=[None], toSave=False, saveName="")

    # Covariance Abs Difference
    plot_and_save_boxplot(data=df.loc[config.exp_keys[1]].astype(float).to_numpy(), xlabel="1",
                          ylabel=config.exp_keys[1], title_plot="Absolute Percentage Difference in Covariance Matrix",
                          dataLabels=[None], toSave=False, saveName="")

    # Exact Sample Chi2 Test Statistic Histogram
    dfs = config.timeDim
    fig, ax = plt.subplots()
    org_chi2 = df.loc[config.exp_keys[4]].to_list()
    true_chi2 = []
    for j in range(config.num_runs):
        true_chi2 += (ast.literal_eval(org_chi2[j]))
    xlinspace = np.linspace(scipy.stats.chi2.ppf(0.0001, dfs), scipy.stats.chi2.ppf(0.9999, dfs), 1000)
    pdfvals = scipy.stats.chi2.pdf(xlinspace, df=dfs)
    plot_histogram(np.array(true_chi2), pdf_vals=pdfvals, xlinspace=xlinspace, num_bins=200, xlabel="Chi2 Statistic",
                   ylabel="density", plotlabel="Chi2 with {} DoF".format(dfs),
                   plottitle="Histogram of exact samples' Chi2 Test Statistic", fig=fig, ax=ax)
    plt.show()
    print(ks_2samp(true_chi2, scipy.stats.chi2.rvs(df=dfs, size=len(true_chi2)), alternative="two-sided"))
    # Synthetic Sample Chi2 Test Statistic Histogram
    fig, ax = plt.subplots()
    f_chi2 = df.loc[config.exp_keys[5]].to_list()
    synth_chi2 = []
    for j in range(config.num_runs):
        synth_chi2 += (ast.literal_eval(f_chi2[j]))
    plot_histogram(np.array(synth_chi2), pdf_vals=pdfvals, xlinspace=xlinspace, num_bins=200, xlabel="Chi2 Statistic",
                   ylabel="density", plotlabel="Chi2 with {} DoF".format(dfs),
                   plottitle="Histogram of synthetic samples' Chi2 Test Statistic", fig=fig, ax=ax)
    plt.show()
    print(ks_2samp(synth_chi2, scipy.stats.chi2.rvs(df=dfs, size=len(synth_chi2)), alternative="two-sided"))

    """if str(df.loc[config.exp_keys[7]][0]) != "nan":
        # Predictive Scores
        org_pred = df.loc[config.exp_keys[7]].astype(float).to_numpy().reshape((config.num_runs,))
        synth_pred = df.loc[config.exp_keys[8]].astype(float).to_numpy().reshape((config.num_runs,))
        plot_and_save_boxplot(data=np.array([org_pred, synth_pred]).reshape((config.num_runs, 2)), xlabel="1",
                              ylabel=config.exp_keys[5],
                              title_plot="Predictive Scores", dataLabels=["True", "Generated"], toSave=False,
                              saveName="")
    if str(df.loc[config.exp_keys[9]][0]) != "nan":
        # Discriminative Scores
        org_disc = df.loc[config.exp_keys[9]].astype(float).to_numpy().reshape((config.num_runs,))
        synth_disc = df.loc[config.exp_keys[10]].astype(float).to_numpy().reshape((config.num_runs,))
        plot_and_save_boxplot(data=np.array([org_disc, synth_disc]).reshape((config.num_runs, 2)), xlabel="1",
                              ylabel=config.exp_keys[5],
                              title_plot="Discriminative Scores", dataLabels=["True", "Generated"], toSave=False,
                              saveName="")
    """
    # Histogram of exact samples Hurst parameter
    fig, ax = plt.subplots()
    ax.axvline(x=config.hurst, color="blue", label="True Hurst")

    literal_trues = df.loc[config.exp_keys[11]].to_list()
    true_Hs = []
    for j in range(config.num_runs):
        true_Hs += (ast.literal_eval(literal_trues[j]))
    plot_histogram(np.array(true_Hs), num_bins=200, xlabel="H", ylabel="density",
                   plottitle="Histogram of exact samples' estimated Hurst parameter", fig=fig, ax=ax)
    plt.show()

    # Histogram of exact samples Hurst parameter
    fig, ax = plt.subplots()
    ax.axvline(x=config.hurst, color="blue", label="True Hurst")
    literal_synths = df.loc[config.exp_keys[12]].to_list()
    synth_Hs = []
    for j in range(config.num_runs):
        synth_Hs += (ast.literal_eval(literal_synths[j]))
    plot_histogram(np.array(synth_Hs), num_bins=200, xlabel="H", ylabel="density",
                   plottitle="Histogram of synthetic samples' estimated Hurst parameter", fig=fig, ax=ax)
    plt.show()

    print(ks_2samp(synth_Hs, true_Hs, alternative="two-sided"))

    """pvals = df.loc[config.exp_keys[6]].to_list()
    for i in range(config.timeDim):
        pval = []
        for j in range(config.num_runs):
            pval_j = ast.literal_eval(pvals[j])
            pval.append(pval_j[i])
        plot_and_save_boxplot(data=np.array(pval), xlabel="1",
                              ylabel="KS Test p-value",
                              title_plot="KS p-val for dimension {}".format(i + 1), dataLabels=[None], toSave=False,
                              saveName="")"""


def run_plotting_from_csv() -> None:
    from configs.VESDE.fBm_T32_H07 import get_config
    config = get_config()
    plot_fBm_results_from_csv(config)


if __name__ == "__main__":
    run_plotting_from_csv()
