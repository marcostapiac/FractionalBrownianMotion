import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE

import configs.VESDE.fBm_T32_H07 as configfile32
import configs.VESDE.fBm_T2_H07 as configfile2
from src.generative_modelling.models.ClassVESDEDiffusion import VESDEDiffusion
from src.generative_modelling.models.TimeDependentScoreNetworks.ClassNaiveMLP import NaiveMLP
from src.generative_modelling.models.TimeDependentScoreNetworks.ClassTimeSeriesScoreMatching import \
    TimeSeriesScoreMatching
from utils.plotting_functions import plot_dataset, plot_tSNE


def visualise_forward_diffusion():
    config32 = configfile32.get_config()
    config2 = configfile2.get_config()
    data32 = np.load(config32.data_path, allow_pickle=True).cumsum(axis=1)[:1000,:]
    diffusion32 = VESDEDiffusion(stdMax=config32.std_max, stdMin=config32.std_min)
    data2 = np.load(config2.data_path, allow_pickle = True).cumsum(axis=1)[:1000,:]
    diffusion2 = VESDEDiffusion(stdMax=config2.std_max, stdMin=config2.std_min)

    diffused_2 = []
    diffused_32 = []
    Ndiff = 100
    ts = np.linspace(0.,1., num=Ndiff)
    for i in range(Ndiff):
        xts, _ = diffusion2.noising_process(torch.Tensor(data2), diffusion2.get_eff_times(diff_times=torch.Tensor([ts[i]])))
        diffused_2.append(xts)
        xts, _ = diffusion32.noising_process(torch.Tensor(data32),
                                            diffusion32.get_eff_times(diff_times=torch.Tensor([ts[i]])))
        diffused_32.append(xts)
        #plot_dataset(diffused_2[-1].numpy(), diffused_2[-1].numpy(), image_path="", labels=["Time " + str(i)])
        tsne=TSNE(n_components=3).fit_transform(diffused_32[-1].numpy())
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.scatter3D(tsne[:,0], tsne[:,1], tsne[:,2])


visualise_forward_diffusion()