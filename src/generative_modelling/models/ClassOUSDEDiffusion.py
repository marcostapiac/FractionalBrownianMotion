from typing import Tuple, Union

import numpy as np
import torch
from torch import nn

from src.generative_modelling.models.TimeDependentScoreNetworks.ClassConditionalLSTMTSScoreMatching import \
    ConditionalLSTMTSScoreMatching
from src.generative_modelling.models.TimeDependentScoreNetworks.ClassConditionalMarkovianTSScoreMatching import \
    ConditionalMarkovianTSScoreMatching
from src.generative_modelling.models.TimeDependentScoreNetworks.ClassConditionalTSScoreMatching import \
    ConditionalTSScoreMatching
from src.generative_modelling.models.TimeDependentScoreNetworks.ClassNaiveMLP import NaiveMLP
from src.generative_modelling.models.TimeDependentScoreNetworks.ClassTSScoreMatching import \
    TSScoreMatching


class OUSDEDiffusion(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def noising_process(dataSamples: torch.Tensor, effTimes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward diffusion in continuous time
            :param dataSamples: Initial time SDE value
            :param effTimes: Effective SDE time
            :return:
                - Forward diffused samples
                - Score

        """
        epsts = torch.randn_like(dataSamples)
        xts = torch.exp(-0.5 * effTimes) * dataSamples + torch.sqrt(1. - torch.exp(-effTimes)) * epsts
        return xts, -epsts / torch.sqrt(1. - torch.exp(-effTimes))

    @staticmethod
    def get_loss_weighting(eff_times: torch.Tensor) -> torch.Tensor:
        """
        Weighting for objective function during training
            :param eff_times: Effective SDE time
            :return: Square root of weight
        """
        return torch.sqrt(1. - torch.exp(-eff_times))

    @staticmethod
    def get_eff_times(diff_times: torch.Tensor) -> torch.Tensor:
        """
        Return effective SDE times
            :param diff_times: Discrete times at which we evaluate SDE
            :return: Effective time
        """
        return diff_times

    @staticmethod
    def prior_sampling(shape: Tuple[int, int]) -> torch.Tensor:
        """ Sample from the target in the forward diffusion
        :param shape: Shape of desired sample
        :returns: Normal random sample
        """
        return torch.randn(shape)

    @staticmethod
    def get_reverse_sde(x_prev: torch.Tensor, score_network: Union[
        NaiveMLP, TSScoreMatching, ConditionalLSTMTSScoreMatching, ConditionalTSScoreMatching, ConditionalMarkovianTSScoreMatching],
                        t: torch.Tensor, dt: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """ Euler Maruyama discretisation of reverse-time SDE
            :param x_prev: Current sample
            :param score_network: Trained score matching network
            :param t: Reverse diffusion times
            :param dt: Discretisation step
            :returns
                - Predicted score
                - Drift
                - Diffusion
        """
        assert (dt < 0.)
        score_network.eval()
        with torch.no_grad():
            predicted_score = score_network.forward(x_prev, t.squeeze(-1)).squeeze(1)
            drift = x_prev + (-0.5 * x_prev - predicted_score) * dt
            diffusion = np.sqrt(-dt)
        return predicted_score, drift, diffusion
