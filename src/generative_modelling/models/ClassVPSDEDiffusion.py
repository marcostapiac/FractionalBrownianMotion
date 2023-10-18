from typing import Union, Tuple

import torch
from torch import nn

from src.generative_modelling.models.TimeDependentScoreNetworks.ClassNaiveMLP import NaiveMLP
from src.generative_modelling.models.TimeDependentScoreNetworks.ClassTimeSeriesScoreMatching import \
    TimeSeriesScoreMatching


class VPSDEDiffusion(nn.Module):
    def __init__(self, beta_max: float, beta_min: float, ):
        super().__init__()
        self._beta_max = beta_max
        self._beta_min = beta_min

    def get_discretised_beta(self, diff_index: torch.Tensor, max_diff_steps: int) -> torch.Tensor:
        """
        Return discretised variance value at corresponding forward diffusion indices
            :param diff_index: FORWARD diffusion index
            :return: Beta value
        """
        device = diff_index.device
        max_diff_steps = torch.Tensor([max_diff_steps]).to(device)
        beta_min = (self.get_beta_min()).to(device) / max_diff_steps
        beta_max = (self.get_beta_max()).to(device) / max_diff_steps
        assert (beta_max < max_diff_steps)
        return beta_min + (beta_max - beta_min) * diff_index / (max_diff_steps - 1)

    def get_discretised_alpha(self, diff_index: int, max_diff_steps: int) -> torch.Tensor:
        """
        Return DDPM alpha value at corresponding forward diffusion index
            :param diff_index: FORWARD diffusion index
            :return: Alpha value
        """
        return torch.cumprod(
            1. - self.get_discretised_beta(torch.linspace(start=0, end=diff_index), max_diff_steps=max_diff_steps),
            dim=0)

    def get_beta_min(self) -> torch.Tensor:
        """ Return minimum variance parameter as a torch.Tensor """
        return torch.Tensor([self._beta_min])

    def get_beta_max(self) -> torch.Tensor:
        """ Return maximum variance parameter as a torch.Tensor """
        return torch.Tensor([self._beta_max])

    @staticmethod
    def noising_process(dataSamples: torch.Tensor, effTimes: torch.Tensor) -> [torch.Tensor, torch.Tensor]:
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
        return xts, -epsts / torch.sqrt((1. - torch.exp(-effTimes)))

    @staticmethod
    def get_loss_weighting(eff_times: torch.Tensor) -> torch.Tensor:
        """
        Weighting for objective function during training
            :param eff_times: Effective SDE time
            :return: Square root of weight
        """
        return torch.sqrt(1. - torch.exp(-eff_times))

    def get_eff_times(self, diff_times: torch.Tensor) -> torch.Tensor:
        """
        Return effective SDE times
            :param diff_times: Discrete times at which we evaluate SDE
            :return: Effective time
        """
        device = diff_times.device
        beta_max = self.get_beta_max().to(device)
        beta_min = self.get_beta_min().to(device)
        return (0.5 * diff_times ** 2 * (beta_max - beta_min) + diff_times * beta_min)

    def prior_sampling(self, shape: Tuple[int, int]) -> torch.Tensor:
        """ Sample from the target in the forward diffusion """
        return torch.randn(shape)

    def get_ancestral_sampling(self, x: torch.Tensor, t: torch.Tensor,
                               score_network: Union[NaiveMLP, TimeSeriesScoreMatching],
                               diff_index: torch.Tensor, max_diff_steps: int) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute parameters for one-step in ancestral sampling of reverse-time diffusion
            :param x: Current reverse-time diffusion sample
            :param t: Current reverse-time diffusion time
            :param score_network: Trained score matching function
            :param diff_index: REVERSE diffusion index
            :param max_diff_steps: Maximum number of diffusion steps
            :return:
                - Predicted Score
                - Ancestral Sampling Drift
                - Ancestral Sampling Diffusion Coefficient
        """
        score_network.eval()
        with torch.no_grad():
            predicted_score = score_network.forward(x, t.squeeze(-1)).squeeze(1)
            beta_t = self.get_discretised_beta(max_diff_steps - 1 - diff_index, max_diff_steps)
        return predicted_score, x * (2. - torch.sqrt(1. - beta_t)) + beta_t * predicted_score, torch.sqrt(beta_t)
