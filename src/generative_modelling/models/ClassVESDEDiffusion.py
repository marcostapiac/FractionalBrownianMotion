from typing import Tuple, Union

import torch
from torch import nn

from src.generative_modelling.models.TimeDependentScoreNetworks.ClassNaiveMLP import NaiveMLP
from src.generative_modelling.models.TimeDependentScoreNetworks.ClassTimeSeriesScoreMatching import \
    TimeSeriesScoreMatching


class VESDEDiffusion(nn.Module):
    """ Diffusion class for VE SDE model """

    def __init__(self, stdMax: float, stdMin: float) -> None:
        super().__init__()
        self.stdMax = stdMax
        self.stdMin = stdMin

    def get_var_max(self) -> torch.Tensor:
        """ Get maximum variance parameter """
        return torch.Tensor([self.stdMax ** 2]).to(torch.float32)

    def get_var_min(self) -> torch.Tensor:
        """ Get minimum variance parameter """
        return torch.Tensor([self.stdMin ** 2]).to(torch.float32)
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
        epsts = torch.randn_like(dataSamples)  # Already in same device as dataSamples
        return dataSamples + torch.sqrt(effTimes) * epsts, -epsts / torch.sqrt(effTimes)

    @staticmethod
    def get_loss_weighting(eff_times: torch.Tensor) -> torch.Tensor:
        """
        Weighting for objective function during training
            :param eff_times: Effective SDE time
            :return: Square root of weight
        """
        return torch.sqrt(eff_times)

    def get_eff_times(self, diff_times: torch.Tensor) -> torch.Tensor:
        """
        Return effective SDE times
            :param diff_times: Discrete times at which we evaluate SDE
            :return: Effective time
        """
        device = diff_times.device
        var_max = self.get_var_max().to(device)
        var_min = self.get_var_min().to(device)
        return var_min * (var_max / var_min) ** diff_times - var_min

    def prior_sampling(self, shape: Tuple[int, int]) -> torch.Tensor:
        """ Sample from the target in the forward diffusion
            :param shape: Dimension of required sample
            :return: Desired sample
        """
        return torch.sqrt(self.get_var_max()) * torch.randn(shape)

    def get_ancestral_var(self, max_diff_steps: torch.Tensor, diff_index: torch.Tensor) -> torch.Tensor:
        """
        Discretisation of noise schedule for ancestral sampling
            :param max_diff_steps: Number of reverse-time discretisation steps
            :param diff_index: Forward time diffusion index
            :return: None
        """
        device = diff_index.device
        var_max = self.get_var_max().to(device)
        var_min = self.get_var_min().to(device)
        vars = var_min * torch.pow((var_max / var_min), diff_index / (max_diff_steps - 1))
        return vars

    def get_ancestral_drift_coeff(self, max_diff_steps: torch.Tensor, diff_index: torch.Tensor):
        """
        Function to compute the noise term in front of score in VESDE model
        :param max_diff_steps: Maximum number of diffusion steps
        :param diff_index: FORWARD diffusion index
        :return: Noise term
        """
        device = diff_index.device
        curr_var = self.get_ancestral_var(max_diff_steps=max_diff_steps, diff_index=max_diff_steps - 1 - diff_index)
        next_var = self.get_ancestral_var(max_diff_steps=max_diff_steps,
                                          diff_index=max_diff_steps - 1 - diff_index - 1)
        noise_diff = curr_var - (next_var if diff_index < max_diff_steps - 1 else torch.Tensor([0]).to(device))
        return noise_diff

    def get_ancestral_drift(self, x: torch.Tensor, pred_score: torch.Tensor, diff_index: torch.Tensor,
                            max_diff_steps: torch.Tensor) -> torch.Tensor:
        """
        Compute drift for one-step of reverse-time diffusion
            :param x: Current samples
            :param pred_score: Predicted score vector
            :param diff_index: FORWARD diffusion index
            :param max_diff_steps: Maximum number of diffusion steps
            :return: Drift
        """
        noise_diff = self.get_ancestral_drift_coeff(max_diff_steps=max_diff_steps, diff_index=diff_index)
        return x + noise_diff * pred_score

    def get_ancestral_diff(self, diff_index: torch.Tensor, max_diff_steps: torch.Tensor) -> torch.Tensor:
        """
        Compute diffusion parameter for one-step of reverse-time diffusion
            :param diff_index: FORWARD diffusion index
            :param max_diff_steps: Maximum number of diffusion steps
            :return: Diffusion parameter
        """
        device = diff_index.device
        curr_var = self.get_ancestral_var(max_diff_steps=max_diff_steps, diff_index=max_diff_steps - 1 - diff_index)
        next_var = self.get_ancestral_var(max_diff_steps=max_diff_steps,
                                          diff_index=max_diff_steps - 1 - diff_index - 1)
        noise_diff = self.get_ancestral_drift_coeff(max_diff_steps=max_diff_steps, diff_index=diff_index)
        return torch.sqrt(
            noise_diff * next_var / curr_var if diff_index < max_diff_steps - 1 else torch.Tensor([0]).to(device))

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
            device = diff_index.device
            max_diff_steps = torch.Tensor([max_diff_steps]).to(device)
            predicted_score = score_network.forward(x, t.squeeze(-1)).squeeze(1)
            drift = self.get_ancestral_drift(x=x, pred_score=predicted_score, diff_index=diff_index,
                                             max_diff_steps=max_diff_steps)
            diffusion_param = self.get_ancestral_diff(diff_index=diff_index, max_diff_steps=max_diff_steps)
        return predicted_score, drift, diffusion_param
