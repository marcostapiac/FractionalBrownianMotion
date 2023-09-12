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
        epsts = torch.randn_like(dataSamples) # Already in same device as dataSamples
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
        var_max = self.get_var_max().to(diff_times.get_device())
        var_min = self.get_var_min().to(diff_times.get_device())
        return var_min * (var_max / var_min) ** diff_times

    def prior_sampling(self, shape: Tuple[int, int]) -> torch.Tensor:
        """ Sample from the target in the forward diffusion """
        return torch.sqrt(self.get_var_max()) * torch.randn(shape)  # device= TODO

    def get_ancestral_var(self, max_diff_steps: torch.Tensor, diff_index: torch.Tensor) -> torch.Tensor:
        """
        Discretisation of noise schedule for ancestral sampling
        :return: None
        """
        device = diff_index.get_device()
        var_max = self.get_var_max().to(device)
        var_min = self.get_var_min().to(device)
        vars = var_min * torch.pow((var_max / var_min), diff_index / (max_diff_steps - 1))
        return vars

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
            device = diff_index.get_device()
            max_diff_steps = torch.Tensor([max_diff_steps]).to(device)
            predicted_score = score_network.forward(x, t.squeeze(-1)).squeeze(1)
            curr_var = self.get_ancestral_var(max_diff_steps=max_diff_steps, diff_index=max_diff_steps - 1 - diff_index)
            next_var = self.get_ancestral_var(max_diff_steps=max_diff_steps,
                                              diff_index=max_diff_steps - 1 - diff_index - 1)
            drift_param = curr_var - (next_var if diff_index < max_diff_steps - 1 else torch.Tensor([0]).to(device))
            diffusion_param = torch.sqrt(
                drift_param * next_var / curr_var if diff_index < max_diff_steps - 1 else torch.Tensor([0]).to(device))
        return predicted_score, x + drift_param * predicted_score, diffusion_param


"""
    def reverse_process(self, data: np.ndarray, dataSize: int, timeDim: int, sampleEps: float,
                        sigNoiseRatio: float, numLangevinSteps: int,
                        timeLim: int = 0):
        assert (sampleEps >= self.trainEps)

        # Initialise
        x = torch.sqrt(self.get_var_max()) * torch.randn((dataSize, timeDim), device=self.torchDevice)
        self.model.eval()
        reverseTimes = torch.linspace(start=self.endDiffTime, end=sampleEps, steps=self.numDiffSteps)
        with torch.no_grad():
            for i in tqdm(iterable=(range(0, self.numDiffSteps - timeLim)), dynamic_ncols=False,
                          desc="Sampling :: ", position=0):

                i_s = i * torch.ones((dataSize, 1), dtype=torch.long, device=self.torchDevice)
                ts = reverseTimes[i_s]  # time-index for each data-sample
                predicted_score = self.score_network.forward(x, ts.squeeze(-1)).squeeze(1)
                drift_var_param = self.vars[self.numDiffSteps - 1 - i] - (
                    self.vars[self.numDiffSteps - 1 - i - 1] if i < self.numDiffSteps - 1 else torch.Tensor([0]))
                noise_var_param = drift_var_param * self.vars[self.numDiffSteps - 1 - i - 1] / self.vars[
                    self.numDiffSteps - 1 - i] if i < self.numDiffSteps - 1 else torch.Tensor([0])
                z = torch.randn_like(x)
                x = x + drift_var_param * predicted_score + torch.sqrt(noise_var_param) * z
                for _ in range(numLangevinSteps):
                    e = 2 * (sigNoiseRatio * np.linalg.norm(z) / np.linalg.norm(predicted_score)) ** 2
                    z = torch.randn_like(x)
                    x = x + e * predicted_score + np.sqrt(2. * e) * z
            return x.detach().numpy()  # Approximately distributed according to desired distribution
"""
