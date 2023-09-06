from typing import Tuple

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
        epsts = torch.randn_like(dataSamples)
        return dataSamples + torch.sqrt(effTimes) * epsts, -epsts / torch.sqrt(effTimes)

    @staticmethod
    def get_loss_weighting(effTimes: torch.Tensor) -> torch.Tensor:
        """
        Weighting for objective function during training
        :param effTimes: Effective SDE time
        :return: Square root of weight
        """
        return torch.sqrt(effTimes)

    def get_eff_times(self, diff_times: torch.Tensor) -> torch.Tensor:
        """
        Return effective SDE times
        :param diff_times: Discrete times at which we evaluate SDE
        :return: Effective time
        """
        return self.get_var_min() * (self.get_var_max() / self.get_var_min()) ** diff_times

    def prior_sampling(self, shape: Tuple[int, int]) -> torch.Tensor:
        return torch.sqrt(self.get_var_max()) * torch.randn(shape)  # device= TODO

    def get_ancestral_var(self, max_diff_steps: int, diff_index: torch.Tensor) -> torch.Tensor:
        """
        Discretisation of noise schedule for ancestral sampling
        :return: None
        """
        var_max = self.get_var_max()
        var_min = self.get_var_min()
        vars = var_min * torch.pow((var_max / var_min), diff_index / (max_diff_steps - 1))
        return vars

    def get_ancestral_sampling(self, x: torch.Tensor, t: torch.Tensor,
                               score_network: Tuple[NaiveMLP, TimeSeriesScoreMatching],
                               diff_index: torch.Tensor, max_diff_steps: int):
        score_network.eval()
        with torch.no_grad():
            predicted_score = score_network.forward(x, t.squeeze(-1)).squeeze(1)
            curr_var = self.get_ancestral_var(max_diff_steps=max_diff_steps, diff_index=diff_index)
            next_var = self.get_ancestral_var(max_diff_steps=max_diff_steps, diff_index=diff_index - 1)
            drift_param = curr_var - (next_var if diff_index < max_diff_steps - 1 else torch.Tensor([0]))
            diffusion_param = torch.sqrt(drift_param * next_var / curr_var if diff_index < max_diff_steps - 1 else torch.Tensor([0]))
        return predicted_score, x+drift_param*predicted_score, diffusion_param


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
