import abc
from typing import Union, Tuple

import numpy as np
import torch

from src.generative_modelling.models.ClassOUDiffusion import OUDiffusion
from src.generative_modelling.models.ClassVESDEDiffusion import VESDEDiffusion
from src.generative_modelling.models.ClassVPSDEDiffusion import VPSDEDiffusion
from src.generative_modelling.models.TimeDependentScoreNetworks.ClassNaiveMLP import NaiveMLP
from src.generative_modelling.models.TimeDependentScoreNetworks.ClassTimeSeriesScoreMatching import \
    TimeSeriesScoreMatching


class Predictor(abc.ABC):
    """ Base class for all predictor algorithms during reverse-time sampling """

    def __init__(self, diffusion: Union[VESDEDiffusion, VPSDEDiffusion, OUDiffusion],
                 score_function: Union[NaiveMLP, TimeSeriesScoreMatching], end_diff_time:float, max_diff_steps: int):
        super().__init__()
        self.score_network = score_function
        self.diffusion = diffusion
        self.end_diff_time = end_diff_time
        self.max_diff_steps = max_diff_steps

    def step(self, x_prev: torch.Tensor, t: torch.Tensor, diff_index: int) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Single step during reverse-time sampling
            :param x_prev: Current
            :param t: Reverse-time SDE true time
            :param diff_index: Diffusion index
            :returns:
                - Next reverse-time sample
                - Score network output
                - Standard noise
        """
        return x_prev, x_prev, x_prev


class EulerMaruyamaPredictor(Predictor):
    # TODO: Is this not the same as reverse-time diffusion discretisation?
    def __init__(self, diffusion: Union[VESDEDiffusion, VPSDEDiffusion, OUDiffusion],
                 score_function: Union[NaiveMLP, TimeSeriesScoreMatching], end_diff_time: float, max_diff_steps: int):
        super().__init__(diffusion, score_function, end_diff_time, max_diff_steps)

    def step(self, x_prev: torch.Tensor, t: torch.Tensor, diff_index: int) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor]:
        dt = -self.end_diff_time / self.max_diff_steps
        score, drift, diffusion = self.diffusion.get_reverse_sde(x_prev, score_network=self.score_network, t=t,
                                                                 diff_index=self.max_diff_steps - 1 - diff_index,
                                                                 max_diff_steps=self.max_diff_steps)
        z = torch.randn(size=x_prev.shape)
        return x_prev + drift * dt + np.sqrt(-dt) * z, score, z


class AncestralSamplingPredictor(Predictor):
    def __init__(self, diffusion: Union[VESDEDiffusion, VPSDEDiffusion],
                 score_function: Union[NaiveMLP, TimeSeriesScoreMatching], end_diff_time: float, max_diff_steps: int):
        try:
            assert (type(diffusion) != OUDiffusion)
        except AssertionError:
            raise NotImplementedError("Ancestral sampling is only valid for VE and VP diffusion models")
        super().__init__(diffusion, score_function, end_diff_time, max_diff_steps)

    def step(self, x_prev: torch.Tensor, t: torch.Tensor, diff_index: int) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor]:
        score, drift, diffusion = self.diffusion.get_ancestral_sampling(x_prev, t=t, score_network=self.score_network,
                                                                        diff_index=self.max_diff_steps - 1 - diff_index,
                                                                        max_diff_steps=self.max_diff_steps)
        z = torch.randn(size=x_prev.shape)
        return drift + diffusion * z, score, z
