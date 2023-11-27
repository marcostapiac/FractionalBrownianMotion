import abc
from typing import Union, Tuple

import torch

from src.generative_modelling.models.ClassOUSDEDiffusion import OUSDEDiffusion
from src.generative_modelling.models.ClassVESDEDiffusion import VESDEDiffusion
from src.generative_modelling.models.ClassVPSDEDiffusion import VPSDEDiffusion
from src.generative_modelling.models.TimeDependentScoreNetworks.ClassNaiveMLP import NaiveMLP
from src.generative_modelling.models.TimeDependentScoreNetworks.ClassTimeSeriesScoreMatching import \
    TimeSeriesScoreMatching


class Predictor(abc.ABC):
    """ Base class for all predictor algorithms during reverse-time sampling """

    def __init__(self, diffusion: Union[VESDEDiffusion, VPSDEDiffusion, OUSDEDiffusion],
                 score_function: Union[NaiveMLP, TimeSeriesScoreMatching], end_diff_time: float, max_diff_steps: int,
                 device: Union[int, torch.device], sample_eps:float):
        super().__init__()
        self.score_network = score_function
        self.torch_device = device
        self.diffusion = diffusion
        self.end_diff_time = end_diff_time
        self.max_diff_steps = max_diff_steps
        self.sample_eps = sample_eps

        self.score_network = self.score_network.to(self.torch_device)
        # TODO: DDP
        # self.score_network = DDP(self.score_network, device_ids = [self.torch_device])

    def step(self, x_prev: torch.Tensor, t: torch.Tensor, diff_index: torch.Tensor) -> Tuple[
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
    def __init__(self, diffusion: Union[VESDEDiffusion, VPSDEDiffusion, OUSDEDiffusion],
                 score_function: Union[NaiveMLP, TimeSeriesScoreMatching], end_diff_time: float, max_diff_steps: int,
                 device: Union[int, torch.device], sample_eps: float):
        super().__init__(diffusion, score_function, end_diff_time, max_diff_steps, device, sample_eps)

    def step(self, x_prev: torch.Tensor, t: torch.Tensor, diff_index: torch.Tensor) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor]:
        dt = - (self.end_diff_time - self.sample_eps)/ self.max_diff_steps
        score, drift, diffusion = self.diffusion.get_reverse_sde(x_prev, score_network=self.score_network, t=t,
                                                                 dt=torch.Tensor([dt]).to(self.torch_device))
        z = torch.randn_like(x_prev)
        return drift + diffusion * z, score, z


class AncestralSamplingPredictor(Predictor):
    def __init__(self, diffusion: Union[VESDEDiffusion, VPSDEDiffusion],
                 score_function: Union[NaiveMLP, TimeSeriesScoreMatching], end_diff_time: float, max_diff_steps: int,
                 device: Union[int, torch.device], sample_eps:float):
        try:
            assert (type(diffusion) != OUSDEDiffusion)
        except AssertionError:
            raise NotImplementedError("Ancestral sampling is only valid for VE and VP diffusion models")
        super().__init__(diffusion, score_function, end_diff_time, max_diff_steps, device, sample_eps)

    def step(self, x_prev: torch.Tensor, t: torch.Tensor, diff_index: torch.Tensor) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor]:
        score, drift, diffusion = self.diffusion.get_ancestral_sampling(x_prev, t=t, score_network=self.score_network,
                                                                        diff_index=diff_index,
                                                                        max_diff_steps=self.max_diff_steps)
        z = torch.randn_like(x_prev)
        return drift + diffusion * z, score, z
