import abc
from typing import Union, Tuple

import torch

from src.generative_modelling.models.ClassOUSDEDiffusion import OUSDEDiffusion
from src.generative_modelling.models.ClassVESDEDiffusion import VESDEDiffusion
from src.generative_modelling.models.ClassVPSDEDiffusion import VPSDEDiffusion
from src.generative_modelling.models.TimeDependentScoreNetworks.ClassConditionalTimeSeriesScoreMatching import \
    ConditionalTimeSeriesScoreMatching
from src.generative_modelling.models.TimeDependentScoreNetworks.ClassNaiveMLP import NaiveMLP
from src.generative_modelling.models.TimeDependentScoreNetworks.ClassTimeSeriesScoreMatching import \
    TimeSeriesScoreMatching


class Predictor(abc.ABC):
    """ Base class for all predictor algorithms during reverse-time sampling """

    def __init__(self, diffusion: Union[VESDEDiffusion, VPSDEDiffusion, OUSDEDiffusion],
                 score_function: Union[NaiveMLP, TimeSeriesScoreMatching, ConditionalTimeSeriesScoreMatching],
                 end_diff_time: float, max_diff_steps: int,
                 device: Union[int, torch.device], sample_eps: float):
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
        dt = - (self.end_diff_time - self.sample_eps) / self.max_diff_steps
        score, drift, diffusion = self.diffusion.get_reverse_sde(x_prev, score_network=self.score_network, t=t,
                                                                 dt=torch.Tensor([dt]).to(self.torch_device))
        z = torch.randn_like(x_prev)
        return drift + diffusion * z, score, z


class AncestralSamplingPredictor(Predictor):
    def __init__(self, diffusion: Union[VESDEDiffusion, VPSDEDiffusion],
                 score_function: Union[NaiveMLP, TimeSeriesScoreMatching], end_diff_time: float, max_diff_steps: int,
                 device: Union[int, torch.device], sample_eps: float):
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


class ConditionalAncestralSamplingPredictor(Predictor):
    def __init__(self, diffusion: Union[VESDEDiffusion, VPSDEDiffusion],
                 score_function: ConditionalTimeSeriesScoreMatching, end_diff_time: float, max_diff_steps: int,
                 device: Union[int, torch.device], sample_eps: float):
        try:
            assert (type(diffusion) != OUSDEDiffusion)
        except AssertionError:
            raise NotImplementedError("Ancestral sampling is only valid for VE and VP diffusion models")
        super().__init__(diffusion, score_function, end_diff_time, max_diff_steps, device, sample_eps)

    def step(self, x_prev: torch.Tensor, feature: torch.Tensor, t: torch.Tensor, diff_index: torch.Tensor, ts_step:float, param_est_time:float) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, Union[None,torch.Tensor],Union[None,torch.Tensor]]:
        score, drift, diffusion = self.diffusion.get_conditional_ancestral_sampling(x=x_prev, t=t, feature=feature,
                                                                                    score_network=self.score_network,
                                                                                    diff_index=diff_index,
                                                                                    max_diff_steps=self.max_diff_steps)
        mean_est = None
        var_est = None
        if diff_index != torch.Tensor([param_est_time - 2]).to(diff_index.device):
            with torch.no_grad():
                z = torch.randn_like(drift)
                x_new = drift + diffusion * z
        else:
            z = torch.randn_like(drift)
            x_new = drift + diffusion * z
        if diff_index == torch.Tensor([param_est_time]).to(diff_index.device):
            # Zero out gradients to avoid accumulation
            self.score_network.zero_grad()
            # Compute gradients of output with respect to input_data
            #grad_score = torch.autograd.grad(outputs=score, inputs=x_prev, grad_outputs=torch.ones_like(score),
            #                               retain_graph=False)[0].squeeze(dim=-1)
            with torch.no_grad():
                diffusion_mean2 = torch.atleast_2d(torch.exp(-self.diffusion.get_eff_times(diff_times=t))).T
                diffusion_var = 1.-diffusion_mean2
                # TODO: element wise multiplication along dim=1 (0-indexed) without squeezing
                var_est = torch.ones((x_prev.shape[0],1))#-torch.pow(diffusion_mean2, -1)*(torch.pow(grad_score, -1)+diffusion_var)
                grad_score = torch.pow(-(diffusion_var+diffusion_mean2*ts_step), -1)
                mean_est = (torch.pow(grad_score, -1)*score.squeeze(dim=-1))-x_prev.squeeze(dim=-1)
                mean_est *= -torch.pow(diffusion_mean2, -0.5)/ts_step
                assert(var_est.shape == (x_prev.shape[0],1) and mean_est.shape == (x_prev.shape[0],1))
        return x_new, score, z, mean_est, var_est
