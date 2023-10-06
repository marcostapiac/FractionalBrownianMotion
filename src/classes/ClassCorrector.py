import abc
from typing import Union

import torch

from src.generative_modelling.models.ClassVESDEDiffusion import VESDEDiffusion
from src.generative_modelling.models.ClassVPSDEDiffusion import VPSDEDiffusion


class Corrector(abc.ABC):
    """ Base class for corrector algorithms """

    def __init__(self, N_lang: int, r: torch.Tensor, device: Union[int, torch.device]):
        self.max_lang_steps = N_lang
        self.torch_device = device
        print(self.torch_device)
        self.snr = r.to(self.torch_device)

    def _step(self, x: torch.Tensor, predicted_score: torch.Tensor, predictor_noise: torch.Tensor,
              diff_index: int, *args) -> torch.Tensor:
        """
        Abstract private method for single step of corrector
            :param x: Current sample
            :param predicted_score: Score network output for current time
            :param predictor_noise: Standard noise used in predictor step
            :param diff_index: Diffusion index of reverse time sampling
        :return: Next sample in discretised Langevin SDE
        """
        return x

    def sample(self, x: torch.Tensor, predicted_score: torch.Tensor, predictor_noise: torch.Tensor,
               diff_index: int, *args) -> torch.Tensor:
        """
        Parent function to run corrector sampling
            :param x: Current sample
            :param predicted_score: Score network output for current time
            :param predictor_noise: Standard noise used in predictor step
            :param diff_index: Diffusion index of reverse time sampling
        :return: Final sample after Langevin steps
        """
        for i in range(self.max_lang_steps):
            x = self._step(x, predicted_score, predictor_noise, diff_index, args)
        return x


class VESDECorrector(Corrector):
    """ Corrector class for VE SDE diffusion model """

    def __init__(self, N_lang: int, r: torch.Tensor, device: Union[int, torch.device], diffusion: VESDEDiffusion):
        super().__init__(N_lang, r, device)
        self.diffusion = diffusion

    def _step(self, x: torch.Tensor, predicted_score: torch.Tensor, predictor_noise: torch.Tensor,
              diff_index: int, *args) -> torch.Tensor:
        """ Single corrector step for VE SDE diffusion models
            :param x: Current sample
            :param predicted_score: Score network output for current time
            :param predictor_noise: Standard noise used in predictor step
            :param diff_index: Diffusion index of reverse time sampling
            :return: Final sample after Langevin steps
        """
        e = 2. * torch.pow(self.snr * torch.linalg.norm(predictor_noise) / torch.linalg.norm(predicted_score), 2.)
        return x + e * predicted_score + torch.sqrt(2. * e) * torch.randn_like(x)


class VPSDECorrector(Corrector):
    """ Corrector class for VP SDE diffusion model """

    def __init__(self, N_lang: int, r: torch.Tensor, device: Union[int, torch.device], diffusion: VPSDEDiffusion):
        super().__init__(N_lang, r, device)
        self.diffusion = diffusion

    def _step(self, x: torch.Tensor, predicted_score: torch.Tensor, predictor_noise: torch.Tensor,
              diff_index: int, *args) -> torch.Tensor:
        """
        Single corrector step for VP SDE diffusion models
            :param x: Current sample
            :param predicted_score: Score network output for current time
            :param predictor_noise: Standard noise used in predictor step
            :param diff_index: Diffusion index of reverse time sampling
            :param max_diff_steps: Maximum number of forward diffusion steps
            :return: Final sample after Langevin steps
        """
        assert len(args) == 1 and type(args[0]) == int
        max_diff_steps = args[0]
        e = 2. * self.diffusion.get_discretised_alpha(diff_index, max_diff_steps=max_diff_steps) * torch.pow(
            self.snr * torch.linalg.norm(predictor_noise) / torch.linalg.norm(predicted_score), 2.)
        return x + e * predicted_score + torch.sqrt(2. * e) * torch.randn_like(x)
