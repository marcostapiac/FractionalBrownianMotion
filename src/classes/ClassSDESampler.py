from configs.project_config import NoneType
from typing import Union, Tuple

import torch
from tqdm import tqdm

from src.classes.ClassCorrector import Corrector
from src.classes.ClassPredictor import Predictor
from src.generative_modelling.models.ClassOUSDEDiffusion import OUSDEDiffusion
from src.generative_modelling.models.ClassVESDEDiffusion import VESDEDiffusion
from src.generative_modelling.models.ClassVPSDEDiffusion import VPSDEDiffusion


class SDESampler:
    """
    Reverse-time SDE sampler for diffusion models
    """

    def __init__(self, diffusion: Union[VESDEDiffusion, VPSDEDiffusion, OUSDEDiffusion], sample_eps: float,
                 predictor: Predictor, corrector: Union[Corrector, NoneType]):
        self.diffusion = diffusion
        self.predictor = predictor
        self.corrector = corrector
        self.sample_eps = sample_eps

    def sample(self, shape: Tuple[int, int], torch_device: Union[int, torch.device]) -> torch.Tensor:
        timesteps = torch.linspace(start=self.predictor.end_diff_time, end=self.sample_eps,
                                   steps=self.predictor.max_diff_steps)
        x = self.diffusion.prior_sampling(shape=shape).to(torch_device)  # Move to correct device
        for i in tqdm(iterable=(range(0, self.predictor.max_diff_steps)), dynamic_ncols=False, desc="Sampling :: ",
                      position=0):
            diff_index = torch.Tensor([i]).to(torch_device)
            x, pred_score, noise = self.predictor.step(x, t=timesteps[i] * torch.ones((x.shape[0], 1)).to(torch_device),
                                                       diff_index=diff_index)  # device = TODO
            if type(self.corrector) != NoneType:
                x = self.corrector.sample(x, pred_score, noise, diff_index, self.predictor.max_diff_steps)
        return x
