from configs.project_config import NoneType
from typing import Union, Tuple

import torch
from tqdm import tqdm

from src.classes.ClassCorrector import Corrector
from src.classes.ClassPredictor import Predictor, ConditionalAncestralSamplingPredictor
from src.generative_modelling.models.ClassOUSDEDiffusion import OUSDEDiffusion
from src.generative_modelling.models.ClassVESDEDiffusion import VESDEDiffusion
from src.generative_modelling.models.ClassVPSDEDiffusion import VPSDEDiffusion


class ConditionalSDESampler:
    """
    Reverse-time SDE sampler for diffusion models
    """

    def __init__(self, diffusion: Union[VESDEDiffusion, VPSDEDiffusion, OUSDEDiffusion], sample_eps: float,
                 predictor: ConditionalAncestralSamplingPredictor, corrector: Union[Corrector, NoneType]):
        self.diffusion = diffusion
        self.predictor = predictor
        self.corrector = corrector
        self.sample_eps = sample_eps

    def sample(self, shape: Tuple[int, int], feature:torch.Tensor, torch_device: Union[int, torch.device], early_stop_idx:int=0) -> torch.Tensor:
        timesteps = torch.linspace(start=self.predictor.end_diff_time, end=self.sample_eps,
                                   steps=self.predictor.max_diff_steps)
        x = self.diffusion.prior_sampling(shape=shape).to(torch_device)  # Move to correct device
        x = x.unsqueeze(1)
        for i in tqdm(iterable=(range(0, self.predictor.max_diff_steps-early_stop_idx)), dynamic_ncols=False, desc="Sampling :: ",
                      position=0):
            diff_index = torch.Tensor([i]).to(torch_device)
            t = timesteps[i] * torch.ones((x.shape[0], )).to(torch_device)
            x, pred_score, noise = self.predictor.step(x, t=t, diff_index=diff_index, feature=feature)  # device = TODO

            if type(self.corrector) != NoneType:
                x = self.corrector.sample(x, pred_score, noise, diff_index, self.predictor.max_diff_steps)
        return x