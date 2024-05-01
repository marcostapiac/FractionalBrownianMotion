from typing import Union, Tuple

import torch
from tqdm import tqdm

from configs.project_config import NoneType
from src.classes.ClassCorrector import Corrector
from src.classes.ClassPredictor import ConditionalAncestralSamplingPredictor, \
    ConditionalReverseDiffusionSamplingPredictor, ConditionalLowVarReverseDiffusionSamplingPredictor, \
    ConditionalProbODESamplingPredictor, Predictor
from src.generative_modelling.models.ClassOUSDEDiffusion import OUSDEDiffusion
from src.generative_modelling.models.ClassVESDEDiffusion import VESDEDiffusion
from src.generative_modelling.models.ClassVPSDEDiffusion import VPSDEDiffusion


class ConditionalSDESampler:
    """
    Reverse-time SDE sampler for diffusion models
    """

    def __init__(self, diffusion: Union[VESDEDiffusion, VPSDEDiffusion, OUSDEDiffusion], sample_eps: float,
                 predictor: Predictor, corrector: Union[Corrector, NoneType]):
        self.diffusion = diffusion
        self.predictor = predictor
        self.corrector = corrector
        self.sample_eps = sample_eps

    def sample(self, shape: Tuple[int, int], feature: torch.Tensor, torch_device: Union[int, torch.device], ts_step:float, param_time:float,
               early_stop_idx: int = 0) -> Tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
        timesteps = torch.linspace(start=self.predictor.end_diff_time, end=self.sample_eps,
                                   steps=self.predictor.max_diff_steps)
        x = self.diffusion.prior_sampling(shape=shape).to(torch_device)  # Move to correct device
        x = x.unsqueeze(1)
        mean_est = None
        var_est = None
        for i in tqdm(iterable=(range(0, self.predictor.max_diff_steps - early_stop_idx)), dynamic_ncols=False,
                      desc="Sampling :: ",
                      position=0):
            diff_index = torch.Tensor([i]).to(torch_device)
            t = timesteps[i] * torch.ones((x.shape[0],)).to(torch_device)
            x, pred_score, noise, curr_mean, curr_var = self.predictor.step(x, t=t, diff_index=diff_index, feature=feature, ts_step=ts_step, param_est_time=param_time)
            if isinstance(curr_mean, torch.Tensor) and isinstance(curr_var, torch.Tensor):
                mean_est = curr_mean
                print(mean_est)
                var_est = curr_var
            if type(self.corrector) != NoneType:
                x = self.corrector.sample(x, pred_score, noise, diff_index, self.predictor.max_diff_steps)
        assert(mean_est.shape == shape and var_est.shape == shape)
        return x, mean_est, var_est
