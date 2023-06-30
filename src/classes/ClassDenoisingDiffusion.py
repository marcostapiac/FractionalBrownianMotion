import numpy as np
import torch
from torch import nn
from torchmetrics import MeanMetric
from tqdm import tqdm

from src.classes import ClassTimeSeriesNoiseMatching


class DenoisingDiffusion(nn.Module):
    def __init__(self, device: torch.cuda.Device, model: ClassTimeSeriesNoiseMatching, N: int,
                 rng: np.random.Generator = np.random.default_rng(), *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.torchDevice = device
        self.model = model
        self.numDiffSteps = N

        self.betas = self.get_betas()
        self.alphas = 1. - self.betas
        self.alphaBars = torch.cumprod(self.alphas, dim=0)
        self.sqrtAlphaBars = torch.sqrt(self.alphaBars)
        self.sqrtAlphas = torch.sqrt(self.alphas)

        self.oneMinusAlphaBars = 1. - self.alphaBars
        self.sqrtOneMinusAlphaBars = torch.sqrt(self.oneMinusAlphaBars)
        # self.oneMinusPrevAlphaBars = 1. - F.pad(self.alphaBars[:-1], (1, 0), value=1.0) # No need if reverse_vars are simple

        self.postMeanCoeff1 = torch.pow(self.sqrtAlphas, -1)
        self.postMeanCoeff2 = self.betas * self.postMeanCoeff1 * torch.pow(self.sqrtOneMinusAlphaBars, -1)

        self.postCoeff1 = (1. / torch.sqrt(self.alphas))
        self.postCoeff2 = self.postCoeff1 * self.betas / self.sqrtOneMinusAlphaBars

        self.reverseVars = self.get_reverse_vars()
        self.lossWeightings = torch.sqrt(
            0.5 * torch.pow(self.betas, 2) * torch.pow(self.reverseVars * self.alphas * self.oneMinusAlphaBars, -1))

        self.rng = rng

    def forward_process(self, dataSamples: torch.Tensor, diffusionTimes: torch.Tensor, noise: torch.Tensor = None) -> [
        torch.Tensor, torch.Tensor]:
        x0s = dataSamples.to(torch.float32)
        epsts = torch.randn_like(x0s) if noise is None else noise
        usedSqrtAlphaBars = self.sqrtAlphaBars[diffusionTimes].view(dataSamples.shape[0],
                                                                    *([1] * len(dataSamples.shape[1:]))).to(
            torch.float32)
        usedOneMinusSqrtAlphaBars = self.sqrtOneMinusAlphaBars[diffusionTimes].view(dataSamples.shape[0], *(
                [1] * len(dataSamples.shape[1:]))).to(torch.float32)
        xts = usedSqrtAlphaBars * x0s + usedOneMinusSqrtAlphaBars * epsts
        return xts, epsts

    def get_betas(self) -> torch.Tensor:
        """ This function specifies a variance schedule which fits best to the fBm data """
        # TODO: How to set betas for fBm data
        scale = 1. / self.numDiffSteps
        beta_min = 0.1
        beta_max = 20.
        return torch.linspace(
            beta_min * scale,
            min(beta_max * scale, .5),
            steps=self.numDiffSteps,
            dtype=torch.float64,
            device=self.torchDevice,
            requires_grad=False)

    def get_reverse_vars(self):
        return self.betas  # * (self.oneMinusPrevAlphaBars / self.oneMinusAlphaBars)

    def get_loss_weighting(self, t: np.ndarray) -> torch.Tensor:
        return self.lossWeightings[t].to(torch.float32)

    @staticmethod
    def training_loss_fn(weighted_true: torch.Tensor, weighted_predicted: torch.Tensor):
        return nn.MSELoss()(weighted_predicted, weighted_true)

    def one_epoch_diffusion_train(self, opt: torch.optim.Optimizer,
                                  trainLoader: torch.utils.data.DataLoader) -> float:
        mean_loss = MeanMetric()
        self.train()
        for x0s in iter(trainLoader):  # Iterate over batches (training data is already randomly selected)
            x0s = x0s[0]  # TODO: For some reason the original x0s is a list
            ts = torch.randint(low=0, high=self.numDiffSteps, dtype=torch.int32,
                               size=(x0s.shape[0],))  # Randomly sample uniform time integer

            xts, true_noise = self.forward_process(dataSamples=x0s, diffusionTimes=ts)

            # Perturb target (https://arxiv.org/pdf/2301.11706.pdf) yts = xts + 0.1 * torch.diag(
            # diffusion.sqrtOneMinusAlphaBars[ts]).to(torch.float32) @ torch.randn_like(xts)

            # A single batch of data should have dimensions [batch_size, channelNum, size_of_each_datapoint]
            xts = torch.unsqueeze(xts, 1)
            pred = self.model.forward(xts, ts)
            loss = self.training_loss_fn(
                weighted_predicted=torch.diag(self.get_loss_weighting(ts.numpy())) @ pred.squeeze(1),
                weighted_true=torch.diag(self.get_loss_weighting(ts.numpy())) @ true_noise)
            opt.zero_grad()
            loss.backward()
            opt.step()
            mean_loss.update(loss.detach().item())
        return float(mean_loss.compute().item())

    def evaluate_diffusion_model(self, loader: torch.utils.data.DataLoader) -> float:
        mean_loss = MeanMetric()
        for x0s in (iter(loader)):
            self.eval()
            with torch.no_grad():
                x0s = x0s[0]
                ts = torch.randint(low=0, high=self.numDiffSteps, dtype=torch.int32,
                                   size=(x0s.shape[0],))  # Randomly sample uniform time integer
                xts, true_noise = self.forward_process(dataSamples=x0s, diffusionTimes=ts)

                # Perturb target (https://arxiv.org/pdf/2301.11706.pdf) yts = xts + 0.1 * torch.diag(
                # diffusion.sqrtOneMinusAlphaBars[ts]).to(torch.float32) @ torch.randn_like(xts)

                # A single batch of data should have dimensions [batch_size, channelNum, size_of_each_datapoint]
                xts = torch.unsqueeze(xts, 1)
                pred = self.model.forward(xts, ts)

                loss = self.training_loss_fn(
                    weighted_predicted=torch.diag(self.get_loss_weighting(ts.numpy())) @ pred.squeeze(1),
                    weighted_true=torch.diag(self.get_loss_weighting(ts.numpy())) @ true_noise)

                mean_loss.update(loss.detach().item())
        return float(mean_loss.compute().item())

    def reverse_process(self, dataSize: int, timeDim: int, timeLim: int = 0) -> np.ndarray:
        """ Reverse process for D datapoints of size T """
        x = torch.randn((dataSize, timeDim), device=self.torchDevice)
        if timeLim == self.numDiffSteps: return x.detach().numpy()
        x = np.divide(x, np.broadcast_to(np.amax(np.abs(x.numpy()), axis=0).reshape(1, timeDim), (dataSize, timeDim)))
        self.model.eval()
        with torch.no_grad():
            for t in tqdm(iterable=reversed(range(timeLim, self.numDiffSteps)), dynamic_ncols=False,
                          desc="Sampling :: ", position=0):
                ts = torch.ones(dataSize, dtype=torch.long,
                                device=self.torchDevice) * t  # time-index for each data-sample
                z = torch.randn_like(x) if t > 0 else torch.zeros_like(x)

                predicted_noise = self.model.forward(x.unsqueeze(1), ts)
                x = self.postCoeff1[t] * x - self.postCoeff2[t] * predicted_noise.squeeze(1) + torch.sqrt(
                    self.reverseVars[t]) * z

            return x.detach().numpy()  # Approximately distributed according to desired distribution
