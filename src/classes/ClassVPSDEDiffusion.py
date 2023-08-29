import numpy as np
import torch
from torch import nn
from torchmetrics import MeanMetric
from tqdm import tqdm


class VPSDEDiffusion(nn.Module):
    def __init__(self, device: torch.cuda.Device, model, numDiffSteps: int, endDiffTime:float, trainEps: float, betaMax: float,betaMin: float,
                 rng: np.random.Generator = np.random.default_rng(), *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.torchDevice = device
        self.model = model
        self.numDiffSteps = numDiffSteps
        self.endDiffTime = endDiffTime
        self.trainEps = trainEps
        self.beta_max = betaMax
        self.beta_min = betaMin
        self.betas = self.get_betas()
        self.alphas = torch.cumprod(1. - self.betas, dim=0)

        self.rng = rng

    def get_betas(self):
        beta_min = self.get_beta_min()
        beta_max = self.get_beta_max()
        assert (beta_max < self.numDiffSteps)
        return torch.linspace(start=beta_min, end=beta_max, steps=self.numDiffSteps,
                              dtype=torch.float32) / self.numDiffSteps  # Discretised noise schedule

    def get_beta_min(self):
        return self.beta_min  # *1.8 for fBm, *1 for circle, *1.85 for fBn

    def get_beta_max(self):
        return self.beta_max

    def forward_process(self, dataSamples: torch.Tensor, effTimes: torch.Tensor) -> [torch.Tensor, torch.Tensor]:
        epsts = torch.randn_like(dataSamples)
        xts = torch.exp(-0.5 * effTimes) * dataSamples + torch.sqrt(1. - torch.exp(-effTimes)) * epsts
        return xts, -epsts / torch.sqrt((1. - torch.exp(-effTimes)))

    def get_loss_weighting(self, effTimes: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(1. - torch.exp(-effTimes))

    @staticmethod
    def training_loss_fn(weighted_true: torch.Tensor, weighted_predicted: torch.Tensor):
        return nn.MSELoss()(weighted_predicted, weighted_true)

    def one_epoch_diffusion_train(self, opt: torch.optim.Optimizer,
                                  trainLoader: torch.utils.data.DataLoader) -> float:
        mean_loss = MeanMetric()
        self.train()
        timesteps = torch.linspace(self.trainEps, end=self.endDiffTime, steps=self.numDiffSteps)
        for x0s in iter(trainLoader):  # Iterate over batches (training data is already randomly selected)
            x0s = x0s[0].to(self.torchDevice)  # TODO: For some reason the original x0s is a list
            # Randomly sample uniform time integer and each time-element in seq_length is diffused by different time
            i_s = torch.randint(low=0, high=self.numDiffSteps, dtype=torch.int32, size=(x0s.shape[0], 1), device=self.torchDevice)
            diffTimes = timesteps[i_s].view(x0s.shape[0], *([1] * len(x0s.shape[1:])))
            effTimes = (0.5 * diffTimes ** 2 * (self.get_beta_max() - self.get_beta_min()) + diffTimes * self.get_beta_min())

            xts, true_score = self.forward_process(dataSamples=x0s, effTimes=effTimes)
            pred = self.model.forward(xts, diffTimes.squeeze(-1))

            loss = self.training_loss_fn(
                weighted_predicted=self.get_loss_weighting(effTimes) * pred.squeeze(1),
                weighted_true=self.get_loss_weighting(effTimes) * true_score)
            opt.zero_grad()
            loss.backward()
            opt.step()
            mean_loss.update(loss.detach().item())
        return float(mean_loss.compute().item())

    def evaluate_diffusion_model(self, loader: torch.utils.data.DataLoader) -> float:
        mean_loss = MeanMetric()
        timesteps = torch.linspace(self.trainEps, end=self.endDiffTime, steps=self.numDiffSteps)
        for x0s in (iter(loader)):
            self.eval()
            with torch.no_grad():
                x0s = x0s[0].to(self.torchDevice)  # TODO: For some reason the original x0s is a list
                # Randomly sample uniform time integer and each time-element in seq_length is diffused by different time
                i_s = torch.randint(low=0, high=self.numDiffSteps, dtype=torch.int32, size=(x0s.shape[0], 1),
                                    device=self.torchDevice)
                diffTimes = timesteps[i_s]
                effTimes = (0.5 * diffTimes ** 2 * (self.get_beta_max() - self.get_beta_min()) + diffTimes * self.get_beta_min())

                xts, true_score = self.forward_process(dataSamples=x0s, effTimes=effTimes)
                pred = self.model.forward(xts, diffTimes.squeeze(-1))
                loss = self.training_loss_fn(
                    weighted_predicted=self.get_loss_weighting(effTimes) * pred.squeeze(1),
                    weighted_true=self.get_loss_weighting(effTimes) * true_score)

                mean_loss.update(loss.detach().item())
        return float(mean_loss.compute().item())

    def reverse_process(self, data: np.ndarray, dataSize: int, timeDim: int, sampleEps: float,
                        sigNoiseRatio: float, numLangevinSteps:int,
                        timeLim: int = 0) -> np.ndarray:
        """ Reverse process for D datapoints of size T """
        # Ensure we don't sample from times we haven't seen during training
        assert (sampleEps >= self.trainEps)

        # Initialise
        x = torch.randn((dataSize, timeDim), device=self.torchDevice)
        self.model.eval()
        reverseTimes = torch.linspace(start=self.endDiffTime, end=sampleEps,steps=self.numDiffSteps)
        with torch.no_grad():
            for i in tqdm(iterable=(range(timeLim, self.numDiffSteps)), dynamic_ncols=False,
                          desc="Sampling :: ", position=0):
                i_s = i * torch.ones((dataSize, 1), dtype=torch.long,
                                     device=self.torchDevice)
                ts = reverseTimes[i_s]  # time-index for each data-sample

                beta_t = self.betas[self.numDiffSteps - 1 - i]  # dt absorbed already

                predicted_score = self.model.forward(x, ts.squeeze(-1)).squeeze(1)  # Score == Noise/STD!
                z = torch.randn_like(x)
                x = x*(2.-torch.sqrt(1.-beta_t)) + beta_t*predicted_score + np.sqrt(beta_t) * z

                for _ in range(numLangevinSteps):
                    e = 2. * self.alphas[self.numDiffSteps - 1 - i] * (
                                sigNoiseRatio * np.linalg.norm(z) / np.linalg.norm(predicted_score)) ** 2
                    z = torch.randn_like(x)
                    x = x + e * predicted_score + np.sqrt(2. * e) * z
            return x.detach().numpy()  # Approximately distributed according to desired distribution
