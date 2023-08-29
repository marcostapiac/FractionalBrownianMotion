import numpy as np
import torch
from torch import nn
from torchmetrics import MeanMetric
from tqdm import tqdm


class OUDiffusion(nn.Module):
    def __init__(self, device: torch.cuda.Device, model, N: int, trainEps: float,
                 rng: np.random.Generator = np.random.default_rng(), Tdiff: float = 1., *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.torchDevice = device
        self.model = model
        self.numDiffSteps = N
        self.trainEps = trainEps
        self.endDiffTime = Tdiff

        self.rng = rng

    def forward_process(self, dataSamples: torch.Tensor, diffusionTimes: torch.Tensor) -> [
        torch.Tensor, torch.Tensor]:
        x0s = dataSamples.to(torch.float32)
        epsts = torch.randn_like(x0s)
        xts = torch.exp(-0.5 * diffusionTimes) * x0s + torch.sqrt((1. - torch.exp(-diffusionTimes))) * epsts
        return xts, -epsts / torch.sqrt((1. - torch.exp(-diffusionTimes)))

    def get_loss_weighting(self, diffTimes: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(1. - torch.exp(-diffTimes))

    @staticmethod
    def training_loss_fn(weighted_true: torch.Tensor, weighted_predicted: torch.Tensor):
        return nn.MSELoss()(weighted_predicted, weighted_true)

    def one_epoch_diffusion_train(self, opt: torch.optim.Optimizer,
                                  trainLoader: torch.utils.data.DataLoader) -> float:
        mean_loss = MeanMetric()
        self.train()
        timesteps = torch.linspace(self.trainEps, end=self.endDiffTime, steps=self.numDiffSteps)
        for x0s in iter(trainLoader):  # Iterate over batches (training data is already randomly selected)
            x0s = x0s[0]  # TODO: For some reason the original x0s is a list
            ts = torch.randint(low=0, high=self.numDiffSteps, dtype=torch.int32,
                               size=(x0s.shape[0],))  # Randomly sample uniform time integer
            diffTimes = timesteps[ts].view(x0s.shape[0], *([1] * len(x0s.shape[1:])))
            xts, true_score = self.forward_process(dataSamples=x0s, diffusionTimes=diffTimes)
            pred = self.model.forward(xts, diffTimes.squeeze(-1))
            loss = self.training_loss_fn(
                weighted_predicted=self.get_loss_weighting(diffTimes) * pred.squeeze(1),
                weighted_true=self.get_loss_weighting(diffTimes) * true_score)
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
                x0s = x0s[0]  # TODO: For some reason the original x0s is a list
                ts = torch.randint(low=0, high=self.numDiffSteps, dtype=torch.int32,
                                   size=(x0s.shape[0],))  # Randomly sample uniform time integer

                diffTimes = timesteps[ts].view(x0s.shape[0], *([1] * len(x0s.shape[1:])))
                xts, true_score = self.forward_process(dataSamples=x0s, diffusionTimes=diffTimes)
                pred = self.model.forward(xts, diffTimes.squeeze(-1))
                loss = self.training_loss_fn(
                    weighted_predicted=self.get_loss_weighting(diffTimes) * pred.squeeze(1),
                    weighted_true=self.get_loss_weighting(diffTimes) * true_score)

                mean_loss.update(loss.detach().item())
        return float(mean_loss.compute().item())

    def reverse_process(self, dataSize: int, timeDim: int, sampleEps: float, data: np.ndarray,
                        timeLim: int = 0) -> np.ndarray:
        """ Reverse process for D datapoints of size T """
        # Ensure we don't sample from times we haven't seen during training
        assert (sampleEps >= self.trainEps)
        x = torch.randn((dataSize, timeDim), device=self.torchDevice).to(torch.float32)
        self.model.eval()
        dt = 1. / self.numDiffSteps
        reverseTimes = torch.linspace(start=self.endDiffTime, end=sampleEps, steps=self.numDiffSteps)
        with torch.no_grad():
            for i in tqdm(iterable=(range(timeLim, self.numDiffSteps)), dynamic_ncols=False,
                          desc="Sampling :: ", position=0):
                ts = reverseTimes[i] * torch.ones((dataSize, 1), dtype=torch.long,
                                                  device=self.torchDevice)  # time-index for each data-sample
                predicted_score = self.model.forward(x, ts.squeeze(-1)).squeeze(1)  # Score == Noise/STD!
                z = torch.randn_like(x)
                x = x + (0.5 * x + predicted_score) * dt + np.sqrt(dt) * z

            return x.detach().numpy()  # Approximately distributed according to desired distribution

    def conditional_reverse_process(self, observations: torch.Tensor, latent: torch.Tensor, dataSize: int, timeDim: int,
                                    reverseTimes: torch.Tensor, timeLim: int = 0) -> np.ndarray:
        """ Reverse process for D datapoints of size T """
        # Ensure we don't sample from times we haven't seen during training
        assert (reverseTimes[-1] >= self.trainEps and reverseTimes.shape[0] == self.numDiffSteps)
        x = torch.randn((dataSize, timeDim), device=self.torchDevice)
        self.model.eval()
        dt = 1. / self.numDiffSteps
        with torch.no_grad():
            for i in tqdm(iterable=(range(timeLim, self.numDiffSteps)), dynamic_ncols=False,
                          desc="Sampling :: ", position=0):
                ts = reverseTimes[i] * torch.ones((dataSize, 1), dtype=torch.long,
                                                  device=self.torchDevice)  # time-index for each data-sample
                predicted_score = self.model.forward(x.unsqueeze(1), ts).squeeze(1)
                x0_hat = torch.exp(0.5 * ts) * x - np.sqrt(torch.exp(ts) - 1.) * torch.randn_like(x)
                observed_score = torch.exp(0.5 * ts) * (observations - (x0_hat + 6.)) / (
                    0.01)  # grad log p(y0|xt)
                score = predicted_score + observed_score
                z = torch.randn_like(x)
                x = x + (0.5 * x + score) * dt + np.sqrt(dt) * z

            return x.detach().numpy()  # Approximately distributed according to desired distribution
