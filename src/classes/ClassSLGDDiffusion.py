import numpy as np
import torch
from torch import nn
from torchmetrics import MeanMetric
from tqdm import tqdm

from src.classes import ClassTimeSeriesNoiseMatching


class SLGDDiffusion(nn.Module):

    def __init__(self, device: torch.cuda.Device, model: ClassTimeSeriesNoiseMatching, rng: np.random.Generator,
                 numDiffSteps: int,
                 numLangevinSteps: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.torchDevice = device
        self.model = model.to(device)
        self.rng = rng

        self.numDiffSteps = numDiffSteps
        self.numLangevinSteps = numLangevinSteps
        self.epsilon = 2e-5
        self.vars = self.get_vars()  # Decreasing positive geometric sequence
        self.alphas = self.get_alphas()

    def get_vars(self) -> torch.Tensor:
        """ This function specifies a variance schedule which fits best to the fBm data """
        sigma_start = 1.
        sigma_end = 0.01
        return torch.pow(torch.exp(
            torch.linspace(np.log(sigma_start), np.log(sigma_end), self.numDiffSteps, dtype=torch.float64,
                           device=self.torchDevice, requires_grad=False)), 2.)

    def get_alphas(self) -> torch.Tensor:
        return self.epsilon * self.vars / self.vars[-1]

    @staticmethod
    def forward_process(dataSamples: torch.Tensor, std: torch.Tensor):
        return dataSamples + std * torch.randn_like(dataSamples)

    def training_loss_fn(self, predicted_score: torch.Tensor, target_score: torch.Tensor, used_vars: torch.Tensor):
        target_score = target_score.view(target_score.shape[0], -1)
        predicted_score = predicted_score.view(predicted_score.shape[0], -1)
        return (0.5 * ((predicted_score - target_score) ** 2).sum(dim=-1) * used_vars.squeeze()).mean(dim=0)

    def one_epoch_diffusion_train(self, opt: torch.optim.Optimizer,
                                  trainLoader: torch.utils.data.DataLoader) -> float:
        mean_loss = MeanMetric().to(device=self.torchDevice)
        self.train()
        # TODO: Parallelize
        for x0s in iter(trainLoader):  # Iterate over batches (training data is already randomly selected)
            x0s = x0s[0].to(self.torchDevice)  # TODO: For some reason the original x0s is a list
            varIndices = torch.randint(0, self.numDiffSteps, (x0s.shape[0],), device=self.torchDevice).to(
                self.torchDevice)
            vars = self.vars[varIndices].view(x0s.shape[0], *([1] * len(x0s.shape[1:]))).to(self.torchDevice)

            # A single batch of data should have dimensions [batch_size, channelNum, size_of_each_datapoint]
            perturbed = self.forward_process(dataSamples=x0s, std=torch.sqrt(vars)).to(torch.float32).to(
                self.torchDevice)
            perturbed = torch.unsqueeze(perturbed, 1)

            predicted_score = self.model.forward(perturbed, varIndices).squeeze(1).to(self.torchDevice)
            target_score = (- (perturbed.squeeze(1) - x0s) / vars).to(self.torchDevice)
            assert (target_score.shape == predicted_score.shape)

            loss = self.training_loss_fn(predicted_score=predicted_score, target_score=target_score, used_vars=vars)

            opt.zero_grad()
            loss.backward()
            opt.step()
            mean_loss.update(loss.detach().item())  # Remove loss from computational graph and return

        return float(mean_loss.compute().item())

    def evaluate_diffusion_model(self, loader: torch.utils.data.DataLoader) -> float:
        mean_loss = MeanMetric().to(device=self.torchDevice)
        self.eval()
        with torch.no_grad():
            for x0s in iter(loader):  # Iterate over batches (training data is already randomly selected)
                x0s = x0s[0].to(self.torchDevice)  # TODO: For some reason the original x0s is a list
                varIndices = (torch.randint(0, self.numDiffSteps, (x0s.shape[0],), device=self.torchDevice)).to(
                    self.torchDevice)
                vars = self.vars[varIndices].view(x0s.shape[0], *([1] * len(x0s.shape[1:]))).to(self.torchDevice)

                perturbed = self.forward_process(dataSamples=x0s, std=torch.sqrt(vars)).to(torch.float32).to(
                    self.torchDevice)
                # A single batch of data should have dimensions [batch_size, channelNum, size_of_each_datapoint]
                perturbed = torch.unsqueeze(perturbed, 1)
                predicted_score = self.model.forward(perturbed, varIndices).squeeze(1).to(self.torchDevice)

                target_score = (- (perturbed.squeeze(1) - x0s) / vars).to(self.torchDevice)
                assert (target_score.shape == predicted_score.shape)
                loss = self.training_loss_fn(predicted_score=predicted_score, target_score=target_score, used_vars=vars)

                mean_loss.update(loss.detach().item())
        return float(mean_loss.compute().item())

    def reverse_process(self, dataSize: int, timeDim: int, timeLim: int = 0):
        x = torch.randn((dataSize, timeDim), device=self.torchDevice)
        x = np.divide(x, np.broadcast_to(np.amax(np.abs(x.numpy()), axis=0).reshape(1, timeDim), (
            dataSize, timeDim)))  # TODO: Normalise column wise (feature) or row-wise (time-series wise)?

        self.eval()
        with torch.no_grad():
            for i in tqdm(iterable=reversed(range(timeLim, self.numDiffSteps)), dynamic_ncols=False,
                          desc="Sampling :: ", position=0):
                y = x
                alpha = self.alphas[i]
                varIndices = torch.from_numpy(np.array(i)).to(torch.int32)
                for _ in (range(0, self.numLangevinSteps)):
                    z = torch.randn_like(y)
                    predicted_score = self.model.forward(y.unsqueeze(1), varIndices).squeeze(1)
                    y = y + 0.5 * alpha * predicted_score + torch.sqrt(alpha) * z
                x = y
            return x.detach().numpy()  # Approximately distributed according to desired distribution
