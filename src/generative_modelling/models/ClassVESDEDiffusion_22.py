import numpy as np
import torch
from torch import nn
from torchmetrics import MeanMetric
from tqdm import tqdm


class VESDEDiffusion_2(nn.Module):

    def __init__(self, device: torch.cuda.Device, model, rng: np.random.Generator,
                 numDiffSteps: int, endDiffTime: float, trainEps: float, stdMax: float, stdMin: float, *args,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.torchDevice = device
        self.model = model.to(device)
        self.rng = rng
        self.trainEps = trainEps
        self.stdMax = stdMax
        self.stdMin = stdMin

        self.numDiffSteps = numDiffSteps
        self.endDiffTime = endDiffTime
        self.vars = self.get_vars()  # Increasing positive geometric sequence (for fwd process)

    def get_vars(self) -> torch.Tensor:
        """ This function specifies a variance schedule which fits best to the fBm data """
        var_max = self.get_var_max()
        var_min = self.get_var_min()
        indices = torch.linspace(start=0, end=self.numDiffSteps, steps=self.numDiffSteps)
        vars = var_min * torch.pow((var_max / var_min),
                                   indices / (self.numDiffSteps - 1))
        return vars

    def get_var_max(self) -> torch.Tensor:
        return torch.Tensor([(self.stdMax) ** 2]).to(torch.float32)

    def get_var_min(self) -> torch.Tensor:
        return torch.Tensor([(self.stdMin) ** 2]).to(torch.float32)

    def forward_process(self, dataSamples: torch.Tensor, effTimes: torch.Tensor):
        epsts = torch.randn_like(dataSamples)
        return dataSamples + torch.sqrt(effTimes) * epsts, -epsts / torch.sqrt(effTimes)

    def training_loss_fn(self, weighted_predicted: torch.Tensor, weighted_true: torch.Tensor) -> torch.Tensor:
        return nn.MSELoss()(weighted_predicted, weighted_true)

    def get_loss_weighting(self, effTimes: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(effTimes)

    def one_epoch_diffusion_train(self, opt: torch.optim.Optimizer,
                                  trainLoader: torch.utils.data.DataLoader) -> float:
        mean_loss = MeanMetric().to(device=self.torchDevice)
        self.train()
        # TODO: Parallelize
        timesteps = torch.linspace(self.trainEps, end=self.endDiffTime, steps=self.numDiffSteps)

        for x0s in iter(trainLoader):  # Iterate over batches (training data is already randomly selected)
            x0s = x0s[0]  # TODO: For some reason the original x0s is a list
            i_s = torch.randint(low=0, high=self.numDiffSteps, dtype=torch.int32,
                                size=(x0s.shape[0], 1))  # Randomly sample uniform time integer
            diffTimes = timesteps[i_s].view(x0s.shape[0], *([1] * len(x0s.shape[1:])))
            effTimes = (self.get_var_min() * (self.get_var_max() / self.get_var_min()) ** diffTimes)

            xts, true_score = self.forward_process(dataSamples=x0s, effTimes=effTimes)

            # A single batch of data should have dimensions [batch_size, channelNum, size_of_each_datapoint]
            pred = self.model.forward(xts, diffTimes.squeeze(-1))  # self.model.forward(xts, diffTimes)

            loss = self.training_loss_fn(
                weighted_predicted=self.get_loss_weighting(effTimes) * pred.squeeze(1),
                weighted_true=self.get_loss_weighting(effTimes) * true_score)

            opt.zero_grad()
            loss.backward()
            opt.step()
            mean_loss.update(loss.detach().item())  # Remove loss from computational graph and return

        return float(mean_loss.compute().item())

    def evaluate_diffusion_model(self, loader: torch.utils.data.DataLoader) -> float:
        mean_loss = MeanMetric().to(device=self.torchDevice)
        self.eval()
        timesteps = torch.linspace(self.trainEps, end=self.endDiffTime, steps=self.numDiffSteps)
        with torch.no_grad():
            for x0s in iter(loader):  # Iterate over batches (training data is already randomly selected)
                x0s = x0s[0]  # TODO: For some reason the original x0s is a list
                ts = torch.randint(low=0, high=self.numDiffSteps, dtype=torch.int32,
                                   size=(x0s.shape[0], 1))  # Randomly sample uniform time integer
                diffTimes = timesteps[ts].view(x0s.shape[0], *([1] * len(x0s.shape[1:])))
                effTimes = (self.get_var_min() * (self.get_var_max() / self.get_var_min()) ** diffTimes).view(
                    x0s.shape[0],
                    *([1] * len(
                        x0s.shape[
                        1:])))
                xts, true_score = self.forward_process(dataSamples=x0s, effTimes=effTimes)

                # A single batch of data should have dimensions [batch_size, channelNum, size_of_each_datapoint]
                pred = self.model.forward(xts, diffTimes.squeeze(-1))

                loss = self.training_loss_fn(
                    weighted_predicted=self.get_loss_weighting(effTimes) * pred.squeeze(1),
                    weighted_true=self.get_loss_weighting(effTimes) * true_score)

                mean_loss.update(loss.detach().item())
        return float(mean_loss.compute().item())

    def reverse_process(self, data: np.ndarray, dataSize: int, timeDim: int, sampleEps: float,
                        sigNoiseRatio: float, numLangevinSteps: int,
                        timeLim: int = 0):
        assert (sampleEps >= self.trainEps)

        # Initialise
        x = torch.sqrt(self.get_var_max()) * torch.randn((dataSize, timeDim), device=self.torchDevice)
        self.model.eval()
        reverseTimes = torch.linspace(start=self.endDiffTime, end=sampleEps, steps=self.numDiffSteps)
        with torch.no_grad():
            for i in tqdm(iterable=(range(0, self.numDiffSteps - timeLim)), dynamic_ncols=False,
                          desc="Sampling :: ", position=0):

                i_s = i * torch.ones((dataSize, 1), dtype=torch.long, device=self.torchDevice)
                ts = reverseTimes[i_s]  # time-index for each data-sample
                predicted_score = self.model.forward(x, ts.squeeze(-1)).squeeze(
                    1)  # Score == Noise/STD!-(x - data) / (effTimes)
                drift_var_param = self.vars[self.numDiffSteps - 1 - i] - (
                    self.vars[self.numDiffSteps - 1 - i - 1] if i < self.numDiffSteps - 1 else torch.Tensor([0]))
                noise_var_param = drift_var_param * self.vars[self.numDiffSteps - 1 - i - 1] / self.vars[
                    self.numDiffSteps - 1 - i] if i < self.numDiffSteps - 1 else torch.Tensor([0])
                z = torch.randn_like(x)
                x = x + drift_var_param * predicted_score + torch.sqrt(noise_var_param) * z
                for _ in range(numLangevinSteps):
                    z = torch.randn_like(x)
                    e = 2 * (sigNoiseRatio * np.linalg.norm(z) / np.linalg.norm(predicted_score)) ** 2
                    x = x + e * predicted_score + np.sqrt(2. * e) * z
            return x.detach().numpy()  # Approximately distributed according to desired distribution
