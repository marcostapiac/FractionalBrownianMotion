from typing import Union

import torch
import torchmetrics
from torchmetrics import MeanMetric

from src.generative_modelling.models import ClassVESDEDiffusion, ClassOUDiffusion, ClassVPSDEDiffusion
from src.generative_modelling.models.ClassOUDiffusion import OUDiffusion
from src.generative_modelling.models.ClassVESDEDiffusion import VESDEDiffusion
from src.generative_modelling.models.ClassVPSDEDiffusion import VPSDEDiffusion
from src.generative_modelling.models.TimeDependentScoreNetworks import ClassNaiveMLP
from src.generative_modelling.models.TimeDependentScoreNetworks import ClassTimeSeriesScoreMatching
from src.generative_modelling.models.TimeDependentScoreNetworks.ClassNaiveMLP import NaiveMLP
from src.generative_modelling.models.TimeDependentScoreNetworks.ClassTimeSeriesScoreMatching import \
    TimeSeriesScoreMatching


class DiffusionModelTrainer:
    """ Trainer class for a single GPU on a single machine, reporting aggregate loss over all batches """

    # Training of a model is a separate module from the optimiser specification and splitting of the data
    def __init__(self,
                 diffusion: Union[VESDEDiffusion, OUDiffusion, VPSDEDiffusion],
                 score_network: Union[NaiveMLP, TimeSeriesScoreMatching],
                 train_data_loader: torch.utils.data.dataloader.DataLoader,
                 train_eps:float,
                 end_diff_time:float,
                 max_diff_steps:int,
                 optimiser: torch.optim.Optimizer,
                 gpu_id: int,
                 checkpoint_freq: int,
                 loss_fn: callable = torch.nn.MSELoss,
                 loss_aggregator: torchmetrics.aggregation = MeanMetric):
        self.diffusion = diffusion
        self.train_eps = train_eps
        self.max_diff_steps = max_diff_steps
        self.end_diff_time = end_diff_time
        self.score_network = score_network
        self.train_loader = train_data_loader
        self.opt = optimiser
        self.gpu_id = gpu_id
        self.save_every = checkpoint_freq  # Specifies how often we choose to save our model during training
        self.loss_fn = loss_fn  # If callable, need to ensure we allow for gradient computation
        self.loss_aggregator = loss_aggregator()#.to(self.gpu_id)  # TODO: Do I need this to be on the same torch device?

    # Trainer should iterate through dataloader, and compute losses

    def _batch_update(self, loss) -> None:
        """
        Backward pass and optimiser update step
        :param loss: loss tensor / function output
        :return: None
        """
        loss.backward()  # single gpu functionality
        self.opt.step()
        # Detach returns the loss as a Tensor that does not require gradients, so you can manipulate it
        # independently of the original value, which does require gradients
        # Item is used to return a 1x1 tensor as a standard Python dtype (determined by Tensor dtype)
        self.loss_aggregator.update(loss.detach().item())

    def _batch_loss_compute(self, outputs: torch.Tensor, targets: torch.Tensor) -> None:
        """
        Computes loss and calls helper function to compute backward pass
        :param outputs: Model forward pass output
        :param targets: Target values to compare against outputs
        :return: None
        """
        self.opt.zero_grad()
        loss = self.loss_fn()(outputs, targets)
        self._batch_update(loss)

    def _run_epoch(self, epoch: int) -> None:
        # TODO: How to deal with more than one batch, how to deal with additional parameters required for training, how to deal with validation losses?
        b_sz = len(next(iter(self.train_loader))[0])
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_loader)} \n")
        self.score_network.train()
        timesteps = torch.linspace(self.train_eps, end=self.end_diff_time,
                                   steps=self.max_diff_steps)
        for x0s in iter(self.train_loader):
            x0s = x0s[0]#.to(self.gpu_id)  # TODO: why is this a list, where to send it to (why is it an int)?
            diff_times = timesteps[torch.randint(low=0, high=self.max_diff_steps, dtype=torch.int32,
                                                 size=(x0s.shape[0], 1))].view(x0s.shape[0],
                                                                               *([1] * len(x0s.shape[1:])))
            eff_times = self.diffusion.get_eff_times(diff_times)
            xts, targets = self.diffusion.noising_process(x0s, eff_times)
            outputs = self.score_network.forward(inputs=xts, times=diff_times.squeeze(-1)).squeeze(1)
            weights = self.diffusion.get_loss_weighting(eff_times)
            self._batch_loss_compute(outputs=weights * outputs, targets=weights * targets)

    def _save_checkpoint(self, epoch: int, filepath: str) -> None:
        """
        Save current state of model during training
        :param epoch: Current epoch number
        :param filepath: Filepath to save model
        :return: None
        """
        ckp = self.score_network.state_dict()
        torch.save(ckp, filepath)
        print(f"Epoch {epoch} | Training checkpoint saved at {filepath}")

    def train(self, max_epochs: int, model_filename: str) -> None:
        """
        Run training for model
        :param max_epochs: Total number of epochs
        :param model_filename: Filepath to save model
        :return: None
        """
        for epoch in range(max_epochs):
            self._run_epoch(epoch)
            print("Percent Completed {:0.4f} Train :: Val Losses, {:0.4f} :: TODO".format((epoch + 1) / max_epochs,
                                                                                          float(
                                                                                              self.loss_aggregator.compute().item())))
            if epoch % self.save_every == 0 or epoch + 1 == max_epochs:
                self._save_checkpoint(epoch=epoch, filepath=model_filename)
