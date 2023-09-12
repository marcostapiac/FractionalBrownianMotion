import os
from typing import Union

import torch
import torchmetrics
from torchmetrics import MeanMetric

from src.generative_modelling.models.ClassOUSDEDiffusion import OUSDEDiffusion
from src.generative_modelling.models.ClassVESDEDiffusion import VESDEDiffusion
from src.generative_modelling.models.ClassVPSDEDiffusion import VPSDEDiffusion
from src.generative_modelling.models.TimeDependentScoreNetworks.ClassNaiveMLP import NaiveMLP
from src.generative_modelling.models.TimeDependentScoreNetworks.ClassTimeSeriesScoreMatching import \
    TimeSeriesScoreMatching


# Link for DDP vs DataParallelism: https://www.run.ai/guides/multi-gpu/pytorch-multi-gpu-4-techniques-explained
# Link for ddp_setup backend: https://pytorch.org/docs/stable/distributed.html
# Tutorial: https://www.youtube.com/watch?v=-LAtx9Q6DA8


class DiffusionModelTrainer:
    """ Trainer class for a single GPU on a single machine, reporting aggregate loss over all batches """

    def __init__(self,
                 diffusion: Union[VESDEDiffusion, OUSDEDiffusion, VPSDEDiffusion],
                 score_network: Union[NaiveMLP, TimeSeriesScoreMatching],
                 train_data_loader: torch.utils.data.dataloader.DataLoader,
                 train_eps: float,
                 end_diff_time: float,
                 max_diff_steps: int,
                 optimiser: torch.optim.Optimizer,
                 snapshot_path: str,
                 rank: int,
                 checkpoint_freq: int,
                 loss_fn: callable = torch.nn.MSELoss,
                 loss_aggregator: torchmetrics.aggregation = MeanMetric):

        self.gpu_id = rank  # int(os.environ["LOCAL_RANK"])  # TODO: Device or rank?
        self.score_network = score_network
        self.epochs_run = 0

        self.opt = optimiser
        self.save_every = checkpoint_freq  # Specifies how often we choose to save our model during training
        self.train_loader = train_data_loader
        self.loss_fn = loss_fn  # If callable, need to ensure we allow for gradient computation
        self.loss_aggregator = loss_aggregator()  # No need to move to device since they

        self.diffusion = diffusion
        self.train_eps = train_eps
        self.max_diff_steps = max_diff_steps
        self.end_diff_time = end_diff_time

        self.snapshot_path = snapshot_path
        # Load snapshot if available
        if os.path.exists(self.snapshot_path):
            print("Loading snapshot")
            self._load_snapshot(self.snapshot_path)

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
        loss = self.loss_fn()(outputs, targets)
        self._batch_update(loss)

    def _run_batch(self, xts: torch.Tensor, target_scores: torch.Tensor, diff_times: torch.Tensor,
                   eff_times: torch.Tensor) -> None:
        """
        Compute batch output and loss
            :param xts: Diffused samples
            :param target_scores: Target scores at corresponding diff times
            :param diff_times: Diffusion times
            :param eff_times: Effective diffusion times
            :return: None
        """
        self.opt.zero_grad()
        outputs = self.score_network.forward(inputs=xts, times=diff_times.squeeze(-1)).squeeze(1)
        weights = self.diffusion.get_loss_weighting(eff_times=eff_times)
        self._batch_loss_compute(outputs=weights * outputs, targets=weights * target_scores)

    def _run_epoch(self, epoch: int) -> None:
        """
        Single epoch run
            :param epoch: Epoch index
            :return: None
        """
        # TODO: How to deal with validation losses?
        b_sz = len(next(iter(self.train_loader))[0])
        print(
            f"[Device {self.gpu_id}] Epoch {epoch + 1} | Batchsize: {b_sz} | Total Num of Batches: {len(self.train_loader)} \n")
        self.score_network.train()
        timesteps = torch.linspace(self.train_eps, end=self.end_diff_time,
                                   steps=self.max_diff_steps)
        for x0s in iter(self.train_loader):
            x0s = x0s[0].to(self.gpu_id)
            diff_times = timesteps[torch.randint(low=0, high=self.max_diff_steps, dtype=torch.int32,
                                                 size=(x0s.shape[0], 1))].view(x0s.shape[0],
                                                                               *([1] * len(x0s.shape[1:]))).to(
                self.gpu_id)
            eff_times = self.diffusion.get_eff_times(diff_times)
            xts, target_scores = self.diffusion.noising_process(x0s, eff_times)
            self._run_batch(xts=xts, target_scores=target_scores, diff_times=diff_times, eff_times=eff_times)

    def _load_snapshot(self, snapshot_path: str) -> None:
        """
        Load training from most recent snapshot
            :param snapshot_path: Path to training snapshot
            :return: None
        """
        # Snapshot should be python dict
        snapshot = torch.load(snapshot_path)
        self.score_network.load_state_dict(snapshot["MODEL_STATE"])
        self.epochs_run = snapshot["EPOCHS_RUN"]
        print("Resuming training from snapshot at epoch {}".format(self.epochs_run + 1))

    def _save_snapshot(self, epoch: int) -> None:
        """
        Save current state of training
            :param epoch: Current epoch number
            :return: None
        """
        snapshot = {}
        # self.score_network now points to DDP wrapped object, so we need to access parameters via ".module"
        snapshot["MODEL_STATE"] = self.score_network.module.state_dict()
        snapshot["EPOCHS_RUN"] = epoch
        torch.save(snapshot, self.snapshot_path)
        print(f"Epoch {epoch + 1} | Training snapshot saved at {self.snapshot_path}")

    def _save_model(self, filepath: str) -> None:
        """
        Save final trained model
            :param filepath: Filepath to save model
            :return: None
        """
        # self.score_network now points to DDP wrapped object so we need to access parameters via ".module"
        ckp = self.score_network.module.state_dict()
        torch.save(ckp, filepath)
        print(f"Trained model saved at {filepath}")
        os.remove(self.snapshot_path)

    def train(self, max_epochs: int, model_filename: str) -> None:
        """
        Run training for model
            :param max_epochs: Total number of epochs
            :param model_filename: Filepath to save model
            :return: None
        """
        for epoch in range(self.epochs_run, max_epochs):
            self._run_epoch(epoch)
            print("Percent Completed {:0.4f} :: Train {:0.4f} ".format((epoch + 1) / max_epochs,
                                                                       float(self.loss_aggregator.compute().item())))
            if (self.gpu_id == 0 or type(self.gpu_id) == torch.device) and epoch + 1 == max_epochs:
                self._save_model(filepath=model_filename)
            elif (self.gpu_id == 0 or type(self.gpu_id) == torch.device) and ((epoch + 1) % self.save_every == 0):
                self._save_snapshot(epoch=epoch)
