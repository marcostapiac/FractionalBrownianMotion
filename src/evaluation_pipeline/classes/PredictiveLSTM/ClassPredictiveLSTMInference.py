import os
from typing import Union

import torch
import torchmetrics
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torchmetrics import MeanMetric

from src.evaluation_pipeline.classes.PredictiveLSTM.ClassPredictiveLSTM import PredictiveLSTM


# Link for DDP vs DataParallelism: https://www.run.ai/guides/multi-gpu/pytorch-multi-gpu-4-techniques-explained
# Link for ddp_setup backend: https://pytorch.org/docs/stable/distributed.html
# Tutorial: https://www.youtube.com/watch?v=-LAtx9Q6DA8


class PredictiveLSTMInference:
    def __init__(self,
                 model: PredictiveLSTM,
                 device: Union[torch.device, int],
                 loss_fn: callable,
                 loss_aggregator: torchmetrics.aggregation = MeanMetric):

        self.device_id = device
        assert (self.device_id == int(os.environ["LOCAL_RANK"]) or self.device_id == torch.device("cpu"))
        self.model = model.to(self.device_id)

        self.loss_fn = loss_fn  # If callable, need to ensure we allow for gradient computation
        self.loss_aggregator = loss_aggregator().to(self.device_id)  # No need to move to device since they

        # Move model to appropriate device
        if type(self.device_id) == int:
            self.model = DDP(self.model, device_ids=[self.device_id])
        else:
            self.model = self.model.to(self.device_id)

    @staticmethod
    def _compute_scale_normaliser(targets: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.abs(targets[:, 1, :] - targets[:, 0, :]))

    def _compute_loss(self, outputs: torch.Tensor, targets: torch.Tensor) -> None:
        """
        Computes loss
            :param outputs: Model forward pass output
            :param targets: Target values to compare against outputs
            :return: None
        """
        loss = self.loss_fn()(outputs, targets)
        self.loss_aggregator.update(loss.detach().item())

    def run(self, test_loader: DataLoader) -> float:
        """
        Run inference for model and compute loss
            :param test_loader: DataLoader
            :return: losses with respect to targets
        """
        self.model.eval()
        with torch.no_grad():
            for X_batch, y_batch in iter(test_loader):
                X_batch = X_batch.to(self.device_id).to(torch.float32)
                y_batch = y_batch[:, -1, :].to(self.device_id).to(torch.float32)
                outputs = self.model.forward(X_batch)
                assert (outputs.shape == y_batch.shape)
                self._compute_loss(outputs, y_batch)
        return float(self.loss_aggregator.compute().item())
