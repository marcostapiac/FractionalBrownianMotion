import os
from typing import Union

import torch
import torchmetrics
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torchmetrics import MeanMetric

from src.evaluation_pipeline.classes.DiscriminativeLSTM.ClassDiscriminativeLSTM import DiscriminativeLSTM


# Link for DDP vs DataParallelism: https://www.run.ai/guides/multi-gpu/pytorch-multi-gpu-4-techniques-explained
# Link for ddp_setup backend: https://pytorch.org/docs/stable/distributed.html
# Tutorial: https://www.youtube.com/watch?v=-LAtx9Q6DA8


class DiscriminativeLSTMInference:
    def __init__(self,
                 model: DiscriminativeLSTM,
                 device: Union[torch.device, int],
                 loss_aggregator: torchmetrics.aggregation = MeanMetric):

        self.device_id = device
        assert (self.device_id == int(os.environ["LOCAL_RANK"]) or self.device_id == torch.device("cpu"))
        self.model = model.to(self.device_id)

        self.loss_aggregator = loss_aggregator().to(self.device_id)  # No need to move to device since they

        # Move model to appropriate device
        if type(self.device_id) == int:
            self.model = DDP(self.model, device_ids=[self.device_id])
        else:
            self.model = self.model.to(self.device_id)

    def _compute_metric(self, outputs: torch.Tensor, targets: torch.Tensor) -> None:
        """
        Computes accuracy (or loss iff input batch contains synthetic samples)
            :param outputs: Model forward pass output
            :param targets: Target values to compare against outputs
            :return: None
        """
        metric = torch.abs(targets - 1. * (outputs < 0.5))
        self.loss_aggregator.update(metric.detach())

    def run(self, test_loader: DataLoader) -> float:
        """
        Run inference for model and compute mean metric
            :param test_loader: DataLoader
            :return: mean metric with respect to targets
        """
        self.model.eval()
        with torch.no_grad():
            for X_batch, y_batch in iter(test_loader):
                X_batch = X_batch.to(self.device_id).to(torch.float32)
                y_batch = y_batch.to(self.device_id).to(torch.float32)
                outputs = self.model.forward(X_batch)
                assert (outputs.shape == y_batch.shape)
                self._compute_metric(outputs, y_batch)
        return float(self.loss_aggregator.compute().item())
