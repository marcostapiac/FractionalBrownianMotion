import os
import time
from typing import Union

import torch
import torchmetrics
from torch.nn.parallel import DistributedDataParallel as DDP
from torchmetrics import MeanMetric

from src.evaluation_pipeline.classes.PredictiveLSTM.ClassPredictiveLSTM import PredictiveLSTM


# Link for DDP vs DataParallelism: https://www.run.ai/guides/multi-gpu/pytorch-multi-gpu-4-techniques-explained
# Link for ddp_setup backend: https://pytorch.org/docs/stable/distributed.html
# Tutorial: https://www.youtube.com/watch?v=-LAtx9Q6DA8


class PredictiveLSTMTrainer:
    def __init__(self,
                 model: PredictiveLSTM,
                 train_data_loader: torch.utils.data.dataloader.DataLoader,
                 optimiser: torch.optim.Optimizer,
                 snapshot_path: str,
                 device: Union[torch.device, int],
                 checkpoint_freq: int,
                 loss_fn: callable,
                 loss_aggregator: torchmetrics.aggregation = MeanMetric):

        self.device_id = device
        assert (self.device_id == int(os.environ["LOCAL_RANK"]) or self.device_id == torch.device("cpu"))
        self.model = model
        self.epochs_run = 0

        self.opt = optimiser
        self.save_every = checkpoint_freq  # Specifies how often we choose to save our model during training
        self.train_loader = train_data_loader
        self.loss_fn = loss_fn  # If callable, need to ensure we allow for gradient computation
        self.loss_aggregator = loss_aggregator().to(self.device_id)  # No need to move to device since they

        # Move model to appropriate device
        if type(self.device_id) == int:
            self.model = DDP(self.model.to(self.device_id), device_ids=[self.device_id])
        else:
            self.model = self.model.to(self.device_id)

        self.snapshot_path = snapshot_path
        # Load snapshot if available
        if os.path.exists(self.snapshot_path):
            print("Device {} :: Loading snapshot\n".format(self.device_id))
            self._load_snapshot(self.snapshot_path)

    def _batch_update(self, loss) -> None:
        """
        Backward pass and optimiser update step
            :param loss: loss tensor / function output
            :return: None
        """
        loss.backward()  # single gpu functionality
        self.opt.step()
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

    def _run_batch(self, input_batch: torch.Tensor, targets: torch.Tensor) -> None:
        """
        Compute batch output and loss
            :param input_batch: Input batch of time-series
            :param targets: 1 step prediction targets for each batch element (time-series)
            :return: None
        """
        self.opt.zero_grad()
        outputs = self.model.forward(input=input_batch)
        self._batch_loss_compute(outputs=outputs, targets=targets[:, -1, :])

    def _run_epoch(self, epoch: int) -> None:
        """
        Single epoch run
            :param epoch: Epoch index
            :return: None
        """
        # TODO: How to deal with validation losses?
        b_sz = len(next(iter(self.train_loader))[0])
        print(
            f"[Device {self.device_id}] Epoch {epoch + 1} | Batchsize: {b_sz} | Total Num of Batches: {len(self.train_loader)} \n")
        if type(self.device_id) != torch.device: self.train_loader.sampler.set_epoch(epoch)
        for X_batch, y_batch in iter(self.train_loader):
            X_batch = X_batch.to(self.device_id).to(torch.float32)
            y_batch = y_batch.to(self.device_id).to(torch.float32)
            self._run_batch(input_batch=X_batch, targets=y_batch)

    def _load_snapshot(self, snapshot_path: str) -> None:
        """
        Load training from most recent snapshot
            :param snapshot_path: Path to training snapshot
            :return: None
        """
        # Snapshot should be python dict
        loc = 'cuda:{}'.format(self.device_id) if type(self.device_id) == int else self.device_id
        snapshot = torch.load(snapshot_path, map_location=loc)
        self.epochs_run = snapshot["EPOCHS_RUN"]
        self.opt.load_state_dict(snapshot["OPTIMISER_STATE"])
        if type(self.device_id) == int:
            self.model.module.load_state_dict(snapshot["MODEL_STATE"])
        else:
            self.model.load_state_dict(snapshot["MODEL_STATE"])
        print("Resuming training from snapshot at epoch {}".format(self.epochs_run + 1))

    def _save_snapshot(self, epoch: int) -> None:
        """
        Save current state of training
            :param epoch: Current epoch number
            :return: None
        """
        snapshot = {"EPOCHS_RUN": epoch, "OPTIMISER_STATE": self.opt.state_dict()}
        # self.model now points to DDP wrapped object, so we need to access parameters via ".module"
        if type(self.device_id) == int:
            snapshot["MODEL_STATE"] = self.model.module.state_dict()
        else:
            snapshot["MODEL_STATE"] = self.model.state_dict()
        torch.save(snapshot, self.snapshot_path)
        print(f"Epoch {epoch + 1} | Training snapshot saved at {self.snapshot_path}")

    def _save_model(self, filepath: str, final_epoch: int) -> None:
        """
        Save final trained model
            :param filepath: Filepath to save model
            :param final_epoch: Final training epoch
            :return: None
        """
        # self.model now points to DDP wrapped object so we need to access parameters via ".module"
        if type(self.device_id) == int:
            ckp = self.model.to(torch.device("cpu")).module.state_dict()  # Save model on CPU
        else:
            ckp = self.model.to(torch.device("cpu")).state_dict()  # Save model on CPU
        filepath = filepath + "_NEp{}".format(final_epoch)
        torch.save(ckp, filepath)
        print(f"Trained model saved at {filepath}\n")
        try:
            pass
            # os.remove(self.snapshot_path)  # Do NOT remove snapshot path yet eventhough training is done
        except FileNotFoundError:
            print("Snapshot file does not exist\n")

    def train(self, max_epochs: int, model_filename: str) -> None:
        """
        Run training for model
            :param max_epochs: Total number of epochs
            :param model_filename: Filepath to save model
            :return: None
        """
        self.model.train()
        for epoch in range(self.epochs_run, max_epochs):
            t0 = time.time()
            self._run_epoch(epoch)
            # NOTE: .compute() cannot be called on only one process since it will wait for other processes
            # see  https://github.com/Lightning-AI/torchmetrics/issues/626
            print("Device {} :: Percent Completed {:0.4f} :: Train {:0.4f} :: Time for One Epoch {:0.4f}\n".format(
                self.device_id, (epoch + 1) / max_epochs,
                float(
                    self.loss_aggregator.compute().item()), float(time.time() - t0)))
            if self.device_id == 0 or type(self.device_id) == torch.device:
                if epoch + 1 == max_epochs:
                    self._save_model(filepath=model_filename, final_epoch=epoch + 1)
                elif (epoch + 1) % self.save_every == 0:
                    self._save_snapshot(epoch=epoch)
