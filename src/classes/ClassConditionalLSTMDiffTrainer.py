import os
import pickle
import time
from typing import Union

import torch
import torch.distributed as dist
import torchmetrics
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torchmetrics import MeanMetric
import numpy as np
from src.generative_modelling.models.ClassOUSDEDiffusion import OUSDEDiffusion
from src.generative_modelling.models.ClassVESDEDiffusion import VESDEDiffusion
from src.generative_modelling.models.ClassVPSDEDiffusion import VPSDEDiffusion
from src.generative_modelling.models.TimeDependentScoreNetworks.ClassConditionalLSTMTimeSeriesScoreMatching import \
    ConditionalLSTMTimeSeriesScoreMatching


# Link for DDP vs DataParallelism: https://www.run.ai/guides/multi-gpu/pytorch-multi-gpu-4-techniques-explained
# Link for ddp_setup backend: https://pytorch.org/docs/stable/distributed.html
# Tutorial: https://www.youtube.com/watch?v=-LAtx9Q6DA8


class ConditionalLSTMDiffusionModelTrainer(nn.Module):

    def __init__(self,
                 diffusion: Union[VESDEDiffusion, OUSDEDiffusion, VPSDEDiffusion],
                 score_network: Union[
                     ConditionalLSTMTimeSeriesScoreMatching],
                 train_data_loader: torch.utils.data.dataloader.DataLoader,
                 train_eps: float,
                 end_diff_time: float,
                 max_diff_steps: int,
                 optimiser: torch.optim.Optimizer,
                 snapshot_path: str,
                 device: Union[torch.device, int],
                 checkpoint_freq: int,
                 to_weight: bool,
                 hybrid_training: bool,
                 loss_fn: callable = torch.nn.MSELoss,
                 loss_aggregator: torchmetrics.aggregation = MeanMetric):
        super().__init__()
        self.device_id = device
        assert (self.device_id == torch.device("cpu") or self.device_id == int(os.environ["LOCAL_RANK"]))
        self.score_network = score_network
        self.epochs_run = 0

        self.opt = optimiser
        self.save_every = checkpoint_freq  # Specifies how often we choose to save our model during training
        self.train_loader = train_data_loader
        self.loss_fn = loss_fn  # If callable, need to ensure we allow for gradient computation
        self.loss_aggregator = loss_aggregator().to(self.device_id)

        self.diffusion = diffusion
        self.train_eps = train_eps
        self.max_diff_steps = max_diff_steps
        self.end_diff_time = end_diff_time
        self.is_hybrid = hybrid_training
        self.include_weightings = to_weight

        # Move score network to appropriate device
        if type(self.device_id) == int:
            print("DDP Setup\n")
            print(self.device_id)
            self.score_network = DDP(self.score_network.to(self.device_id), device_ids=[self.device_id])
        else:
            self.score_network = self.score_network.to(self.device_id)

        self.snapshot_path = snapshot_path
        # Load snapshot if available
        if os.path.exists(self.snapshot_path):
            print("Device {} :: Loading snapshot\n".format(self.device_id))
            self._load_snapshot(self.snapshot_path)
        print("!!Setup Done!!\n")

    def _batch_update(self, loss) -> float:
        """
        Backward pass and optimiser update step
            :param loss: loss tensor / function output
            :return: Batch Loss
        """
        loss.backward()  # single gpu functionality
        self.opt.step()
        # Detach returns the loss as a Tensor that does not require gradients, so you can manipulate it
        # independently of the original value, which does require gradients
        # Item is used to return a 1x1 tensor as a standard Python dtype (determined by Tensor dtype)
        self.loss_aggregator.update(loss.detach().item())
        return loss.detach().item()

    def _batch_loss_compute(self, outputs: torch.Tensor, targets: torch.Tensor) -> float:
        """
        Computes loss and calls helper function to compute backward pass
            :param outputs: Model forward pass output
            :param targets: Target values to compare against outputs
            :return: Batch Loss
        """
        loss = self.loss_fn()(outputs, targets)
        return self._batch_update(loss)

    def _run_batch(self, xts: torch.Tensor, features: torch.Tensor, target_scores: torch.Tensor,
                   diff_times: torch.Tensor,
                   eff_times: torch.Tensor) -> float:
        """
        Compute batch output and loss
            :param xts: Diffused samples
            :param target_scores: Target scores at corresponding diff times
            :param diff_times: Diffusion times
            :param eff_times: Effective diffusion times
            :return: Batch Loss
        """
        self.opt.zero_grad()
        B, T, D = xts.shape
        # Reshaping concatenates vectors in dim=1
        xts = xts.reshape(B * T, 1, -1)
        features = features.reshape(B * T, 1, -1)
        target_scores = target_scores.reshape(B * T, 1, -1)
        diff_times = diff_times.reshape(B * T)
        eff_times = eff_times.reshape(target_scores.shape)
        outputs = self.score_network.forward(inputs=xts, conditioner=features, times=diff_times)
        # Outputs should be (NumBatches, TimeSeriesLength, 1)
        weights = self.diffusion.get_loss_weighting(eff_times=eff_times)
        if not self.include_weightings: weights = torch.ones_like(weights)
        return self._batch_loss_compute(outputs=weights * outputs, targets=weights * target_scores)

    def _run_epoch(self, epoch: int) -> list:
        """
        Single epoch run
            :param epoch: Epoch index
            :return: List of batch Losses
        """
        device_epoch_losses = []
        b_sz = len(next(iter(self.train_loader))[0])
        print(
            f"[Device {self.device_id}] Epoch {epoch + 1} | Batchsize: {b_sz} | Total Num of Batches: {len(self.train_loader)} \n")
        if type(self.device_id) != torch.device: self.train_loader.sampler.set_epoch(epoch)
        if self.is_hybrid:
            timesteps = torch.linspace(self.train_eps, end=self.end_diff_time,
                                       steps=self.max_diff_steps)
        for x0s in (iter(self.train_loader)):
            x0s = x0s[0].to(self.device_id)
            # Generate history vector for each time t for a sample in (batch_id, t, numdims)
            features = self.create_historical_vectors(x0s)
            if self.is_hybrid:
                # We select diffusion time uniformly at random for each sample at each time (i.e., size (NumBatches, TimeSeries Sequence))
                diff_times = timesteps[torch.randint(low=0, high=self.max_diff_steps, dtype=torch.int32,
                                                     size=x0s.shape[0:2]).long()].view(x0s.shape[0], x0s.shape[1],
                                                                                       *([1] * len(x0s.shape[2:]))).to(
                    self.device_id)
            else:
                diff_times = ((self.train_eps - self.end_diff_time) * torch.rand(
                    (x0s.shape[0], 1)) + self.end_diff_time).view(x0s.shape[0], x0s.shape[1],
                                                                  *([1] * len(x0s.shape[2:]))).to(
                    self.device_id)
            # Diffusion times shape (Batch Size, Time Series Sequence, 1)
            # so that each (b, t, 1) entry corresponds to the diffusion time for timeseries "b" at time "t"
            eff_times = self.diffusion.get_eff_times(diff_times)
            # Each eff time entry corresponds to the effective diffusion time for timeseries "b" at time "t"
            xts, target_scores = self.diffusion.noising_process(x0s, eff_times)
            # For each timeseries "b", at time "t", we want the score p(timeseries_b_attime_t_diffusedTo_efftime|time_series_b_attime_t)
            # So target score should be size (NumBatches, Time Series Length, 1)
            # And xts should be size (NumBatches, TimeSeriesLength, NumDimensions)
            batch_loss = self._run_batch(xts=xts, features=features, target_scores=target_scores, diff_times=diff_times,
                                         eff_times=eff_times)
            device_epoch_losses.append(batch_loss)
        return device_epoch_losses

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
            self.score_network.module.load_state_dict(snapshot["MODEL_STATE"])
        else:
            self.score_network.load_state_dict(snapshot["MODEL_STATE"])
        print("Device {} :: Resuming training from snapshot at epoch {} and device {}\n".format(self.device_id,
                                                                                                self.epochs_run + 1,
                                                                                                self.device_id))

    def _save_snapshot(self, epoch: int) -> None:
        """
        Save current state of training
            :param epoch: Current epoch number
            :return: None
        """
        snapshot = {"EPOCHS_RUN": epoch + 1, "OPTIMISER_STATE": self.opt.state_dict()}
        # self.score_network now points to DDP wrapped object, so we need to access parameters via ".module"
        if type(self.device_id) == int:
            snapshot["MODEL_STATE"] = self.score_network.module.state_dict()
        else:
            snapshot["MODEL_STATE"] = self.score_network.state_dict()
        torch.save(snapshot, self.snapshot_path)
        print(f"Epoch {epoch + 1} | Training snapshot saved at {self.snapshot_path}\n")

    def _save_model(self, filepath: str, final_epoch: int) -> None:
        """
        Save final trained model
            :param filepath: Filepath to save model
            :param final_epoch: Final training epoch
            :return: None
        """
        # self.score_network now points to DDP wrapped object so we need to access parameters via ".module"
        if type(self.device_id) == int:
            ckp = self.score_network.to(torch.device("cpu")).module.state_dict()  # Save model on CPU
        else:
            ckp = self.score_network.to(torch.device("cpu")).state_dict()  # Save model on CPU
        filepath = filepath + "_NEp{}".format(final_epoch)
        torch.save(ckp, filepath)
        print(f"Trained model saved at {filepath}\n")
        self.score_network.to(self.device_id)  # In the event we continue training after saving
        try:
            pass
            # os.remove(self.snapshot_path)  # Do NOT remove snapshot path yet eventhough training is done
        except FileNotFoundError:
            print("Snapshot file does not exist\n")

    def create_historical_vectors(self, batch):
        """
        Create history vectors using LSTM architecture
            :return: History vectors for each timestamp
        """

        # batch shape (N_batches, Time Series Length, Input Size)
        # hidden states: (D*NumLayers, N, Hidden Dims), D is 2 if bidirectional, else 1.
        dbatch = torch.cat([torch.zeros((batch.shape[0], 1, batch.shape[-1])).to(batch.device), batch], dim=1)
        if type(self.device_id) == int:
            output, (hn, cn) = (self.score_network.module.rnn(dbatch, None))
        else:
            output, (hn, cn) = (self.score_network.rnn(dbatch, None))
        return output[:, :-1, :]

    def _save_loss(self, losses: list, filepath: str):
        """
        Save loss tracker
            :param losses: Epoch losses averaged over GPU and Batches
            :param filepath: Path of file
            :return: None
        """
        with open(filepath.replace("/trained_models/", "/training_losses/") + "_loss",
                  'wb') as fp:
            pickle.dump(losses, fp)

    def _load_loss_tracker(self, filepath: str) -> list:
        """
        Load loss tracking list from stored file (if it exists)
            :param filepath: Path of file
            :return: Loss Tracking List
        """
        try:
            with open(filepath.replace("/trained_models/", "/training_losses/") + "_loss", 'rb') as fp:
                l = pickle.load(fp)
                print("Loading Loss Tracker at Epoch {} with Length {}\n".format(self.epochs_run, len(l)))
                assert (len(l) >= self.epochs_run)
                return l[:self.epochs_run]
        except FileNotFoundError:
            return []

    def train(self, max_epochs: list, model_filename: str) -> None:
        """
        Run training for model
            :param max_epochs: List of maximum number of epochs (to allow for iterative training)
            :param model_filename: Filepath to save model
            :return: None
        """
        max_epochs = sorted(max_epochs)
        self.score_network.train()
        all_losses_per_epoch = self._load_loss_tracker(model_filename)  # This will contain synchronised losses
        end_epoch = max(max_epochs)
        print(len(all_losses_per_epoch), end_epoch, max_epochs)
        for epoch in range(self.epochs_run, end_epoch):
            t0 = time.time()
            device_epoch_losses = self._run_epoch(epoch)
            # Average epoch loss for each device over batches
            epoch_losses_tensor = torch.tensor(torch.mean(torch.tensor(device_epoch_losses)).item())
            if type(self.device_id) == int:
                epoch_losses_tensor = epoch_losses_tensor.cuda()
                all_gpus_losses = [torch.zeros_like(epoch_losses_tensor) for _ in range(torch.cuda.device_count())]
                torch.distributed.all_gather(all_gpus_losses, epoch_losses_tensor)
            else:
                all_gpus_losses = [epoch_losses_tensor]
            # Obtain epoch loss averaged over devices
            average_loss_per_epoch = torch.mean(torch.stack(all_gpus_losses), dim=0)
            all_losses_per_epoch.append(float(average_loss_per_epoch.cpu().numpy()))
            # NOTE: .compute() cannot be called on only one process since it will wait for other processes
            # see  https://github.com/Lightning-AI/torchmetrics/issues/626
            print("Device {} :: Percent Completed {:0.4f} :: Train {:0.4f} :: Time for One Epoch {:0.4f}\n".format(
                self.device_id, (epoch + 1) / end_epoch,
                float(
                    self.loss_aggregator.compute().item()), float(time.time() - t0)))
            if self.device_id == 0 or type(self.device_id) == torch.device:
                print("Stored Running Mean {} vs Aggregator Mean {}\n".format(
                    float(torch.mean(torch.tensor(all_losses_per_epoch[self.epochs_run:])).cpu().numpy()), float(
                        self.loss_aggregator.compute().item())))
                if epoch + 1 in max_epochs:
                    self._save_snapshot(epoch=epoch)
                    self._save_loss(losses=all_losses_per_epoch, filepath=model_filename)
                    self._save_model(filepath=model_filename, final_epoch=epoch + 1)
                elif (epoch + 1) % self.save_every == 0:
                    self._save_loss(losses=all_losses_per_epoch, filepath=model_filename)
                    self._save_snapshot(epoch=epoch)
            if type(self.device_id) == int: dist.barrier()
