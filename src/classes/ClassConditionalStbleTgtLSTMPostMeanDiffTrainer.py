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

from src.generative_modelling.models.ClassOUSDEDiffusion import OUSDEDiffusion
from src.generative_modelling.models.ClassVESDEDiffusion import VESDEDiffusion
from src.generative_modelling.models.ClassVPSDEDiffusion import VPSDEDiffusion
from src.generative_modelling.models.TimeDependentScoreNetworks.ClassConditionalLSTMTSPostMeanScoreMatching import \
    ConditionalLSTMTSPostMeanScoreMatching


# Link for DDP vs DataParallelism: https://www.run.ai/guides/multi-gpu/pytorch-multi-gpu-4-techniques-explained
# Link for ddp_setup backend: https://pytorch.org/docs/stable/distributed.html
# Tutorial: https://www.youtube.com/watch?v=-LAtx9Q6DA8


class ConditionalStbleTgtLSTMPostMeanDiffTrainer(nn.Module):

    def __init__(self, diffusion: Union[VPSDEDiffusion, OUSDEDiffusion, VESDEDiffusion], score_network: Union[
        ConditionalLSTMTSPostMeanScoreMatching], train_data_loader: torch.utils.data.dataloader.DataLoader,
                 train_eps: float, end_diff_time: float, max_diff_steps: int, optimiser: torch.optim.Optimizer,
                 snapshot_path: str, device: Union[torch.device, int], checkpoint_freq: int, to_weight: bool,
                 hybrid_training: bool, init_state: torch.Tensor, loss_factor: float, deltaT: float,
                 loss_fn: callable = torch.nn.MSELoss,
                 loss_aggregator: torchmetrics.aggregation = MeanMetric):
        super().__init__()
        self.device_id = device
        assert (self.device_id == torch.device("cpu") or self.device_id == int(os.environ["LOCAL_RANK"]))
        self.score_network = score_network
        self.epochs_run = 0

        self.init_state = init_state
        self.opt = optimiser
        self.save_every = checkpoint_freq  # Specifies how often we choose to save our model during training
        self.train_loader = train_data_loader
        self.loss_fn = loss_fn  # If callable, need to ensure we allow for gradient computation
        self.loss_aggregator = loss_aggregator().to(self.device_id)
        self.loss_factor = loss_factor
        self.deltaT = torch.Tensor([deltaT]).to(self.device_id)

        self.diffusion = diffusion
        self.train_eps = train_eps
        self.max_diff_steps = max_diff_steps
        self.end_diff_time = end_diff_time
        self.is_hybrid = hybrid_training
        self.include_weightings = to_weight
        assert (to_weight == True)
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
        #self.opt.optimizer.step()
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

    def _run_batch(self, xts: torch.Tensor, features: torch.Tensor, stable_targets: torch.Tensor,
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
        #self.opt.optimizer.zero_grad()
        self.opt.zero_grad()
        B, T, D = xts.shape
        # Reshaping concatenates vectors in dim=1
        xts = xts.reshape(B * T, 1, -1)
        features = features.reshape(B * T, 1, -1)
        stable_targets = stable_targets.reshape(B * T, 1, -1)
        diff_times = diff_times.reshape(B * T)
        eff_times = torch.cat([eff_times] * D, dim=2).reshape(stable_targets.shape)
        outputs = self.score_network.forward(inputs=xts, conditioner=features, times=diff_times, eff_times=eff_times)
        # For times larger than tau0, use inverse_weighting
        sigma_tau = 1. - torch.exp(-eff_times)  # This is sigma2
        beta_tau = torch.exp(-0.5 * eff_times)
        if self.loss_factor == 0:  # PM
            weights = torch.ones_like(outputs)
        elif self.loss_factor == 1:  # PMScaled (meaning not scaled)
            weights = self.diffusion.get_loss_weighting(eff_times=outputs)
        elif self.loss_factor == 2:  # PM with deltaT scaling
            weights = torch.ones_like(outputs) / torch.sqrt(self.deltaT)
        # Outputs should be (NumBatches, TimeSeriesLength, 1)
        # Now implement the stable target field
        outputs = (outputs + xts / sigma_tau) * (sigma_tau / beta_tau)  # This gives us the network D_theta
        assert (outputs.shape == stable_targets.shape)
        return self._batch_loss_compute(outputs=outputs * weights, targets=stable_targets * weights)

    def _compute_stable_targets(self, batch: torch.Tensor, eff_times: torch.Tensor, ref_batch: torch.Tensor):

        B1, T, D = batch.shape
        B2, T, D = ref_batch.shape
        dX = 1 / 1000.

        pos_ref_batch = self._from_incs_to_positions(batch=ref_batch)[:, :-1, :]  # shape: [B1, T, D]
        pos_batch = self._from_incs_to_positions(batch=batch)[:, :-1, :]  # shape: [B2, T, D]
        assert pos_batch.shape == batch.shape, "pos_batch must match batch shape"
        pos_ref_batch = pos_ref_batch.reshape(-1, pos_ref_batch.shape[-1])
        pos_batch = pos_batch.reshape(-1, pos_batch.shape[-1])
        ref_batch = ref_batch.reshape(-1, ref_batch.shape[-1])
        batch = batch.reshape(-1, batch.shape[-1])
        eff_times = eff_times.reshape(-1, eff_times.shape[-1])

        """# For every increment (a value) in batch, I want to find a set of increments (values) in the ref_batch
        # whose preceding position in
        stable_scores = []
        from tqdm import tqdm
        for i in tqdm(range(pos_batch.shape[0])):
            x = pos_batch[i, :].squeeze()
            z = batch[i, :].squeeze()
            # (z, x) are the increment, position pair
            eff_tau = eff_times[i, :].squeeze()
            noised_z, _ = self.diffusion.noising_process(z, eff_tau)
            beta_tau = torch.exp(-0.5 * eff_tau)
            sigma_tau = 1. - torch.exp(-eff_tau)
            # Now find ALL positions in position reference batch which are close to our current position
            xmin = x - dX
            xmax = x + dX
            # Compute the mask over the entire sim_data matrix
            mask = ((pos_ref_batch >= xmin) & (pos_ref_batch <= xmax)).float()
            # Get indices where mask is True (each index is [i, j])
            indices = mask.nonzero(as_tuple=False)
            assert (indices.shape[0] > 0)
            weights = []
            Zs = []
            # Now find the next increment corresponding to those positions in position reference batch
            for idx in indices:
                # Uncomment two lines below if we are using pos_ref_batch[:, :, :]
                # if idx[1] < ref_batch.shape[1] - 1:
                # candidate_Z = ref_batch[idx[0], idx[1] + 1, idx[2]]
                assert (idx[1] == 0)
                candidate_Z = ref_batch[idx[0], idx[1]]
                Zs.append(candidate_Z)
                weights.append(torch.distributions.Normal(beta_tau * candidate_Z, torch.sqrt(sigma_tau)).log_prob(
                    noised_z).exp())
            Zs = torch.Tensor(Zs).to(self.device_id)
            weights = torch.Tensor(weights).to(self.device_id)
            weights /= torch.sum(weights)
            stable_scores.append(torch.sum(weights * Zs))
        stable_targets = torch.Tensor(stable_scores)#.reshape(batch.shape).to(self.device_id)
        errs1 = torch.pow(stable_targets.squeeze().cpu() - tds.squeeze().cpu(), 2)
        print(f"Errs1: {torch.mean(errs1), torch.std(errs1)}")"""

        target_x = pos_batch  # [B2*T, D]
        target_x_exp = target_x.unsqueeze(1)  # [B2*T, 1, D]
        candidate_x = pos_ref_batch.unsqueeze(0)  # [1, B1*T, D]
        candidate_Z = ref_batch.unsqueeze(0)  # [1, B1*T, D]

        noised_z, _ = self.diffusion.noising_process(batch, eff_times)
        assert (noised_z.shape == (B1*T, D))
        beta_tau = torch.exp(-0.5 * eff_times)
        sigma_tau = 1. - torch.exp(-eff_times)

        target_noised_z = noised_z.unsqueeze(1)  # [B2*T, 1, D]
        target_beta_tau = beta_tau.unsqueeze(1)  # [B2*T, 1, D]
        target_sigma_tau = sigma_tau.unsqueeze(1)  # [B2*T, 1, D]


        """mask = ((candidate_x.cpu() >= (target_x_exp.cpu() - dX)) & (
                candidate_x.cpu() <= (target_x_exp.cpu() + dX))).float()

        dist_mean = target_beta_tau.cpu() * candidate_Z.cpu()
        dist = torch.distributions.Normal(dist_mean,
                                          torch.sqrt(target_sigma_tau).cpu())

        weights = dist.log_prob(target_noised_z.cpu()).exp()  # [B2*T, B1*T, D]

        weights_masked = weights * mask  # [B2*T, B1*T, D]
        weight_sum = weights_masked.sum(dim=1).to(self.device_id)  # [B2*T, D]
        weighted_Z_sum = (weights_masked * candidate_Z).sum(dim=1).to(self.device_id)  # [B2*T, D]
        stable_targets2 = weighted_Z_sum / (weight_sum)  # [B2*T, D]"""

        chunk_size = 2048  # Adjust as needed based on your available memory.
        stable_targets_chunks = []

        # Loop over the target tensors in chunks
        for i in range(0, target_x_exp.shape[0], chunk_size):
            i_end = min(i + chunk_size, target_x_exp.shape[0])

            # Extract the current chunk of target tensors.
            target_chunk = target_x_exp[i:i_end]  # [chunk, 1, D]
            noised_z_chunk = target_noised_z[i:i_end]  # [chunk, 1, D]
            beta_tau_chunk = target_beta_tau[i:i_end]  # [chunk, 1, D]
            sigma_tau_chunk = target_sigma_tau[i:i_end]  # [chunk, 1, D]

            # --- Compute the mask ---
            # For each target point, we want candidate positions within +/- dX.
            # Broadcasting: candidate_x is [1, B1*T, D] and target_chunk is [chunk, 1, D].
            mask_chunk = ((candidate_x >= (target_chunk - dX)) &
                          (candidate_x <= (target_chunk + dX))).float()
            # mask_chunk has shape: [chunk, B1*T, D]

            # --- Compute the distribution parameters (chunk) ---
            # Compute dist_mean for this chunk: target_beta_tau_chunk * candidate_Z
            dist_mean_chunk = beta_tau_chunk * candidate_Z  # [chunk, B1*T, D]

            # Create a Normal distribution with mean=dist_mean_chunk and std = sqrt(sigma_tau_chunk).
            dist_chunk = torch.distributions.Normal(dist_mean_chunk, torch.sqrt(sigma_tau_chunk))

            # Compute weights via the log probability of noised_z_chunk, then exponentiate.
            weights_chunk = dist_chunk.log_prob(noised_z_chunk).exp()  # [chunk, B1*T, D]

            # Apply the mask to zero out values that are not in the desired range.
            weights_masked_chunk = weights_chunk * mask_chunk  # [chunk, B1*T, D]

            # --- Aggregate weights and candidate_Z contributions ---
            # Sum over the candidate dimension (dim=1) to get total weights per target element.
            weight_sum_chunk = weights_masked_chunk.sum(dim=1)  # [chunk, D]
            weighted_Z_sum_chunk = (weights_masked_chunk * candidate_Z).sum(dim=1)  # [chunk, D]

            # Compute stable target estimates for this chunk.
            # Add a small epsilon to avoid division by zero.
            epsilon = 0.
            stable_targets_chunk = weighted_Z_sum_chunk / (weight_sum_chunk + epsilon)  # [chunk, D]
            stable_targets_chunks.append(stable_targets_chunk)
        # Concatenate all chunks to form the full result.
        stable_targets = torch.cat(stable_targets_chunks, dim=0)  # [B1*T, D]
        assert (stable_targets.shape == (B1*T, D))
        del pos_batch, pos_ref_batch, mask_chunk, dist_mean_chunk, stable_targets_chunks, target_noised_z, target_beta_tau, target_sigma_tau
        return stable_targets.to(self.device_id)



    def _run_epoch(self, epoch: int, batch_size: int) -> list:
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
            ref_x0s = x0s[0].to(self.device_id)
            indices = torch.randperm(ref_x0s.shape[0])[:batch_size]
            x0s = ref_x0s[indices, :, :]
            # Generate history vector for each time t for a sample in (batch_id, t, numdims)
            features = self.create_historical_vectors(x0s)
            if self.is_hybrid:
                # We select diffusion time uniformly at random
                # for each sample at each time (i.e., size (NumBatches, TimeSeries Sequence))
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
            xts, _ = self.diffusion.noising_process(x0s, eff_times)

            stable_targets = self._compute_stable_targets(batch=x0s, ref_batch=ref_x0s, eff_times=eff_times)

            batch_loss = self._run_batch(xts=xts, features=features, stable_targets=stable_targets,
                                         diff_times=diff_times,
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

    def _from_incs_to_positions(self, batch):
        # dbatch = torch.cat([torch.zeros((batch.shape[0], 1, batch.shape[-1])).to(batch.device), batch], dim=1)
        # batch shape (N_batches, Time Series Length, Input Size)
        # hidden states: (D*NumLayers, N, Hidden Dims), D is 2 if bidirectional, else 1.
        init_state = self.init_state.to(batch.device).view(1, 1, batch.shape[-1])  # Reshape to (1, 1, D)
        init_state = init_state.expand(batch.shape[0], -1, -1)  # Expand to (B, 1, D)
        dbatch = torch.cat([init_state, batch], dim=1)
        dbatch = dbatch.cumsum(dim=1)
        return dbatch

    def create_historical_vectors(self, batch):
        """
        Create history vectors using LSTM architecture
            :return: History vectors for each timestamp
        """
        pos_batch = self._from_incs_to_positions(batch)
        if type(self.device_id) == int:
            output, (hn, cn) = (self.score_network.module.rnn(pos_batch, None))
        else:
            output, (hn, cn) = (self.score_network.rnn(pos_batch, None))
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

    def train(self, max_epochs: list, model_filename: str, batch_size: int) -> None:
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
        for epoch in range(self.epochs_run, end_epoch):
            t0 = time.time()
            device_epoch_losses = self._run_epoch(epoch=epoch, batch_size=batch_size)
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
