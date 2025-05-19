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
from src.generative_modelling.models.TimeDependentScoreNetworks.ClassConditionalMarkovianTSPostMeanScoreMatching import \
    ConditionalMarkovianTSPostMeanScoreMatching
import numpy as np
from configs import project_config
from tqdm import tqdm

from utils.drift_evaluation_functions import MLP_2D_drifts, MLP_1D_drifts, multivar_score_based_MLP_drift_OOS


# Link for DDP vs DataParallelism: https://www.run.ai/guides/multi-gpu/pytorch-multi-gpu-4-techniques-explained
# Link for ddp_setup backend: https://pytorch.org/docs/stable/distributed.html
# Tutorial: https://www.youtube.com/watch?v=-LAtx9Q6DA8


class ConditionalStbleTgtMarkovianPostMeanDiffTrainer(nn.Module):

    def __init__(self,
                 diffusion: Union[VESDEDiffusion, OUSDEDiffusion, VPSDEDiffusion],
                 score_network: Union[ConditionalMarkovianTSPostMeanScoreMatching],
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
                 loss_factor: float,
                 deltaT: float,
                 init_state: torch.Tensor,
                 loss_fn: callable = torch.nn.MSELoss,
                 loss_aggregator: torchmetrics.aggregation = MeanMetric):
        super().__init__()
        self.device_id = device
        assert (self.device_id == torch.device("cpu") or self.device_id == int(os.environ["LOCAL_RANK"]))
        self.score_network = score_network
        self.epochs_run = 0
        self.init_state = init_state
        self.opt = optimiser
        self.scheduler = None
        self.ewma_loss = 0.
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
        print("!!Setup Done!!\n")


    def _batch_update(self, loss, epoch:int, batch_idx:int, num_batches:int) -> float:
            """
            Backward pass and optimiser update step
                :param loss: loss tensor / function output
                :return: Batch Loss
            """
            loss.backward()  # single gpu functionality
            self.opt.step()
            #try:
            #    self.scheduler.step(epoch + batch_idx / num_batches)
            #except AttributeError as e:
            #    pass

            # Detach returns the loss as a Tensor that does not require gradients, so you can manipulate it
            # independently of the original value, which does require gradients
            # Item is used to return a 1x1 tensor as a standard Python dtype (determined by Tensor dtype)
            self.loss_aggregator.update(loss.detach().item())
            return loss.detach().item()

    def _batch_loss_compute(self, outputs: torch.Tensor, targets: torch.Tensor, epoch:int, batch_idx:int, num_batches:int) -> float:
        """
        Computes loss and calls helper function to compute backward pass
            :param outputs: Model forward pass output
            :param targets: Target values to compare against outputs
            :return: Batch Loss
        """
        loss = self.loss_fn()(outputs, targets)
        var_loss = ((self.score_network.module.mlp_state_mapper.hybrid.log_scale - self.score_network.module.mlp_state_mapper.hybrid.log_scale.mean()) ** 2).mean()
        loss += 0.001*var_loss
        return self._batch_update(loss, epoch=epoch, batch_idx=batch_idx, num_batches=num_batches)


    def _run_batch(self, xts: torch.Tensor, features: torch.Tensor, stable_targets: torch.Tensor,
                   diff_times: torch.Tensor,
                   eff_times: torch.Tensor, epoch:int, batch_idx:int, num_batches:int) -> float:
        """
        Compute batch output and loss
            :param xts: Diffused samples
            :param stable_targets: Target scores at corresponding diff times
            :param diff_times: Diffusion times
            :param eff_times: Effective diffusion times
            :return: Batch Loss
        """
        self.opt.zero_grad()
        B, T, D = xts.shape
        assert (features.shape[:2] == (B, T) and features.shape[-1] == D)
        # Reshaping concatenates vectors in dim=1
        xts = xts.reshape(B * T, 1, D)
        # Features is originally shaped (NumTimeSeries, TimeSeriesLength, LookBackWindow, TimeSeriesDim)
        # Reshape so that we have (NumTimeSeries*TimeSeriesLength, 1, LookBackWindow, TimeSeriesDim)
        ##features = features.reshape(B * T, 1, 1, D)
        # Now reshape again into (NumTimeSeries*TimeSeriesLength, 1, LookBackWindow*TimeSeriesDim)
        # Note this is for the simplest implementation of CondUpsampler which is simply an MLP
        ##features = features.reshape(B * T, 1, 1 * D, 1).permute((0, 1, 3, 2)).squeeze(2)
        features = features.reshape(B*T, 1, D)
        stable_targets = stable_targets.reshape(B * T, 1, -1)
        diff_times = diff_times.reshape(B * T)
        eff_times = torch.cat([eff_times]*D, dim=2).reshape(stable_targets.shape)
        outputs = self.score_network.forward(inputs=xts, conditioner=features, times=diff_times, eff_times=eff_times)
        # Outputs should be (NumBatches, TimeSeriesLength, 1)
        # For times larger than tau0, use inverse_weighting
        sigma_tau = 1. - torch.exp(-eff_times)  # This is sigma2
        beta_tau = torch.exp(-0.5 * eff_times)
        if self.loss_factor == 0:  # PM
            weights = torch.ones_like(outputs)
        elif self.loss_factor == 1:  # PMScaled (meaning not scaled)
            weights = self.diffusion.get_loss_weighting(eff_times=outputs)
        elif self.loss_factor == 2 or self.loss_factor == 21:  # PM with deltaT scaling
            weights = torch.ones_like(outputs) / torch.sqrt(self.deltaT)
        # Outputs should be (NumBatches, TimeSeriesLength, 1)
        # Now implement the stable target field
        outputs = (outputs + xts / sigma_tau) * (sigma_tau / beta_tau)  # This gives us the network D_theta
        assert (outputs.shape == stable_targets.shape)
        return self._batch_loss_compute(outputs=outputs * weights, targets=stable_targets * weights, epoch=epoch, batch_idx=batch_idx, num_batches=num_batches)

    def _compute_stable_targets(self, batch: torch.Tensor, noised_z:torch.Tensor, eff_times: torch.Tensor, ref_batch: torch.Tensor, chunk_size:int, feat_thresh:float):
        import time
        t0 = time.time()
        B1, T, D = batch.shape
        B2, T, D = ref_batch.shape
        print(B2, B1)
        assert (B2 > B1)
        dX = feat_thresh
        # ref_batch, batch, eff_times = ref_batch.to("cpu"), batch.to("cpu"), eff_times.to("cpu")
        pos_ref_batch = self._from_incs_to_positions(batch=ref_batch)[:, :-1, :]  # shape: [B2, T, D]
        pos_batch = self._from_incs_to_positions(batch=batch)[:, :-1, :]  # shape: [B1, T, D]
        assert pos_batch.shape == batch.shape, "pos_batch must match batch shape"
        assert pos_batch.shape == (B1, T, D)
        assert pos_ref_batch.shape == (B2, T, D)
        pos_ref_batch = pos_ref_batch.reshape(-1, pos_ref_batch.shape[-1])
        assert pos_ref_batch.shape == (B2 * T, D)
        pos_batch = pos_batch.reshape(-1, pos_batch.shape[-1])
        assert pos_batch.shape == (B1 * T, D)
        ref_batch = ref_batch.reshape(-1, ref_batch.shape[-1])
        assert ref_batch.shape == pos_ref_batch.shape
        batch = batch.reshape(-1, batch.shape[-1])
        assert batch.shape == pos_batch.shape
        eff_times = eff_times.reshape(-1, eff_times.shape[-1])
        if eff_times.shape[-1] == 1 and D > 1:
            eff_times = eff_times.expand(-1, D)
        assert eff_times.shape == (B1 * T, D)  # Because the ref batch is only for the purposes of importance sampling

        target_x = pos_batch  # [B1*T, D]
        target_x_exp = target_x.unsqueeze(1)  # [B1*T, 1, D]
        assert target_x_exp.shape == (B1 * T, 1, D)
        # candidate -> potential positions which are close to our "X" feature
        candidate_x = pos_ref_batch.unsqueeze(0)  # [1, B2*T, D]
        assert candidate_x.shape == (1, B2 * T, D)
        # candidate_Z -> potential next increments whose previous position is close to our "X"
        candidate_Z = ref_batch.unsqueeze(0)  # [1, B2*T, D]
        assert candidate_Z.shape == (1, B2 * T, D)

        # batch, eff_times = batch.to(self.device_id), eff_times.to(self.device_id)
        noised_z = noised_z.reshape(-1, noised_z.shape[-1])
        assert (noised_z.shape == (B1 * T, D))
        # batch, eff_times = batch.to("cpu"), eff_times.to("cpu")
        beta_tau = torch.exp(-0.5 * eff_times)
        assert beta_tau.shape == (B1 * T, D)
        sigma_tau = 1. - torch.exp(-eff_times)
        assert sigma_tau.shape == beta_tau.shape
        # noised_z, beta_tau, sigma_tau = noised_z.to("cpu"), beta_tau.to("cpu"), sigma_tau.to("cpu")

        target_noised_z = noised_z.unsqueeze(1)  # [B1*T, 1, D]
        target_beta_tau = beta_tau.unsqueeze(1)  # [B1*T, 1, D]
        target_sigma_tau = sigma_tau.unsqueeze(1)  # [B1*T, 1, D]
        assert target_noised_z.shape == target_beta_tau.shape == target_sigma_tau.shape == (B1 * T, 1, D)
        # We will iterate over all targets in our sub-sampled batch
        stable_targets_chunks = []
        stable_targets_masks = []
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
            # Broadcasting: candidate_x is [1, B2*T, D] and target_chunk is [chunk, 1, D].
            # candidate_x, target_chunk = candidate_x.to(self.device_id), target_chunk.to(self.device_id)
            mask_chunk = ((torch.norm(candidate_x - target_chunk, p=2, dim=-1) / D) <= dX).float()
            assert (mask_chunk.shape == (chunk_size, B2 * T) or mask_chunk.shape == (chunk_size, B2*T, 1))
            if mask_chunk.dim() > 2: mask_chunk = mask_chunk.squeeze(-1)
            assert mask_chunk.shape == (chunk_size, B2 * T)
            # 2. find columns where no element is 1
            #    `any` over dim=0 gives [B2*T] bool telling us if each column has any True
            rows_has_any = mask_chunk.bool().any(dim=1)  # shape: [chunk_size]
            # 3. if some columns are all zero, recompute them with 2*dX
            ddX = dX
            while not rows_has_any.all():
                # recompute full mask at 2*dX
                ddX = 1.2*ddX
                mask2 = ((torch.norm(candidate_x - target_chunk, p=2, dim=-1) / D) <= ddX).float()
                if mask2.dim() > 2: mask2 = mask2.squeeze(-1)
                # replace only the “all-zero” columns
                zero_rows = ~rows_has_any  # shape: [chunk_size]
                mask_chunk[zero_rows, :] = mask2[zero_rows, :]
                rows_has_any = mask_chunk.bool().any(dim=1)  # shape: [chunk_size]

            if mask_chunk.dim() == 2:
                mask_chunk = mask_chunk.unsqueeze(-1)
            assert mask_chunk.shape == (chunk_size, B2 * T, 1)

            # candidate_x, target_chunk = candidate_x.to("cpu"), target_chunk.to("cpu")

            # mask_chunk is size [chunk, B2*T, D]

            # candidate_Z = candidate_Z.to(self.device_id)
            # beta_tau_chunk, sigma_tau_chunk = beta_tau_chunk.to(self.device_id), sigma_tau_chunk.to(self.device_id)

            # --- Compute the distribution parameters (chunk) ---
            # Compute dist_mean for this chunk: target_beta_tau_chunk * candidate_Z
            dist_mean_chunk = beta_tau_chunk * candidate_Z  # [chunk, B1*T, D]
            # Create a Normal distribution with mean=dist_mean_chunk and std = sqrt(sigma_tau_chunk).
            dist_chunk = torch.distributions.Normal(dist_mean_chunk, torch.sqrt(sigma_tau_chunk))

            # beta_tau_chunk, sigma_tau_chunk = beta_tau_chunk.to("cpu"), sigma_tau_chunk.to("cpu")
            # noised_z_chunk = noised_z_chunk.to(self.device_id)
            # Compute weights via the log probability of noised_z_chunk, then exponentiate.
            weights_chunk = dist_chunk.log_prob(noised_z_chunk).sum(dim=-1, keepdim=True).exp()  # [chunk, B2*T, 1]
            assert weights_chunk.shape == (chunk_size, B2*T, 1)
            # noised_z_chunk = noised_z_chunk.to("cpu")

            # Apply the mask to zero out values that are not in the desired range.
            weights_masked_chunk = weights_chunk * mask_chunk  # [chunk_size, B2*T, 1]
            assert weights_masked_chunk.shape == (chunk_size, B2*T, 1)
            # --- Aggregate weights and candidate_Z contributions ---
            # Sum over the candidate dimension (dim=1) to get total weights per target element.
            weight_sum_chunk = weights_masked_chunk.sum(dim=1)  # [chunk_size, 1]
            assert weight_sum_chunk.shape == (chunk_size, 1)
            c = 1./torch.max(torch.abs(weights_masked_chunk[:,:, 0]))
            num = torch.pow(torch.sum(c * weights_masked_chunk, dim=1), 2)
            denom = (torch.sum(torch.pow(c * weights_masked_chunk, 2), dim=1)) + 1e-12
            ESS = torch.where(denom == 0, torch.tensor(0.0, device=denom.device, dtype=denom.dtype), num / denom).to("cpu")
            stable_targets_masks.append(ESS)
            assert (not torch.any(torch.isnan(ESS)))
            weighted_Z_sum_chunk = (weights_masked_chunk * candidate_Z).sum(dim=1)  # [chunk, D]
            assert weighted_Z_sum_chunk.shape == (chunk_size, D)
            # candidate_Z = candidate_Z.to("cpu")
            # Compute stable target estimates for this chunk.
            # Add a small epsilon to avoid division by zero.
            if torch.any(weight_sum_chunk == 0):
                min_positive = torch.min(weight_sum_chunk[weight_sum_chunk > 0]).item()
                epsilon = min_positive / 1000
            else:
                epsilon = 0.
            # Now apply epsilon only where needed
            denominator = weight_sum_chunk + (weight_sum_chunk == 0) * epsilon
            stable_targets_chunk = weighted_Z_sum_chunk / denominator # [chunk_size, D]
            assert stable_targets_chunk.shape == (chunk_size, D)
            stable_targets_chunks.append(stable_targets_chunk)
            assert (not torch.any(torch.isnan(stable_targets_chunk)))
            del weight_sum_chunk, weighted_Z_sum_chunk
            if ddX != dX: print(f"Final feat thresh vs original feat thresh: {ddX, dX}\n")
        stable_targets_masks = (torch.cat(stable_targets_masks, dim=0))
        assert stable_targets_masks.shape == (B1 * T, 1)
        print(f"IQR ESS: {torch.quantile(stable_targets_masks, q=0.005, dim=0).item(), torch.quantile(stable_targets_masks, q=0.995, dim=0).item()}\n")
        # Concatenate all chunks to form the full result.
        stable_targets = torch.cat(stable_targets_chunks, dim=0)  # [B1*T, D]
        assert (stable_targets.shape == (B1 * T, D))
        del pos_batch, pos_ref_batch, mask_chunk, dist_mean_chunk, stable_targets_chunks, target_noised_z, target_beta_tau, target_sigma_tau
        # ref_batch, batch, eff_times = ref_batch.to(self.device_id), batch.to(self.device_id), eff_times.to(self.device_id)
        return stable_targets.to(self.device_id)

    def _run_epoch(self, epoch: int,batch_size: int, chunk_size:int, feat_thresh:float) -> list:
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
        for batch_idx, x0s in enumerate(self.train_loader):
            ref_x0s = x0s[0].to(self.device_id)
            indices = torch.randperm(ref_x0s.shape[0])[:batch_size]
            # x0s is the subsampled set of increments from the larger reference batch
            x0s = ref_x0s[indices, :, :]
            # Generate history vector for each time t for a sample in (batch_id, t, numdims)
            features = self.create_feature_vectors_from_position(x0s)
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
            xts, _ = self.diffusion.noising_process(x0s, eff_times)
            # For each timeseries "b", at time "t", we want the score p(timeseries_b_attime_t_diffusedTo_efftime|time_series_b_attime_t)
            # So target score should be size (NumBatches, Time Series Length, 1)
            # And xts should be size (NumBatches, TimeSeriesLength, NumDimensions)
            stable_targets = self._compute_stable_targets(batch=x0s, noised_z=xts, ref_batch=ref_x0s, eff_times=eff_times, chunk_size=chunk_size, feat_thresh=feat_thresh)

            batch_loss = self._run_batch(xts=xts, features=features, stable_targets=stable_targets,
                                         diff_times=diff_times,
                                         eff_times=eff_times, epoch=epoch, batch_idx=batch_idx,
                                         num_batches=len(self.train_loader))
            device_epoch_losses.append(batch_loss)
        return device_epoch_losses

    def _load_snapshot(self, snapshot_path: str, config) -> None:
        """
        Load training from most recent snapshot
            :param snapshot_path: Path to training snapshot
            :return: None
        """
        # Snapshot should be python dict
        for param_group in self.opt.param_groups:
            param_group['lr'] = 1e-3
        print(f"Before loading snapshot Epochs Run, EWMA Loss, LR: {self.epochs_run, self.ewma_loss, self.opt.param_groups[0]['lr']}\n")

        loc = 'cuda:{}'.format(self.device_id) if type(self.device_id) == int else self.device_id
        try:
            snapshot = torch.load(snapshot_path, map_location=loc)
            self.epochs_run = snapshot["EPOCHS_RUN"]
            self.opt.load_state_dict(snapshot["OPTIMISER_STATE"])
            # Here to manually change the LR
            #if "BiPot" in config.data_path and self.opt.param_groups[0]["lr"] == 0.001 and config.feat_thresh == 1./50 and 550<= self.epochs_run <= 600: self.opt.param_groups[0]["lr"]=0.0001
            #if "QuadSin" in config.data_path and config.sin_space_scale == 4. and self.opt.param_groups[0]["lr"] == 0.001 and config.feat_thresh == 1./50 and 630<= self.epochs_run <= 640 : self.opt.param_groups[0]["lr"]=0.0001
            if "QuadSin" in config.data_path and config.sin_space_scale == 25. and self.opt.param_groups[0]["lr"] == 0.001 and config.feat_thresh == 1./50. and 100<= self.epochs_run <= 120 : self.opt.param_groups[0]["lr"]=0.0001

            try:
                self.ewma_loss = snapshot["EWMA_LOSS"]
            except KeyError as e:
                print(e)
                pass
            if type(self.device_id) == int:
                self.score_network.module.load_state_dict(snapshot["MODEL_STATE"])
            else:
                self.score_network.load_state_dict(snapshot["MODEL_STATE"])
            #if ("QuadSinHF" in config.data_path and "004b" in config.data_path and config.feat_thresh == 1. / 50.):
            #    print("Using linear LR increase over 1000 epochs\n")
            #    self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.opt, lambda e: (1e-4 / 1e-5) ** (e / 1000),
            #                                                       last_epoch=-1)
            #else:
            print("Using RLRP scheduler\n")
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    self.opt,
                    mode='min',  # We're monitoring a loss that should decrease.
                    factor=0.5,  # Reduce learning rate by 50% (more conservative than 90%).
                    patience=50,  # Wait for 50 epochs of no sufficient improvement.
                    verbose=True,  # Print a message when the LR is reduced.
                    threshold=1e-4,  # Set the threshold for what counts as improvement.
                    threshold_mode='rel',  # Relative change compared to the best value so far.
                    cooldown=200,  # Optionally, add cooldown epochs after a reduction.
                    min_lr=1e-5
                )
            try:
                self.scheduler.load_state_dict(snapshot["SCHEDULER_STATE"])
            except (KeyError, AttributeError) as e:
                print(e)
                pass
        except FileNotFoundError:
            #if ("QuadSinHF" in config.data_path and "004b" in config.data_path and config.feat_thresh == 1. / 50.):
            #    print("Using linear LR increase over 1000 epochs\n")
            #    self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.opt, lambda e: (1e-3 / 1e-5) ** min(1.,(e / 2000)),
            #                                                       last_epoch=-1)
            #else:
            print("Using RLRP scheduler\n")
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    self.opt,
                    mode='min',  # We're monitoring a loss that should decrease.
                    factor=0.5,  # Reduce learning rate by 50% (more conservative than 90%).
                    patience=50,  # Wait for 50 epochs of no sufficient improvement.
                    verbose=True,  # Print a message when the LR is reduced.
                    threshold=1e-4,  # Set the threshold for what counts as improvement.
                    threshold_mode='rel',  # Relative change compared to the best value so far.
                    cooldown=200,  # Optionally, add cooldown epochs after a reduction.
                    min_lr=1e-5
                )
        print(f"After loading snapshot Epochs Run, EWMA Loss, LR: {self.epochs_run, self.ewma_loss, self.opt.param_groups[0]['lr']}\n")


    def _save_snapshot(self, epoch: int) -> None:
        """
        Save current state of training
            :param epoch: Current epoch number
            :return: None
        """
        try:
            snapshot = {"EPOCHS_RUN": epoch + 1, "OPTIMISER_STATE": self.opt.state_dict(), "SCHEDULER_STATE":self.scheduler.state_dict(), "EWMA_LOSS":self.ewma_loss}
        except AttributeError as e:
            print(e)
            snapshot = {"EPOCHS_RUN": epoch + 1, "OPTIMISER_STATE": self.opt.state_dict(), "EWMA_LOSS":self.ewma_loss}

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
        dbatch[:, 0, :] +=  1e-3*torch.randn((dbatch.shape[0], dbatch.shape[-1])).to(dbatch.device)
        return dbatch

    def create_feature_vectors_from_position(self, batch):
        """
        Create history vectors using Markovian architecture
            :return: History vectors for each timestamp
        """
        return self._from_incs_to_positions(batch)[:, :-1, :]

    def _save_loss(self, losses: list, learning_rates: list, filepath: str):
        """
        Save loss tracker
            :param losses: Epoch losses averaged over GPU and Batches
            :param learning_rates: Per epoch learning rates
            :param filepath: Path of file
            :return: None
        """
        with open(filepath.replace("/trained_models/", "/training_losses/") + "_loss",
                  'wb') as fp:
            pickle.dump(losses, fp)
        with open(filepath.replace("/trained_models/", "/training_losses/") + "_loss_LR",
                  'wb') as fp:
            pickle.dump(learning_rates, fp)

    def _load_loss_tracker(self, filepath: str) -> [list, list]:
        """
        Load loss tracking list from stored file (if it exists)
            :param filepath: Path of file
            :return: Loss Tracking List, Learning Rate List
        """
        try:
            with open(filepath.replace("/trained_models/", "/training_losses/") + "_loss", 'rb') as fp:
                l = pickle.load(fp)
                print("Loading Loss Tracker at Epoch {} with Length {}\n".format(self.epochs_run, len(l)))
                assert (len(l) >= self.epochs_run)
                l = l[:self.epochs_run]
        except FileNotFoundError:
            l = []
        try:
            with open(filepath.replace("/trained_models/", "/training_losses/") + "_loss_LR", 'rb') as fp:
                learning_rates = pickle.load(fp)
                print("Loading Loss Tracker at Epoch {} with Length {}\n".format(self.epochs_run, len(learning_rates)))
                learning_rates = learning_rates[:self.epochs_run]
        except FileNotFoundError:
            learning_rates = []
        if len(l) > len(learning_rates) and len(learning_rates) == 0: # Issue due to unsaved learning rates
            learning_rates = [self.opt.param_groups[0]["lr"]]*len(l)
        assert (len(learning_rates) >= self.epochs_run)
        return l, learning_rates


    def _domain_rmse(self, epoch, config):
        assert (config.ndims <= 2)
        if "MullerBrown" in config.data_path:
            final_vec_mu_hats = MLP_2D_drifts(PM=self.score_network.module, config=config)
        else:
            final_vec_mu_hats = MLP_1D_drifts(PM=self.score_network.module, config=config)
        type = "PM"
        assert (type in config.scoreNet_trained_path)
        assert ("_ST_" in config.scoreNet_trained_path)
        if "BiPot" in config.data_path:
            save_path = (
                    project_config.ROOT_DIR + f"experiments/results/TSPM_MLP_ST_{config.feat_thresh:.3f}FTh_fBiPot_DriftEvalExp_{epoch}Nep_{config.t0}t0_{config.deltaT:.3e}dT_{config.quartic_coeff}a_{config.quad_coeff}b_{config.const}c_{config.residual_layers}ResLay_{config.loss_factor}LFac_BetaMax{config.beta_max:.1e}").replace(
                ".", "")
        elif "QuadSin" in config.data_path:
            save_path = (
                    project_config.ROOT_DIR + f"experiments/results/TSPM_MLP_ST_{config.feat_thresh:.3f}FTh_fQuadSinHF_DriftEvalExp_{epoch}Nep_{config.t0}t0_{config.deltaT:.3e}dT_{config.quad_coeff}a_{config.sin_coeff}b_{config.sin_space_scale}c_{config.residual_layers}ResLay_{config.loss_factor}LFac_BetaMax{config.beta_max:.1e}").replace(
                ".", "")
        elif "SinLog" in config.data_path:
            save_path = (
                    project_config.ROOT_DIR + f"experiments/results/TSPM_MLP_ST_{config.feat_thresh:.3f}FTh_fSinLog_DriftEvalExp_{epoch}Nep_{config.t0}t0_{config.deltaT:.3e}dT_{config.log_space_scale}b_{config.sin_space_scale}c_{config.residual_layers}ResLay_{config.loss_factor}LFac_BetaMax{config.beta_max:.1e}").replace(
                ".", "")
        elif "MullerBrown" in config.data_path:
            save_path = (
                    project_config.ROOT_DIR + f"experiments/results/TSPM_MLP_ST_{config.feat_thresh:.3f}FTh_fMullerBrown_DriftEvalExp_{epoch}Nep_{config.t0}t0_{config.deltaT:.3e}dT_{config.residual_layers}ResLay_{config.loss_factor}LFac_BetaMax{config.beta_max:.1e}").replace(
                ".", "")
        print(f"Save path:{save_path}\n")
        np.save(save_path + "_muhats.npy", final_vec_mu_hats)
        self.score_network.module.train()
        self.score_network.module.to(self.device_id)

    def _tracking_errors(self, epoch, config):
        def true_drift(prev, num_paths, config):
            assert (prev.shape == (num_paths, config.ndims))
            if "BiPot" in config.data_path:
                drift_X = -(4. * config.quartic_coeff * np.power(prev,
                                                                 3) + 2. * config.quad_coeff * prev + config.const)
                return drift_X[:, np.newaxis, :]
            elif "QuadSin" in config.data_path:
                drift_X = -2. * config.quad_coeff * prev + config.sin_coeff * config.sin_space_scale * np.sin(
                    config.sin_space_scale * prev)
                return drift_X[:, np.newaxis, :]
            elif "SinLog" in config.data_path:
                drift_X = -np.sin(config.sin_space_scale*prev)*np.log(1+config.log_space_scale*np.abs(prev))/config.sin_space_scale
                return drift_X[:, np.newaxis, :]
            elif "MullerBrown" in config.data_path:
                Aks = np.array(config.Aks)[np.newaxis, :]
                aks = np.array(config.aks)[np.newaxis, :]
                bks = np.array(config.bks)[np.newaxis, :]
                cks = np.array(config.cks)[np.newaxis, :]
                X0s = np.array(config.X0s)[np.newaxis, :]
                Y0s = np.array(config.Y0s)[np.newaxis, :]
                common = Aks * np.exp(aks * np.power(prev[:, [0]] - X0s, 2) \
                                      + bks * (prev[:, [0]] - X0s) * (prev[:, [1]] - Y0s)
                                      + cks * np.power(prev[:, [1]] - Y0s, 2))
                assert (common.shape == (num_paths, 4))
                drift_X = np.zeros((num_paths, config.ndims))
                drift_X[:, 0] = -np.sum(common * (2. * aks * (prev[:, [0]] - X0s) + bks * (prev[:, [1]] - Y0s)), axis=1)
                drift_X[:, 1] = -np.sum(common * (2. * cks * (prev[:, [1]] - Y0s) + bks * (prev[:, [0]] - X0s)), axis=1)

                return drift_X[:, np.newaxis, :]
            elif "Lnz" in config.data_path and config.ndims == 3:
                drift_X = np.zeros((num_paths, config.ndims))
                drift_X[:, 0] = config.ts_sigma * (prev[:, 1] - prev[:, 0])
                drift_X[:, 1] = (prev[:, 0] * (config.ts_rho - prev[:, 2]) - prev[:, 1])
                drift_X[:, 2] = (prev[:, 0] * prev[:, 1] - config.ts_beta * prev[:, 2])
                return drift_X[:, np.newaxis, :]
            elif "Lnz" in config.data_path:
                drift_X = np.zeros((num_paths, config.ndims))
                for i in range(config.ndims):
                    drift_X[:, i] = (prev[:, (i + 1) % config.ndims] - prev[:, i - 2]) * prev[:, i - 1] - prev[:,
                                                                                                          i] + config.forcing_const
                return drift_X[:, np.newaxis, :]

        diffusion = VPSDEDiffusion(beta_max=config.beta_max, beta_min=config.beta_min)
        num_diff_times = 1
        rmse_quantile_nums = 20
        num_paths = 100
        num_time_steps = 100
        all_true_states = np.zeros(shape=(rmse_quantile_nums, num_paths, 1 + num_time_steps, config.ndims))
        # all_global_states = np.zeros(shape=(rmse_quantile_nums, num_paths, 1 + num_time_steps, config.ndims))
        all_local_states = np.zeros(shape=(rmse_quantile_nums, num_paths, 1 + num_time_steps, config.ndims))
        for quant_idx in tqdm(range(rmse_quantile_nums)):
            self.score_network.module.eval()
            num_paths = 100
            num_time_steps = 100
            deltaT = config.deltaT
            initial_state = np.repeat(np.atleast_2d(config.initState)[np.newaxis, :], num_paths, axis=0)
            assert (initial_state.shape == (num_paths, 1, config.ndims))

            true_states = np.zeros(shape=(num_paths, 1 + num_time_steps, config.ndims))
            # global_states = np.zeros(shape=(num_paths, 1 + num_time_steps, config.ndims))
            local_states = np.zeros(shape=(num_paths, 1 + num_time_steps, config.ndims))

            # Initialise the "true paths"
            true_states[:, [0], :] = initial_state + 0.00001 * np.random.randn(*initial_state.shape)
            # Initialise the "global score-based drift paths"
            # global_states[:, [0], :] = true_states[:, [0], :]
            local_states[:, [0], :] = true_states[:, [0],
                                      :]  # np.repeat(initial_state[np.newaxis, :], num_diff_times, axis=0)

            # Euler-Maruyama Scheme for Tracking Errors
            for i in range(1, num_time_steps + 1):
                eps = np.random.randn(num_paths, 1, config.ndims) * np.sqrt(deltaT)
                assert (eps.shape == (num_paths, 1, config.ndims))
                true_mean = true_drift(true_states[:, i - 1, :], num_paths=num_paths, config=config)

                true_states[:, [i], :] = true_states[:, [i - 1], :] \
                                         + true_mean * deltaT \
                                         + eps
                local_mean= multivar_score_based_MLP_drift_OOS(
                    score_model=self.score_network.module,
                    num_diff_times=num_diff_times,
                    diffusion=diffusion,
                    num_paths=num_paths, ts_step=deltaT,
                    config=config,
                    device=self.device_id,
                    prev=true_states[:, i - 1, :])

                local_states[:, [i], :] = true_states[:, [i - 1], :] + local_mean * deltaT + eps
            all_true_states[quant_idx, :, :, :] = true_states
            all_local_states[quant_idx, :, :, :] = local_states
        if "BiPot" in config.data_path:
            save_path = (
                    project_config.ROOT_DIR + f"experiments/results/TSPM_MLP_ST_{config.feat_thresh:.3f}FTh_fBiPot_OOSDriftTrack_{epoch}Nep_{config.t0}t0_{config.deltaT:.3e}dT_{config.quartic_coeff}a_{config.quad_coeff}b_{config.const}c_{config.residual_layers}ResLay_{config.loss_factor}LFac_BetaMax{config.beta_max:.1e}").replace(
                ".", "")
        elif "QuadSin" in config.data_path:
            save_path = (
                    project_config.ROOT_DIR + f"experiments/results/TSPM_MLP_ST_{config.feat_thresh:.3f}FTh_fQuadSinHF_OOSDriftTrack_{epoch}Nep_{config.t0}t0_{config.deltaT:.3e}dT_{config.quad_coeff}a_{config.sin_coeff}b_{config.sin_space_scale}c_{config.residual_layers}ResLay_{config.loss_factor}LFac_BetaMax{config.beta_max:.1e}").replace(
                ".", "")
        elif "SinLog" in config.data_path:
            save_path = (
                    project_config.ROOT_DIR + f"experiments/results/TSPM_MLP_ST_{config.feat_thresh:.3f}FTh_fSinLog_OOSDriftTrack_{epoch}Nep_{config.t0}t0_{config.deltaT:.3e}dT_{config.log_space_scale}b_{config.sin_space_scale}c_{config.residual_layers}ResLay_{config.loss_factor}LFac_BetaMax{config.beta_max:.1e}").replace(
                ".", "")
        elif "MullerBrown" in config.data_path:
            save_path = (
                    project_config.ROOT_DIR + f"experiments/results/TSPM_MLP_ST_{config.feat_thresh:.3f}FTh_fMullerBrown_DriftTrack_{epoch}Nep_{config.t0}t0_{config.deltaT:.3e}dT_{config.residual_layers}ResLay_{config.loss_factor}LFac_BetaMax{config.beta_max:.1e}").replace(
                ".", "")
        elif "Lnz" in config.data_path:
            save_path = (
                    project_config.ROOT_DIR + f"experiments/results/TSPM_MLP_ST_{config.feat_thresh:.3f}FTh_{config.ndims}DLorenz_OOSDriftTrack_{epoch}Nep_tl{config.tdata_mult}data_{config.t0}t0_{config.deltaT:.3e}dT_{num_diff_times}NDT_{config.loss_factor}LFac_BetaMax{config.beta_max:.1e}_{round(config.forcing_const, 3)}FConst").replace(
                ".", "")
        print(f"Save path for OOS DriftTrack:{save_path}\n")
        np.save(save_path + "_true_states.npy", all_true_states)
        # np.save(save_path + "_global_states.npy", all_global_states)
        np.save(save_path + "_local_states.npy", all_local_states)
        self.score_network.module.train()
        self.score_network.module.to(self.device_id)


    def train(self, max_epochs: list, model_filename: str,batch_size: int, config) -> None:
        """
        Run training for model
            :param max_epochs: List of maximum number of epochs (to allow for iterative training)
            :param model_filename: Filepath to save model
            :return: None
        """
        assert ("_ST_" in config.scoreNet_trained_path)
        # Load snapshot if available
        # if os.path.exists(self.snapshot_path):
        print("Device {} :: Loading snapshot\n".format(self.device_id))
        self._load_snapshot(self.snapshot_path, config=config)
        max_epochs = sorted(max_epochs)
        self.score_network.train()
        all_losses_per_epoch, learning_rates = self._load_loss_tracker(
            model_filename)  # This will contain synchronised losses
        end_epoch = max(max_epochs)
        self.ewma_loss = 0. # Force recomputation of EWMA losses each time
        for epoch in range(self.epochs_run, end_epoch):
            t0 = time.time()
            device_epoch_losses = self._run_epoch(epoch=epoch, batch_size=batch_size, chunk_size=config.chunk_size,
                                                  feat_thresh=config.feat_thresh)
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
            curr_loss = float(torch.mean(torch.tensor(all_losses_per_epoch[-1])).cpu().numpy())
            # Step the scheduler with the validation loss:
            if epoch == 0:
                self.ewma_loss = curr_loss
            else:
                if self.ewma_loss == 0.:  # Issue with saving ewma_loss
                    for i in range(1, len(all_losses_per_epoch)):
                        self.ewma_loss = (1. - 0.95) * all_losses_per_epoch[i] + 0.95 * self.ewma_loss
                    assert (self.ewma_loss != 0.)
                self.ewma_loss = (1. - 0.95) * curr_loss + 0.95 * self.ewma_loss
            if isinstance(self.scheduler, torch.optim.lr_scheduler.LambdaLR):
                print("Using LambdaLR")
                self.scheduler.step()
            else:
                self.scheduler.step(self.ewma_loss)
            # Log current learning rate:
            current_lr = self.opt.param_groups[0]['lr']
            print(f"Epoch {epoch + 1}: EWMA Loss: {self.ewma_loss:.6f}, LR: {current_lr:.12f}\n")
            learning_rates.append(current_lr)

            if self.device_id == 0 or type(self.device_id) == torch.device:
                print("Stored Running Mean {} vs Aggregator Mean {}\n".format(
                    float(torch.mean(torch.tensor(all_losses_per_epoch[self.epochs_run:])).cpu().numpy()), float(
                        self.loss_aggregator.compute().item())))
                print(f"Current Loss {curr_loss}\n")
                if epoch + 1 in max_epochs:
                    self._save_snapshot(epoch=epoch)
                    self._save_loss(losses=all_losses_per_epoch, learning_rates=learning_rates, filepath=model_filename)
                    self._save_model(filepath=model_filename, final_epoch=epoch + 1)
                elif (epoch + 1) % self.save_every == 0:
                    self._save_loss(losses=all_losses_per_epoch, learning_rates=learning_rates, filepath=model_filename)
                    self._save_snapshot(epoch=epoch)
                    self._tracking_errors(epoch=epoch + 1, config=config)
                    if "Lnz" not in config.data_path:
                        self._domain_rmse(config=config, epoch=epoch + 1)
            if type(self.device_id) == int: dist.barrier()
