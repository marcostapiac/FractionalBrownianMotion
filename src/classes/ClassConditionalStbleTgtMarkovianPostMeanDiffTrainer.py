import glob
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

from utils.drift_evaluation_functions import MLP_1D_drifts, multivar_score_based_MLP_drift_OOS, \
    drifttrack_cummse, driftevalexp_mse_ignore_nans, MLP_fBiPotDDims_drifts, drifttrack_mse


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
        self.curr_best_track_mse = np.inf
        self.curr_best_evalexp_mse = np.inf

        self.diffusion = diffusion
        self.train_eps = train_eps
        self.max_diff_steps = max_diff_steps
        self.end_diff_time = end_diff_time
        self.is_hybrid = hybrid_training
        self.include_weightings = to_weight
        self.var_loss_reg = 1e-8
        self.mean_loss_reg = 1e-8
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

    def _batch_update(self, loss, base_loss: float, var_loss: float, mean_loss:float) -> [float, float, float, float]:
        """
            Backward pass and optimiser update step
                :param loss: loss tensor / function output
                :return: Batch Loss
            """
        loss.backward()  # single gpu functionality
        self.opt.step()
        self.loss_aggregator.update(loss.detach().item())
        return loss.detach().item(), base_loss, var_loss, mean_loss

    def _batch_loss_compute(self, outputs: torch.Tensor, targets: torch.Tensor, epoch: int, batch_idx: int,
                            num_batches: int) -> [float, float, float, float]:
        """
        Computes loss and calls helper function to compute backward pass
            :param outputs: Model forward pass output
            :param targets: Target values to compare against outputs
            :return: Batch Loss
        """
        base_loss = self.loss_fn()(outputs, targets) # Penalise for higher dimensions
        print(f"Loss, {base_loss}\n")
        var_loss = ((self.score_network.module.mlp_state_mapper.hybrid.log_scale - self.score_network.module.mlp_state_mapper.hybrid.log_scale.mean()) ** 2).mean()
        mean_loss = (torch.mean((self.score_network.module.mlp_state_mapper.hybrid.log_scale - 0.)**2))
        reg_var_loss = self.var_loss_reg * var_loss
        reg_mean_loss = self.mean_loss_reg * mean_loss
        loss = base_loss + reg_var_loss + reg_mean_loss
        print(f"VarReg, VarLoss, RegVarLoss, {self.var_loss_reg, var_loss, reg_var_loss}\n")
        print(f"MeanReg, MeanLoss, RegMeanLoss, {self.mean_loss_reg, mean_loss, reg_mean_loss}\n")
        return self._batch_update(loss, base_loss=base_loss.detach().item(), var_loss=var_loss.detach().item(), mean_loss=mean_loss.detach().item())

    def _run_batch(self, xts: torch.Tensor, features: torch.Tensor, stable_targets: torch.Tensor,
                   diff_times: torch.Tensor,
                   eff_times: torch.Tensor, epoch: int, batch_idx: int, num_batches: int) -> [float, float, float, float]:
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
        features = features.reshape(B * T, 1, D)
        stable_targets = stable_targets.reshape(B * T, 1, -1)
        diff_times = diff_times.reshape(B * T)
        eff_times = torch.cat([eff_times] * D, dim=2).reshape(stable_targets.shape)
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
        return self._batch_loss_compute(outputs=outputs * weights, targets=stable_targets * weights, epoch=epoch,
                                        batch_idx=batch_idx, num_batches=num_batches)

    def _compute_stable_targets(self, batch: torch.Tensor, noised_z: torch.Tensor, eff_times: torch.Tensor,
                                ref_batch: torch.Tensor, chunk_size: int, feat_thresh: float):
        import math, time
        t0 = time.time()

        B1, T, D = batch.shape
        B2, T2, D2 = ref_batch.shape
        assert T2 == T and D2 == D
        assert B2 > B1

        # dX = feat_thresh

        # positions (X) and increments (Z)
        pos_ref_batch = self._from_incs_to_positions(batch=ref_batch)[:, :-1, :]  # [B2, T, D]
        pos_batch = self._from_incs_to_positions(batch=batch)[:, :-1, :]  # [B1, T, D]
        assert pos_batch.shape == (B1, T, D)

        pos_ref_batch = pos_ref_batch.reshape(-1, D)  # [B2*T, D]
        pos_batch = pos_batch.reshape(-1, D)  # [B1*T, D]
        ref_batch = ref_batch.reshape(-1, D)  # [B2*T, D]

        eff_times = eff_times.reshape(-1, eff_times.shape[-1])
        if eff_times.shape[-1] == 1 and D > 1:
            eff_times = eff_times.expand(-1, D)
        assert eff_times.shape == (B1 * T, D)

        target_x = pos_batch  # [B1*T, D]
        target_x_exp = target_x.unsqueeze(1)  # [B1*T, 1, D]
        candidate_x = pos_ref_batch.unsqueeze(0)  # [1, B2*T, D]
        candidate_Z = ref_batch.unsqueeze(0)  # [1, B2*T, D]

        noised_z = noised_z.reshape(-1, D)  # [B1*T, D]
        beta_tau = torch.exp(-0.5 * eff_times)  # [B1*T, D]
        sigma_tau = 1. - torch.exp(-eff_times)  # [B1*T, D]

        target_noised_z = noised_z.unsqueeze(1)  # [B1*T, 1, D]
        target_beta_tau = beta_tau.unsqueeze(1)  # [B1*T, 1, D]
        target_sigma_tau = sigma_tau.unsqueeze(1)  # [B1*T, 1, D]

        stable_targets_chunks = []
        stable_targets_masks = []

        for i in range(0, target_x_exp.shape[0], chunk_size):
            i_end = min(i + chunk_size, target_x_exp.shape[0])

            target_chunk = target_x_exp[i:i_end]  # [chunk, 1, D]
            noised_z_chunk = target_noised_z[i:i_end]  # [chunk, 1, D]
            beta_tau_chunk = target_beta_tau[i:i_end]  # [chunk, 1, D]
            sigma_tau_chunk = target_sigma_tau[i:i_end]  # [chunk, 1, D]

            # ---- kNN in X (dimension-stable) ----
            N = candidate_x.shape[1]  # B2*T
            chunk = target_chunk.shape[0]
            dists = torch.cdist(target_chunk.squeeze(1),  # [chunk, D]
                                candidate_x.squeeze(0))  # [N, D] -> [chunk, N]
            dists = dists / math.sqrt(D)

            k = min(512, N)
            vals, idx = torch.topk(dists, k, dim=1, largest=False)  # [chunk, k]
            idx_exp = idx.unsqueeze(-1).expand(-1, -1, D)  # [chunk, k, D]

            cand_x_k = candidate_x.expand(chunk, -1, -1).gather(1, idx_exp)  # [chunk, k, D]
            cand_Z_k = candidate_Z.expand(chunk, -1, -1).gather(1, idx_exp)  # [chunk, k, D]

            # ---- soft kernel in X with adaptive bandwidth (k-th NN) ----
            h = vals[:, -1:].clamp_min(1e-12)  # [chunk, 1]
            Kx = torch.exp(-0.5 * (vals / h) ** 2).unsqueeze(-1)  # [chunk, k, 1]

            # ---- exact Gaussian p(Z_tau | Z0) ----
            nz = noised_z_chunk.expand(-1, k, -1)  # [chunk, k, D]
            mean_k = beta_tau_chunk.expand_as(nz) * cand_Z_k  # [chunk, k, D]
            std_k = torch.sqrt(sigma_tau_chunk.expand_as(nz))  # [chunk, k, D]

            logw = -0.5 * ((nz - mean_k) / std_k).pow(2) - torch.log(std_k) - 0.5 * math.log(2 * math.pi)
            logw = logw.sum(dim=-1, keepdim=True)  # [chunk, k, 1]

            # finite-sample guard against spikes
            q = torch.quantile(logw.detach(), 0.995)
            weights = torch.exp(torch.clamp(logw, max=q)) * Kx  # [chunk, k, 1]

            # ---- aggregate (NW mean; swap for LLR if desired) ----
            num = (weights * cand_Z_k).sum(dim=1)  # [chunk, D]
            den = weights.sum(dim=1).clamp_min(1e-12)  # [chunk, 1]
            stable_targets_chunk = num / den  # [chunk, D]
            stable_targets_chunks.append(stable_targets_chunk)

            # monitor ESS
            ess = (weights.sum(dim=1, keepdim=True).pow(2) /
                   (weights.pow(2).sum(dim=1, keepdim=True) + 1e-12)).squeeze(-1)  # [chunk, 1]
            stable_targets_masks.append(ess.to("cpu"))

        stable_targets_masks = torch.cat(stable_targets_masks, dim=0)  # [B1*T, 1]
        print(
            f"IQR ESS: {torch.quantile(stable_targets_masks, 0.005).item(), torch.quantile(stable_targets_masks, 0.995).item()}")

        stable_targets = torch.cat(stable_targets_chunks, dim=0)  # [B1*T, D]
        assert stable_targets.shape == (B1 * T, D)
        return stable_targets.to(self.device_id)

    def _run_epoch(self, epoch: int, batch_size: int, chunk_size: int, feat_thresh: float) -> [list, list, list, list]:
        """
        Single epoch run
            :param epoch: Epoch index
            :return: List of batch Losses
        """
        device_epoch_losses = []
        device_epoch_base_losses = []
        device_epoch_var_losses = []
        device_epoch_mean_losses = []
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
            stable_targets = self._compute_stable_targets(batch=x0s, noised_z=xts, ref_batch=ref_x0s,
                                                          eff_times=eff_times, chunk_size=chunk_size,
                                                          feat_thresh=feat_thresh)

            batch_loss, batch_base_loss, batch_var_loss, batch_mean_loss = self._run_batch(xts=xts, features=features,
                                                                          stable_targets=stable_targets,
                                                                          diff_times=diff_times,
                                                                          eff_times=eff_times, epoch=epoch,
                                                                          batch_idx=batch_idx,
                                                                          num_batches=len(self.train_loader))
            device_epoch_losses.append(batch_loss)
            device_epoch_base_losses.append(batch_base_loss)
            device_epoch_var_losses.append(batch_var_loss)
            device_epoch_mean_losses.append(batch_mean_loss)
        return device_epoch_losses, device_epoch_base_losses, device_epoch_var_losses, device_epoch_mean_losses

    def _load_snapshot(self, snapshot_path: str, config) -> None:
        """
        Load training from most recent snapshot
            :param snapshot_path: Path to training snapshot
            :return: None
        """
        # Snapshot should be python dict
        for param_group in self.opt.param_groups:
            param_group['lr'] = 1e-3
        print(
            f"Before loading snapshot Epochs Run, EWMA Loss, LR: {self.epochs_run, self.ewma_loss, self.opt.param_groups[0]['lr']}\n")

        loc = 'cuda:{}'.format(self.device_id) if type(self.device_id) == int else self.device_id
        try:
            snapshot = torch.load(snapshot_path, map_location=loc)
            self.epochs_run = snapshot["EPOCHS_RUN"]
            self.opt.load_state_dict(snapshot["OPTIMISER_STATE"])
            # Here to manually change the LR
            # if "BiPot" in config.data_path and self.opt.param_groups[0]["lr"] == 0.001 and config.feat_thresh == 1./50 and 550<= self.epochs_run <= 600: self.opt.param_groups[0]["lr"]=0.0001
            # if "QuadSin" in config.data_path and config.sin_space_scale == 4. and self.opt.param_groups[0]["lr"] == 0.001 and config.feat_thresh == 1./50 and 630<= self.epochs_run <= 640 : self.opt.param_groups[0]["lr"]=0.0001
            # if "QuadSin" in config.data_path and config.sin_space_scale == 25. and self.opt.param_groups[0]["lr"] == 0.001 and config.feat_thresh == 1./50. and 100<= self.epochs_run <= 120 : self.opt.param_groups[0]["lr"]=0.0001

            try:
                self.ewma_loss = snapshot["EWMA_LOSS"]
            except KeyError as e:
                print(e)
                pass
            try:
                self.var_loss_reg = snapshot["VAR_REG"]
            except KeyError as e:
                print(e)
                pass
            try:
                self.mean_loss_reg = snapshot["MEAN_REG"]
            except KeyError as e:
                print(e)
                pass
            try:
                self.curr_best_evalexp_mse = snapshot["EVALEXP_MSE"]
            except KeyError as e:
                print(e)
                pass
            try:
               self.curr_best_track_mse = snapshot["TRACK_MSE"]
            except KeyError as e:
                print(e)
                pass

            if type(self.device_id) == int:
                self.score_network.module.load_state_dict(snapshot["MODEL_STATE"])
            else:
                self.score_network.load_state_dict(snapshot["MODEL_STATE"])
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
        print(
            f"After loading snapshot Epochs Run, EWMA Loss, LR: {self.epochs_run, self.ewma_loss, self.opt.param_groups[0]['lr']}\n")

    def _save_snapshot(self, epoch: int) -> None:
        """
        Save current state of training
            :param epoch: Current epoch number
            :return: None
        """
        try:
            snapshot = {"EPOCHS_RUN": epoch + 1, "OPTIMISER_STATE": self.opt.state_dict(),
                        "SCHEDULER_STATE": self.scheduler.state_dict(), "EWMA_LOSS": self.ewma_loss, "VAR_REG": self.var_loss_reg, "MEAN_REG": self.mean_loss_reg, "TRACK_MSE": self.curr_best_track_mse, "EVALEXP_MSE":self.curr_best_evalexp_mse}
        except AttributeError as e:
            print(e)
            snapshot = {"EPOCHS_RUN": epoch + 1, "OPTIMISER_STATE": self.opt.state_dict(), "EWMA_LOSS": self.ewma_loss, "VAR_REG": self.var_loss_reg, "MEAN_REG": self.mean_loss_reg}

        # self.score_network now points to DDP wrapped object, so we need to access parameters via ".module"
        if type(self.device_id) == int:
            snapshot["MODEL_STATE"] = self.score_network.module.state_dict()
        else:
            snapshot["MODEL_STATE"] = self.score_network.state_dict()
        torch.save(snapshot, self.snapshot_path)
        print(f"Epoch {epoch + 1} | Training snapshot saved at {self.snapshot_path}\n")

    def _save_model(self, filepath: str, final_epoch: int, save_type: str) -> None:
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
        patterns = [f"{filepath}_{save_type}NEp*", f"{filepath}_TrackBestNEp*", f"{filepath}_TrackBNEp*"]
        for pattern in patterns:
            matching_files = glob.glob(pattern)
            for file in matching_files:
                if os.path.isfile(file):
                    print(f"Deleting: {file}")
                    os.remove(file)
        filepath = filepath + f"_{save_type}NEp{final_epoch}"
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
        dbatch[:, 0, :] += 1e-3 * torch.randn((dbatch.shape[0], dbatch.shape[-1])).to(dbatch.device)
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
        if len(l) > len(learning_rates) and len(learning_rates) == 0:  # Issue due to unsaved learning rates
            learning_rates = [self.opt.param_groups[0]["lr"]] * len(l)
        assert (len(learning_rates) >= self.epochs_run)
        return l, learning_rates

    def _domain_rmse(self, epoch, config):
        #assert (config.ndims <= 2)
        if config.ndims > 1 and "BiPot" in config.data_path:
            final_vec_mu_hats = MLP_fBiPotDDims_drifts(PM=self.score_network.module, config=config)
        else:
            final_vec_mu_hats = MLP_1D_drifts(PM=self.score_network.module, config=config)
        if "BiPot" in config.data_path and config.ndims == 1:
            Xs = np.linspace(-1.5, 1.5, num=config.ts_length)
            true_drifts = -(4. * config.quartic_coeff * np.power(Xs,
                                                                 3) + 2. * config.quad_coeff * Xs + config.const)
        elif "BiPot" in config.data_path and config.ndims > 1:
            Xshape = 256
            if config.ndims == 12:
                Xs = np.concatenate([np.linspace(-5, 5, num=Xshape).reshape(-1,1), np.linspace(-4.7, 4.7, num=Xshape).reshape(-1,1), \
                                     np.linspace(-4.4, 4.4, num=Xshape).reshape(-1,1), np.linspace(-4.2, 4.2, num=Xshape).reshape(-1,1), \
                                     np.linspace(-4.05, 4.05, num=Xshape).reshape(-1,1), np.linspace(-3.9, 3.9, num=Xshape).reshape(-1,1), \
                                     np.linspace(-3.7, 3.7, num=Xshape).reshape(-1,1), np.linspace(-3.6, 3.6, num=Xshape).reshape(-1,1), \
                                     np.linspace(-3.55, 3.55, num=Xshape).reshape(-1,1),
                                     np.linspace(-3.48, 3.48, num=Xshape).reshape(-1,1), \
                                     np.linspace(-3.4, 3.4, num=Xshape).reshape(-1,1), np.linspace(-3.4, 3.4, num=Xshape).reshape(-1,1)],
                                    axis=1)
            elif config.ndims == 8:
                Xs = np.concatenate([np.linspace(-4.9, 4.9, num=Xshape).reshape(-1, 1), np.linspace(-4.4, 4.4, num=Xshape).reshape(-1,1), \
                                     np.linspace(-4.05, 4.05, num=Xshape).reshape(-1,1), np.linspace(-3.9, 3.9, num=Xshape).reshape(-1,1), \
                                     np.linspace(-3.7, 3.7, num=Xshape).reshape(-1,1), np.linspace(-3.6, 3.6, num=Xshape).reshape(-1,1), \
                                     np.linspace(-3.5, 3.5, num=Xshape).reshape(-1,1), np.linspace(-3.4, 3.4, num=Xshape).reshape(-1,1)],
                                    axis=1)
            if "coup" in config.data_path:
                true_drifts = -(4. * np.array(config.quartic_coeff) * np.power(Xs,
                                                                               3) + 2. * np.array(
                    config.quad_coeff) * Xs + np.array(config.const))
                xstar = np.sqrt(
                    np.maximum(1e-12, -np.array(config.quad_coeff) / (2.0 * np.array(config.quartic_coeff))))
                s2 = (config.scale * xstar) ** 2 + 1e-12  # (D,) or (K,1,D)
                diff = Xs ** 2 - xstar ** 2  # same shape as prev
                phi = np.exp(-(diff ** 2) / (2.0 * s2 * xstar ** 2 + 1e-12))
                phi_prime = phi * (-2.0 * Xs * diff / ((config.scale ** 2) * (xstar ** 4 + 1e-12)))
                nbr = np.roll(phi, 1, axis=-1) + np.roll(phi, -1, axis=-1)  # same shape as phi
                true_drifts = true_drifts - 0.5 * config.coupling * phi_prime * nbr
                assert true_drifts.shape == Xs.shape
            else:
                true_drifts = -(4. * np.array(config.quartic_coeff) * np.power(Xs,
                                                                           3) + 2. * np.array(
                    config.quad_coeff) * Xs + np.array(config.const))
        elif "QuadSin" in config.data_path:
            Xs = np.linspace(-1.5, 1.5, num=config.ts_length)
            true_drifts = (-2. * config.quad_coeff * Xs + config.sin_coeff * config.sin_space_scale * np.sin(
                config.sin_space_scale * Xs))
        elif "SinLog" in config.data_path:
            Xs = np.linspace(-1.5, 1.5, num=config.ts_length)
            true_drifts = (-np.sin(config.sin_space_scale * Xs) * np.log(
                1 + config.log_space_scale * np.abs(Xs)) / config.sin_space_scale)
        type = "PM"
        assert (type in config.scoreNet_trained_path)
        assert ("_ST_" in config.scoreNet_trained_path)
        enforce_fourier_reg = "NFMReg_" if not config.enforce_fourier_mean_reg else ""
        enforce_fourier_reg += "New_" if "New" in config.scoreNet_trained_path else ""
        if "BiPot" in config.data_path and config.ndims == 1:
            save_path = (
                    project_config.ROOT_DIR + f"experiments/results/TSPM_MLP_ST_{config.feat_thresh:.3f}FTh_{enforce_fourier_reg}fBiPot_DriftEvalExp_{epoch}Nep_{config.t0}t0_{config.deltaT:.3e}dT_{config.quartic_coeff}a_{config.quad_coeff}b_{config.const}c_{config.residual_layers}ResLay_{config.loss_factor}LFac_BetaMax{config.beta_max:.1e}").replace(
                ".", "")
        elif "BiPot" in config.data_path and config.ndims > 1 and "coup" not in config.data_path:
            save_path = (
                    project_config.ROOT_DIR + f"experiments/results/TSPM_MLP_ST_{config.feat_thresh:.3f}FTh_{enforce_fourier_reg}fBiPot_{config.ndims}DDims_DriftEvalExp_{epoch}Nep_{config.t0}t0_{config.deltaT:.3e}dT_{config.quartic_coeff[0]}a_{config.quad_coeff[0]}b_{config.const[0]}c_{config.residual_layers}ResLay_{config.loss_factor}LFac_BetaMax{config.beta_max:.1e}").replace(
                ".", "")
        elif "BiPot" in config.data_path and "coup" in config.data_path:
            save_path = (
                    project_config.ROOT_DIR + f"experiments/results/TSPM_MLP_ST_{config.feat_thresh:.3f}FTh_{enforce_fourier_reg}fBiPot_{config.ndims}DDimsNS_DriftEvalExp_{epoch}Nep_{config.t0}t0_{config.deltaT:.3e}dT_{config.quartic_coeff[0]}a_{config.quad_coeff[0]}b_{config.const[0]}c_{config.residual_layers}ResLay_{config.loss_factor}LFac_BetaMax{config.beta_max:.1e}").replace(
                ".", "")
        elif "QuadSin" in config.data_path:
            save_path = (
                    project_config.ROOT_DIR + f"experiments/results/TSPM_MLP_ST_{config.feat_thresh:.3f}FTh_{enforce_fourier_reg}fQuadSinHF_DriftEvalExp_{epoch}Nep_{config.t0}t0_{config.deltaT:.3e}dT_{config.quad_coeff}a_{config.sin_coeff}b_{config.sin_space_scale}c_{config.residual_layers}ResLay_{config.loss_factor}LFac_BetaMax{config.beta_max:.1e}").replace(
                ".", "")
        elif "SinLog" in config.data_path:
            save_path = (
                    project_config.ROOT_DIR + f"experiments/results/TSPM_MLP_ST_{config.feat_thresh:.3f}FTh_{enforce_fourier_reg}fSinLog_DriftEvalExp_{epoch}Nep_{config.t0}t0_{config.deltaT:.3e}dT_{config.log_space_scale}b_{config.sin_space_scale}c_{config.residual_layers}ResLay_{config.loss_factor}LFac_BetaMax{config.beta_max:.1e}").replace(
                ".", "")
        elif "MullerBrown" in config.data_path:
            save_path = (
                    project_config.ROOT_DIR + f"experiments/results/TSPM_MLP_ST_{config.feat_thresh:.3f}FTh_{enforce_fourier_reg}fMullerBrown_DriftEvalExp_{epoch}Nep_{config.t0}t0_{config.deltaT:.3e}dT_{config.residual_layers}ResLay_{config.loss_factor}LFac_BetaMax{config.beta_max:.1e}").replace(
                ".", "")
        print(f"Save path:{save_path}\n")
        np.save(save_path + "_muhats.npy", final_vec_mu_hats)
        self.score_network.module.train()
        self.score_network.module.to(self.device_id)
        if len(true_drifts.shape) == 1:
            true_drifts = true_drifts.reshape(-1,1)
        mse = driftevalexp_mse_ignore_nans(true=true_drifts, pred=final_vec_mu_hats[:, -1, :,:].reshape(final_vec_mu_hats.shape[0], final_vec_mu_hats.shape[2], final_vec_mu_hats.shape[-1]*1).mean(axis=1))
        print(f"Current vs Best MSE {mse}, {self.curr_best_evalexp_mse} at Epoch {epoch}\n")
        return mse

    def _tracking_errors(self, epoch, config):
        def true_drift(prev, num_paths, config):
            assert (prev.shape == (num_paths, config.ndims))
            if "BiPot" in config.data_path and config.ndims==1:
                drift_X = -(4. * config.quartic_coeff * np.power(prev,
                                                                 3) + 2. * config.quad_coeff * prev + config.const)
                return drift_X[:, np.newaxis, :]
            elif "BiPot" in config.data_path and config.ndims>1 and "coup" not in config.data_path:
                drift_X = -(4. * np.array(config.quartic_coeff) * np.power(prev,
                                                                 3) + 2. * np.array(config.quad_coeff) * prev + np.array(config.const))
                return drift_X[:, np.newaxis, :]
            elif "BiPot" in config.data_path and config.ndims>1 and "coup" in config.data_path:
                drift_X = -(4. * np.array(config.quartic_coeff) * np.power(prev,
                                                                 3) + 2. * np.array(config.quad_coeff) * prev + np.array(config.const))
                xstar = np.sqrt(
                    np.maximum(1e-12, -np.array(config.quad_coeff) / (2.0 * np.array(config.quartic_coeff))))
                s2 = (config.scale * xstar) ** 2 + 1e-12  # (D,) or (K,1,D)
                diff = prev ** 2 - xstar ** 2  # same shape as prev
                phi = np.exp(-(diff ** 2) / (2.0 * s2 * xstar ** 2 + 1e-12))
                phi_prime = phi * (-2.0 * prev * diff / ((config.scale ** 2) * (xstar ** 4 + 1e-12)))
                nbr = np.roll(phi, 1, axis=-1) + np.roll(phi, -1, axis=-1)  # same shape as phi
                drift_X = drift_X - 0.5 * config.coupling * phi_prime * nbr
                return drift_X[:, np.newaxis, :]
            elif "QuadSin" in config.data_path:
                drift_X = -2. * config.quad_coeff * prev + config.sin_coeff * config.sin_space_scale * np.sin(
                    config.sin_space_scale * prev)
                return drift_X[:, np.newaxis, :]
            elif "SinLog" in config.data_path:
                drift_X = -np.sin(config.sin_space_scale * prev) * np.log(
                    1 + config.log_space_scale * np.abs(prev)) / config.sin_space_scale
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
        rmse_quantile_nums = 2
        num_paths = 100
        num_time_steps = 256
        all_true_states = np.zeros(shape=(rmse_quantile_nums, num_paths, 1 + num_time_steps, config.ndims))
        all_global_states = np.zeros(shape=(rmse_quantile_nums, num_paths, 1 + num_time_steps, config.ndims))
        all_local_states = np.zeros(shape=(rmse_quantile_nums, num_paths, 1 + num_time_steps, config.ndims))
        for quant_idx in tqdm(range(rmse_quantile_nums)):
            self.score_network.module.eval()
            num_paths = 100
            num_time_steps = 256
            deltaT = config.deltaT
            initial_state = np.repeat(np.atleast_2d(config.initState)[np.newaxis, :], num_paths, axis=0)
            assert (initial_state.shape == (num_paths, 1, config.ndims))

            true_states = np.zeros(shape=(num_paths, 1 + num_time_steps, config.ndims))
            global_states = np.zeros(shape=(num_paths, 1 + num_time_steps, config.ndims))
            local_states = np.zeros(shape=(num_paths, 1 + num_time_steps, config.ndims))

            # Initialise the "true paths"
            true_states[:, [0], :] = initial_state + 0.00001 * np.random.randn(*initial_state.shape)
            # Initialise the "global score-based drift paths"
            global_states[:, [0], :] = true_states[:, [0], :]
            local_states[:, [0], :] = true_states[:, [0], :]  # np.repeat(initial_state[np.newaxis, :], num_diff_times, axis=0)

            # Euler-Maruyama Scheme for Tracking Errors
            for i in range(1, num_time_steps + 1):
                eps = np.random.randn(num_paths, 1, config.ndims) * np.sqrt(deltaT)*config.diffusion
                assert (eps.shape == (num_paths, 1, config.ndims))
                true_mean = true_drift(true_states[:, i - 1, :], num_paths=num_paths, config=config)

                true_states[:, [i], :] = true_states[:, [i - 1], :] \
                                         + true_mean * deltaT \
                                         + eps
                local_mean = multivar_score_based_MLP_drift_OOS(
                    score_model=self.score_network.module,
                   num_diff_times=num_diff_times,
                    diffusion=diffusion,
                    num_paths=num_paths, ts_step=deltaT,
                    config=config,
                    device=self.device_id,
                    prev=true_states[:, i - 1, :])

                global_mean = multivar_score_based_MLP_drift_OOS(score_model=self.score_network.module,
                                                                                      num_diff_times=num_diff_times,
                                                                                      diffusion=diffusion,
                                                                                      num_paths=num_paths,
                                                                                      ts_step=deltaT, config=config,
                                                                                      device=self.device_id,
                                                                                      prev=global_states[:, i - 1, :])

                local_states[:, [i], :] = true_states[:, [i - 1], :] + local_mean * deltaT + eps
                global_states[:, [i], :] = global_states[:, [i - 1], :] + global_mean * deltaT + eps

            all_true_states[quant_idx, :, :, :] = true_states
            all_local_states[quant_idx, :, :, :] = local_states
            all_global_states[quant_idx, :, :, :] = global_states
        enforce_fourier_reg = "NFMReg_" if not config.enforce_fourier_mean_reg else ""
        enforce_fourier_reg += "New_" if "New" in config.scoreNet_trained_path else ""
        if "BiPot" in config.data_path and config.ndims == 1:
            save_path = (
                    project_config.ROOT_DIR + f"experiments/results/TSPM_MLP_ST_{config.feat_thresh:.3f}FTh_{enforce_fourier_reg}fBiPot_OOSDriftTrack_{epoch}Nep_{config.t0}t0_{config.deltaT:.3e}dT_{config.quartic_coeff}a_{config.quad_coeff}b_{config.const}c_{config.residual_layers}ResLay_{config.loss_factor}LFac_BetaMax{config.beta_max:.1e}").replace(
                ".", "")
        elif "BiPot" in config.data_path and config.ndims > 1 and "coup" not in config.data_path:
            save_path = (
                    project_config.ROOT_DIR + f"experiments/results/TSPM_MLP_ST_{config.feat_thresh:.3f}FTh_{enforce_fourier_reg}fBiPot_{config.ndims}DDims_OOSDriftTrack_{epoch}Nep_{config.t0}t0_{config.deltaT:.3e}dT_{config.quartic_coeff[0]}a_{config.quad_coeff[0]}b_{config.const[0]}c_{config.residual_layers}ResLay_{config.loss_factor}LFac_BetaMax{config.beta_max:.1e}").replace(
                ".", "")
        elif "BiPot" in config.data_path and config.ndims > 1 and "coup" in config.data_path:
            save_path = (
                    project_config.ROOT_DIR + f"experiments/results/TSPM_MLP_ST_{config.feat_thresh:.3f}FTh_{enforce_fourier_reg}fBiPot_{config.ndims}DDimsNS_OOSDriftTrack_{epoch}Nep_{config.t0}t0_{config.deltaT:.3e}dT_{config.quartic_coeff[0]}a_{config.quad_coeff[0]}b_{config.const[0]}c_{config.residual_layers}ResLay_{config.loss_factor}LFac_BetaMax{config.beta_max:.1e}").replace(
                ".", "")
        elif "QuadSin" in config.data_path:
            save_path = (
                    project_config.ROOT_DIR + f"experiments/results/TSPM_MLP_ST_{config.feat_thresh:.3f}FTh_{enforce_fourier_reg}fQuadSinHF_OOSDriftTrack_{epoch}Nep_{config.t0}t0_{config.deltaT:.3e}dT_{config.quad_coeff}a_{config.sin_coeff}b_{config.sin_space_scale}c_{config.residual_layers}ResLay_{config.loss_factor}LFac_BetaMax{config.beta_max:.1e}").replace(
                ".", "")
        elif "SinLog" in config.data_path:
            save_path = (
                    project_config.ROOT_DIR + f"experiments/results/TSPM_MLP_ST_{config.feat_thresh:.3f}FTh_{enforce_fourier_reg}fSinLog_OOSDriftTrack_{epoch}Nep_{config.t0}t0_{config.deltaT:.3e}dT_{config.log_space_scale}b_{config.sin_space_scale}c_{config.residual_layers}ResLay_{config.loss_factor}LFac_BetaMax{config.beta_max:.1e}").replace(
                ".", "")
        elif "MullerBrown" in config.data_path:
            save_path = (
                    project_config.ROOT_DIR + f"experiments/results/TSPM_MLP_ST_{config.feat_thresh:.3f}FTh_{enforce_fourier_reg}fMullerBrown_OOSDriftTrack_{epoch}Nep_{config.t0}t0_{config.deltaT:.3e}dT_{config.residual_layers}ResLay_{config.loss_factor}LFac_BetaMax{config.beta_max:.1e}").replace(
                ".", "")
        elif "Lnz" in config.data_path:
            save_path = (
                        project_config.ROOT_DIR + f"experiments/results/TSPM_MLP_ST_{config.feat_thresh:.3f}FTh_{enforce_fourier_reg}{config.ndims}DLnz_OOSDriftTrack_{epoch}Nep_tl{config.tdata_mult}data_{config.t0}t0_{config.deltaT:.3e}dT_{num_diff_times}NDT_{config.loss_factor}LFac_BetaMax{config.beta_max:.1e}_{round(config.forcing_const, 3)}FConst").replace(
                    ".", "")
        print(f"Save path for OOS DriftTrack:{save_path}\n")
        np.save(save_path + "_true_states.npy", all_true_states)
        np.save(save_path + "_global_states.npy", all_global_states)
        np.save(save_path + "_local_states.npy", all_local_states)
        self.score_network.module.train()
        self.score_network.module.to(self.device_id)
        #mse = drifttrack_cummse(true=all_true_states, local=all_local_states, deltaT=config.deltaT)
        mse = drifttrack_mse(true=all_true_states, local=all_global_states, deltaT=config.deltaT)
        print(f"Current vs Best MSE {mse}, {self.curr_best_track_mse} at Epoch {epoch}\n")
        return mse

    def train(self, max_epochs: list, model_filename: str, batch_size: int, config) -> None:
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
        self.ewma_loss = 0.  # Force recomputation of EWMA losses each time
        #self.curr_best_track_mse = np.inf # Force recomputation once
        for epoch in range(self.epochs_run, end_epoch):
            t0 = time.time()
            # Temperature annealing for gumbel softmax
            if epoch % 20 == 0:
                tau = max(self.score_network.module.mlp_state_mapper.hybrid.final_tau,
                          self.score_network.module.mlp_state_mapper.hybrid.init_tau * (0.9 ** (epoch // 20)))
                self.score_network.module.mlp_state_mapper.hybrid.set_tau(tau)
            device_epoch_losses, device_epoch_base_losses, device_epoch_var_losses,device_epoch_mean_losses  = self._run_epoch(epoch=epoch,
                                                                                                     batch_size=batch_size,
                                                                                                     chunk_size=config.chunk_size,
                                                                                                     feat_thresh=config.feat_thresh)
            # Average epoch loss for each device over batches
            epoch_losses_tensor = torch.tensor(torch.mean(torch.tensor(device_epoch_losses)).item())
            epoch_base_losses_tensor = torch.tensor(torch.mean(torch.tensor(device_epoch_base_losses)).item())
            epoch_var_losses_tensor = torch.tensor(torch.mean(torch.tensor(device_epoch_var_losses)).item())
            epoch_mean_losses_tensor = torch.tensor(torch.mean(torch.tensor(device_epoch_mean_losses)).item())

            if type(self.device_id) == int:
                epoch_losses_tensor = epoch_losses_tensor.cuda()
                all_gpus_losses = [torch.zeros_like(epoch_losses_tensor) for _ in range(torch.cuda.device_count())]
                torch.distributed.all_gather(all_gpus_losses, epoch_losses_tensor)

                epoch_base_losses_tensor = epoch_base_losses_tensor.cuda()
                all_gpus_base_losses = [torch.zeros_like(epoch_base_losses_tensor) for _ in
                                        range(torch.cuda.device_count())]
                torch.distributed.all_gather(all_gpus_base_losses, epoch_base_losses_tensor)

                epoch_var_losses_tensor = epoch_var_losses_tensor.cuda()
                all_gpus_var_losses = [torch.zeros_like(epoch_var_losses_tensor) for _ in
                                       range(torch.cuda.device_count())]
                torch.distributed.all_gather(all_gpus_var_losses, epoch_var_losses_tensor)

                epoch_mean_losses_tensor = epoch_mean_losses_tensor.cuda()
                all_gpus_mean_losses = [torch.zeros_like(epoch_mean_losses_tensor) for _ in
                                       range(torch.cuda.device_count())]
                torch.distributed.all_gather(all_gpus_mean_losses, epoch_mean_losses_tensor)
            else:
                all_gpus_losses = [epoch_losses_tensor]
                all_gpus_base_losses = [epoch_base_losses_tensor]
                all_gpus_var_losses = [epoch_var_losses_tensor]
                all_gpus_mean_losses = [epoch_mean_losses_tensor]

            # Obtain epoch loss averaged over devices
            average_loss_per_epoch = torch.mean(torch.stack(all_gpus_losses), dim=0)
            all_losses_per_epoch.append(float(average_loss_per_epoch.cpu().numpy()))
            if "New" not in config.scoreNet_trained_path:
                average_base_loss_per_epoch = float(torch.mean(torch.stack(all_gpus_base_losses), dim=0).cpu().numpy())
                average_var_loss_per_epoch = float(torch.mean(torch.stack(all_gpus_var_losses), dim=0).cpu().numpy())
                average_mean_loss_per_epoch = float(torch.mean(torch.stack(all_gpus_mean_losses), dim=0).cpu().numpy())
                ratio = (.99 * average_base_loss_per_epoch) / (average_var_loss_per_epoch + 1e-12)
                self.var_loss_reg = min(ratio, 0.01)

                if config.enforce_fourier_mean_reg:
                    ratio = (.75 * average_base_loss_per_epoch) / (average_mean_loss_per_epoch + 1e-12)
                    self.mean_loss_reg = min(ratio, 0.01/config.ts_dims) # vs 0.01
                else:
                    self.mean_loss_reg = 0.
                print(
                f"Calibrating Regulatisation: Base {average_base_loss_per_epoch}, Var {average_var_loss_per_epoch}, Mean {average_mean_loss_per_epoch}\n")
            else:
                self.mean_loss_reg = 0.
                self.var_loss_reg = 0.
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
                if ((epoch + 1) % self.save_every == 0) or epoch == 0:
                    if config.ndims <= 2 or ("BiPot" in config.data_path and config.ndims > 1):
                        evalexp_mse = self._domain_rmse(config=config, epoch=epoch + 1)
                        if evalexp_mse < self.curr_best_evalexp_mse and (epoch + 1) >= 10:
                            self._save_model(filepath=model_filename, final_epoch=epoch + 1, save_type="EE")
                            self.curr_best_evalexp_mse = evalexp_mse
                    track_mse = self._tracking_errors(epoch=epoch + 1, config=config)
                    if track_mse < self.curr_best_track_mse and (epoch + 1) >= 10:
                        self._save_model(filepath=model_filename, final_epoch=epoch + 1, save_type="")
                        self.curr_best_track_mse = track_mse
                    self._save_loss(losses=all_losses_per_epoch, learning_rates=learning_rates, filepath=model_filename)
                    self._save_snapshot(epoch=epoch)
            if type(self.device_id) == int: dist.barrier()
            print(f"TrackMSE, EvalExpMSE: {self.curr_best_track_mse, self.curr_best_evalexp_mse}\n")

