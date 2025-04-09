import os
import pickle
import time
from typing import Union

from tqdm import tqdm
import torch
import torch.distributed as dist
import torchmetrics
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torchmetrics import MeanMetric
from configs import project_config
import numpy as np

from src.generative_modelling.models.ClassOUSDEDiffusion import OUSDEDiffusion
from src.generative_modelling.models.ClassVESDEDiffusion import VESDEDiffusion
from src.generative_modelling.models.ClassVPSDEDiffusion import VPSDEDiffusion
from src.generative_modelling.models.TimeDependentScoreNetworks.ClassConditionalLSTMTSPostMeanScoreMatching import \
    ConditionalLSTMTSPostMeanScoreMatching
from utils.drift_evaluation_functions import multivar_score_based_LSTM_drift_OOS, LSTM_2D_drifts, LSTM_1D_drifts


# Link for DDP vs DataParallelism: https://www.run.ai/guides/multi-gpu/pytorch-multi-gpu-4-techniques-explained
# Link for ddp_setup backend: https://pytorch.org/docs/stable/distributed.html
# Tutorial: https://www.youtube.com/watch?v=-LAtx9Q6DA8


class ConditionalLSTMPostMeanDiffTrainer(nn.Module):

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
        # self.opt.optimizer.step()
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
        # self.opt.optimizer.zero_grad()
        self.opt.zero_grad()
        B, T, D = xts.shape
        # Reshaping concatenates vectors in dim=1
        xts = xts.reshape(B * T, 1, -1)
        features = features.reshape(B * T, 1, -1)
        target_scores = target_scores.reshape(B * T, 1, -1)
        diff_times = diff_times.reshape(B * T)
        eff_times = torch.cat([eff_times] * D, dim=2).reshape(target_scores.shape)
        outputs = self.score_network.forward(inputs=xts, conditioner=features, times=diff_times, eff_times=eff_times)
        # For times larger than tau0, use inverse_weighting
        sigma_tau = 1. - torch.exp(-eff_times)
        beta_tau = torch.exp(-0.5 * eff_times)
        if not self.include_weightings:
            weights = torch.ones_like(eff_times)
        elif self.loss_factor == 0:  # PM
            weights = (sigma_tau / beta_tau)
        elif self.loss_factor == 1:  # PMScaled (meaning not scaled)
            weights = self.diffusion.get_loss_weighting(eff_times=eff_times)
        elif self.loss_factor == 2:  # PM with deltaT scaling
            weights = (sigma_tau / (beta_tau * torch.sqrt(self.deltaT)))
        # Outputs should be (NumBatches, TimeSeriesLength, 1)
        return self._batch_loss_compute(outputs=outputs * weights, targets=target_scores * weights)

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

    def _domain_rmse(self, epoch, config):
        assert config.ndims <= 2
        if "MullerBrown" in config.data_path:
            final_vec_mu_hats = LSTM_2D_drifts(PM=self.score_network.module, config=config)
        else:
            final_vec_mu_hats = LSTM_1D_drifts(PM=self.score_network.module, config=config)
        type = "PM"
        assert (type in config.scoreNet_trained_path)
        if "BiPot" in config.data_path:
            save_path = (
                        project_config.ROOT_DIR + f"experiments/results/TSPM_LSTM_fBiPot_DriftEvalExp_{epoch}Nep_{config.t0}t0_{config.deltaT:.3e}dT_{config.quartic_coeff}a_{config.quad_coeff}b_{config.const}c_{config.residual_layers}ResLay_{config.loss_factor}LFac_BetaMax{config.beta_max:.1e}").replace(
                    ".", "")
        elif "QuadSin" in config.data_path:
            save_path = (
                    project_config.ROOT_DIR + f"experiments/results/TSPM_LSTM_fQuadSinHF_DriftEvalExp_{epoch}Nep_{config.t0}t0_{config.deltaT:.3e}dT_{config.quad_coeff}a_{config.sin_coeff}b_{config.sin_space_scale}c_{config.residual_layers}ResLay_{config.loss_factor}LFac_BetaMax{config.beta_max:.1e}").replace(
                ".", "")
        elif "MullerBrown" in config.data_path:
            save_path = (
                    project_config.ROOT_DIR + f"experiments/results/TSPM_LSTM_fMullerBrown_DriftEvalExp_{epoch}Nep_{config.t0}t0_{config.deltaT:.3e}dT_{config.residual_layers}ResLay_{config.loss_factor}LFac_BetaMax{config.beta_max:.1e}").replace(
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
            # global_h, global_c = None, None
            local_h, local_c = None, None
            for i in range(1, num_time_steps + 1):
                eps = np.random.randn(num_paths, 1, config.ndims) * np.sqrt(deltaT)
                assert (eps.shape == (num_paths, 1, config.ndims))
                true_mean = true_drift(true_states[:, i - 1, :], num_paths=num_paths, config=config)

                true_states[:, [i], :] = true_states[:, [i - 1], :] \
                                         + true_mean * deltaT \
                                         + eps
                local_mean, local_h, local_c = multivar_score_based_LSTM_drift_OOS(
                    score_model=self.score_network.module, time_idx=i - 1,
                    h=local_h, c=local_c,
                    num_diff_times=num_diff_times,
                    diffusion=diffusion,
                    num_paths=num_paths, ts_step=deltaT,
                    config=config,
                    device=self.device_id,
                    prev=true_states[:, i - 1, :])

                local_states[:, [i], :] = true_states[:, [i - 1], :] + local_mean * deltaT + eps
            all_true_states[quant_idx, :, :, :] = true_states
            # all_global_states[quant_idx, :, :, :] = global_states
            all_local_states[quant_idx, :, :, :] = local_states
        if "BiPot" in config.data_path:
            save_path = (
                        project_config.ROOT_DIR + f"experiments/results/TSPM_LSTM_fBiPot_OOSDriftTrack_{epoch}Nep_{config.t0}t0_{config.deltaT:.3e}dT_{config.quartic_coeff}a_{config.quad_coeff}b_{config.const}c_{config.residual_layers}ResLay_{config.loss_factor}LFac_BetaMax{config.beta_max:.1e}").replace(
                    ".", "")
        elif "QuadSin" in config.data_path:
            save_path = (
                    project_config.ROOT_DIR + f"experiments/results/TSPM_LSTM_fQuadSinHF_OOSDriftTrack_{epoch}Nep_{config.t0}t0_{config.deltaT:.3e}dT_{config.quad_coeff}a_{config.sin_coeff}b_{config.sin_space_scale}c_{config.residual_layers}ResLay_{config.loss_factor}LFac_BetaMax{config.beta_max:.1e}").replace(
                ".", "")
        elif "MullerBrown" in config.data_path:
            save_path = (
                    project_config.ROOT_DIR + f"experiments/results/TSPM_LSTM_fMullerBrown_DriftTrack_{epoch}Nep_{config.t0}t0_{config.deltaT:.3e}dT_{config.residual_layers}ResLay_{config.loss_factor}LFac_BetaMax{config.beta_max:.1e}").replace(
                ".", "")
        elif "Lnz" in config.data_path:
            save_path = (
                    project_config.ROOT_DIR + f"experiments/results/TSPM_LSTM_{config.ndims}DLorenz_OOSDriftTrack_{epoch}Nep_tl{config.tdata_mult}data_{config.t0}t0_{config.deltaT:.3e}dT_{num_diff_times}NDT_{config.loss_factor}LFac_BetaMax{config.beta_max:.1e}_{round(config.forcing_const, 3)}FConst").replace(
                ".", "")
        print(f"Save path for OOS DriftTrack:{save_path}\n")
        np.save(save_path + "_true_states.npy", all_true_states)
        # np.save(save_path + "_global_states.npy", all_global_states)
        np.save(save_path + "_local_states.npy", all_local_states)
        self.score_network.module.train()
        self.score_network.module.to(self.device_id)

    def train(self, max_epochs: list, model_filename: str, config) -> None:
        """
        Run training for model
            :param max_epochs: List of maximum number of epochs (to allow for iterative training)
            :param model_filename: Filepath to save model
            :return: None
        """
        assert ("_ST_" not in config.scoreNet_trained_path)
        max_epochs = sorted(max_epochs)
        self.score_network.train()
        all_losses_per_epoch = self._load_loss_tracker(model_filename)  # This will contain synchronised losses
        end_epoch = max(max_epochs)
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
                print(f"Current Loss {float(torch.mean(torch.tensor(all_losses_per_epoch[-1])).cpu().numpy())}\n")
                if epoch + 1 in max_epochs:
                    self._save_snapshot(epoch=epoch)
                    self._save_loss(losses=all_losses_per_epoch, filepath=model_filename)
                    self._save_model(filepath=model_filename, final_epoch=epoch + 1)
                elif (epoch + 1) % self.save_every == 0:
                    self._save_loss(losses=all_losses_per_epoch, filepath=model_filename)
                    self._save_snapshot(epoch=epoch)
                    self._tracking_errors(epoch=epoch + 1, config=config)
                    if "Lnz" not in config.data_path:
                        self._domain_rmse(config=config, epoch=epoch + 1)
            if type(self.device_id) == int: dist.barrier()
