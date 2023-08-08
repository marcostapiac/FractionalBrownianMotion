def one_epoch_diffusion_train(self, opt: torch.optim.Optimizer,
                              trainLoader: torch.utils.data.DataLoader) -> float:
    mean_loss = MeanMetric()
    self.train()
    timesteps = torch.linspace(self.trainEps, end=1., steps=self.numDiffSteps)
    for x0s in iter(trainLoader):  # Iterate over batches (training data is already randomly selected)
        x0s = x0s[0].to(self.torchDevice)  # TODO: For some reason the original x0s is a list
        # Randomly sample uniform time integer and each time-element in seq_length is diffused by different time
        i_s = torch.randint(low=0, high=self.numDiffSteps, dtype=torch.int32, size=(x0s.shape[1],),
                            device=self.torchDevice)
        diffTimes = timesteps[i_s]
        effTimes = (0.5 * diffTimes ** 2 * (
                self.get_beta_max() - self.get_beta_min()) + diffTimes * self.get_beta_min())
        xts, true_score = self.forward_process(dataSamples=x0s, effTimes=effTimes)
        # A single batch of data should have dimensions [batch_size, sequence_length, time_series_dimension]
        rnn_outputs, _ = self.model.rnn(x0s.T)
        xts, effTimes, rnn_outputs = xts.unsqueeze(1).transpose(0, -1), effTimes, rnn_outputs.unsqueeze(1)
        pred = self.model.forward(xts, effTimes, cond=rnn_outputs)
        loss = self.training_loss_fn(
            weighted_predicted=self.get_loss_weighting(effTimes) * pred.squeeze(1).T,
            weighted_true=self.get_loss_weighting(effTimes) * true_score)
        opt.zero_grad()
        loss.backward()
        opt.step()
        mean_loss.update(loss.detach().item())
    return float(mean_loss.compute().item())


def evaluate_diffusion_model(self, loader: torch.utils.data.DataLoader) -> float:
    mean_loss = MeanMetric()
    timesteps = torch.linspace(self.trainEps, end=1., steps=self.numDiffSteps)
    for x0s in (iter(loader)):
        self.eval()
        with torch.no_grad():
            x0s = x0s[0].to(self.torchDevice)  # TODO: For some reason the original x0s is a list
            # Randomly sample uniform time integer and each time-element in seq_length is diffused by different time
            i_s = torch.randint(low=0, high=self.numDiffSteps, dtype=torch.int32, size=(x0s.shape[1],),
                                device=self.torchDevice)
            diffTimes = timesteps[i_s]
            effTimes = (0.5 * diffTimes ** 2 * (
                    self.get_beta_max() - self.get_beta_min()) + diffTimes * self.get_beta_min())
            xts, true_score = self.forward_process(dataSamples=x0s, effTimes=effTimes)
            # A single batch of data should have dimensions [batch_size, sequence_length, time_series_dimension]
            rnn_outputs, _ = self.model.rnn(x0s.T)
            xts, effTimes, rnn_outputs = xts.unsqueeze(1).transpose(0, -1), effTimes, rnn_outputs.unsqueeze(1)
            pred = self.model.forward(xts, effTimes, cond=rnn_outputs)
            loss = self.training_loss_fn(
                weighted_predicted=self.get_loss_weighting(effTimes) * pred.squeeze(1).T,
                weighted_true=self.get_loss_weighting(effTimes) * true_score)

            mean_loss.update(loss.detach().item())
    return float(mean_loss.compute().item())