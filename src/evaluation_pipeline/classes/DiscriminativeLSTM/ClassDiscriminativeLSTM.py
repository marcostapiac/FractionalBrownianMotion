import torch


class DiscriminativeLSTM(torch.nn.Module):
    """ Naive LSTM for discriminating between real and fake time-series samples """

    def __init__(self, ts_dim: int):
        super().__init__()
        self.ts_dim = ts_dim
        self.lstm = torch.nn.LSTM(input_size=self.ts_dim, hidden_size=50, num_layers=2, batch_first=True)
        self.linear = torch.nn.Linear(in_features=50, out_features=self.ts_dim)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        assert (input.shape[-1] == self.ts_dim)
        x, _ = self.lstm(input)
        x = x[:, -1, :]  # For every batch and every "feature"/"dimension",
        # retrieve the last time step (1-step prediction) of the LSTM output
        x = self.linear(x)
        return x
