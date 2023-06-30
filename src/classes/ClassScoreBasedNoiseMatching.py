from torch import nn


class ConditionalBatchNorm1d(nn.Module):
    def __init__(self, num_features, num_classes, bias=True):
        super().__init__()
        self.num_features = num_features
        self.bias = bias
        self.bn = nn.BatchNorm1d(num_features, affine=False)
        if self.bias:
            self.embed = nn.Embedding(num_classes, num_features * 2)
            self.embed.weight.data[:, :num_features].uniform_()  # Initialise scale at N(1, 0.02)
            self.embed.weight.data[:, num_features:].zero_()  # Initialise bias at 0
        else:
            self.embed = nn.Embedding(num_classes, num_features)
            self.embed.weight.data.uniform_()

    def forward(self, x, y):
        out = self.bn(x)
        if self.bias:
            gamma, beta = self.embed(y).chunk(2, dim=1)
            out = gamma.view(-1, self.num_features, 1, 1) * out + beta.view(-1, self.num_features, 1, 1)
        else:
            gamma = self.embed(y)
            out = gamma.view(-1, self.num_features, 1, 1) * out
        return out


class ScoreBasedNoiseMatching(nn.Module):
    """
    For time series data, size of input is [batch_size, num_channels = 1, num_feature=time_series_length]
    """

    def __init__(self, td: int):
        super().__init__()
        self.timeSeriesLength = td
        self.hidden_units = 128
        self.score = nn.Sequential(
            nn.Linear(self.timeSeriesLength, self.hidden_units),
            nn.Softplus(),
            nn.Linear(self.hidden_units, self.hidden_units),
            nn.Softplus(),
            nn.Linear(self.hidden_units, self.timeSeriesLength),
        )
        self.normalizer = ConditionalBatchNorm1d()

    def forward(self, x, y):
        assert (x.shape[-1] == self.timeSeriesLength)
        out = self.begin_conv(x)
        out = self.score(out)
        out = self.normalizer(out, y)
        out = self.act(out)
        out = self.end_conv(out)
        return out
