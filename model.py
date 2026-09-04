from torch import nn


class LSTMStocksModule(nn.Module):

    HIDDEN_SIZE = 32
    NUM_LAYERS = 2
    BIAS = True
    DROPOUT = 0.2

    def init(self):
        super().init()

        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=self.HIDDEN_SIZE,
            num_layers=self.NUM_LAYERS,
            bias=self.BIAS,
            batch_first=True,
            dropout=self.DROPOUT
        )

        self.linear = nn.Linear(
            self.HIDDEN_SIZE,
            1
        )

    def forward(self, x):

        output, (hidden, cell) = self.lstm(
            x.unsqueeze(-1)
        )

        out = hidden[-1]

        out = self.linear(out).squeeze(-1)

        return out
