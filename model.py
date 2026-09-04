import torch
from torch import nn


class LSTMStocksModule(nn.Module):

    INPUT_SIZE = 9
    HIDDEN_SIZE = 32
    NUM_LAYERS = 2
    DROPOUT = 0.20

    def init(self):

        super().init()

        self.lstm = nn.LSTM(
            input_size=self.INPUT_SIZE,
            hidden_size=self.HIDDEN_SIZE,
            num_layers=self.NUM_LAYERS,
            batch_first=True,
            dropout=self.DROPOUT
        )

        self.dropout = nn.Dropout(self.DROPOUT)

        self.fc1 = nn.Linear(
            self.HIDDEN_SIZE,
            16
        )

        self.relu = nn.ReLU()

        self.fc2 = nn.Linear(
            16,
            1
        )

    def forward(self, x):

        # x shape:
        # (batch, sequence_length, 9)

        output, (hidden, cell) = self.lstm(x)

        # Last LSTM layer's final hidden state
        out = hidden[-1]

        out = self.dropout(out)

        out = self.fc1(out)

        out = self.relu(out)

        out = self.fc2(out)

        return out.squeeze(-1)
