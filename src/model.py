import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(
        self,
        input_size=4,
        hidden_size=128,
        num_layers=2,
        dropout=0.3,
        bidirectional=False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_dirs = 2 if bidirectional else 1
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * self.num_dirs, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        out, (h, c) = self.lstm(x)
        mean_out = out.mean(dim=1)
        last_layer_h = h.view(
            self.num_layers, self.num_dirs, x.size(0), self.hidden_size
        )[-1]
        last_h = last_layer_h.transpose(0, 1).reshape(
            x.size(0), self.hidden_size * self.num_dirs
        )
        return self.classifier(last_h)
