import torch
import torch.nn as nn


class GRUNet(nn.Module):
    """
    Gated Recurrent Unit.
    """

    def __init__(self, input_dim, hidden_dim, output_dim, gru_layers, dropout=0.2):
        super(GRUNet, self).__init__()
        self.preds_len = output_dim
        self.gru_layers = gru_layers
        self.hidden_dim = hidden_dim

        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=gru_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.gru_layers, x.size(0), self.hidden_dim)
        output, hn = self.gru(x, h0.detach())
        output = self.dropout(output[:, -self.preds_len:, :])
        output = self.fc(output)
        return output
