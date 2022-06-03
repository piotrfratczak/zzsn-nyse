import torch
import torch.nn as nn


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GRUNet(nn.Module):
    """
    Gated Recurrent Unit.
    """
    def __init__(self, input_size, output_size, hidden_size, gru_layers, proj_len, dropout=0.2):
        super(GRUNet, self).__init__()
        self.proj_len = proj_len
        self.gru_layers = gru_layers
        self.hidden_dim = hidden_size

        self.gru = nn.GRU(input_size, hidden_size, num_layers=gru_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.gru_layers, x.size(0), self.hidden_dim).to(device)
        output, _ = self.gru(x, h0)
        output = output[:, -self.proj_len:, :]
        output = self.fc(output)
        return output
