import torch.nn as nn


class GRUNet(nn.Module):
    """
    Gated Recurrent Unit.
    """

    def __init__(self, d_in, h, d_out, gru_layers, preds_len):
        super().__init__()
        self.preds_len = preds_len
        # self.h0 = torch.randn(gru_layers, batch_size, h)
        self.gru = nn.GRU(d_in, h, batch_first=True, num_layers=gru_layers)
        self.linear = nn.Linear(h, d_out)

    def forward(self, x):
        output, hn = self.gru(x)
        output = self.linear(output[:, -self.preds_len:, :])
        return output
