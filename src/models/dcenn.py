import torch
import torch.nn as nn
import torch.nn.functional as F

class TinyDCENN(nn.Module):
    """
    Tiny discrete Cellular NN encoder.
    Input:  x [B, T, C_in, H, W]
    State:  s [B, C_hid, H, W]
    Update per time step t:
        u = tanh(Conv3x3([s, x_t]))
        g = sigmoid(Conv3x3([s, x_t]))
        s = s + g * u
    Output: z = GAP(s) -> [B, C_hid]
    """
    def __init__(self, in_channels, hidden_channels=256):
        super().__init__()
        self.conv_u = nn.Conv2d(in_channels + hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self.conv_g = nn.Conv2d(in_channels + hidden_channels, hidden_channels, kernel_size=3, padding=1)
        nn.init.kaiming_normal_(self.conv_u.weight)
        nn.init.xavier_normal_(self.conv_g.weight)

    def forward(self, x):
        B, T, C, H, W = x.shape
        s = x.new_zeros((B, self.conv_u.out_channels, H, W))
        for t in range(T):
            xt = x[:, t]  # [B,C,H,W]
            cat = torch.cat([s, xt], dim=1)
            u = torch.tanh(self.conv_u(cat))
            g = torch.sigmoid(self.conv_g(cat))
            s = s + g * u
        # global average pool
        z = F.adaptive_avg_pool2d(s, (1,1)).view(B, -1)
        return z

