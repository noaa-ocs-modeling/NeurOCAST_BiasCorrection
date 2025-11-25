import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from physicsnemo.models.fno import FNO

class Feedforward(nn.Module):
    def __init__(self, channel_size, hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(channel_size * 2 + 4, hidden_size)
        self.fc2 = nn.Linear(hidden_size, channel_size)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x.permute(0, 2, 1)

class Feedforward_aggre(nn.Module):
    def __init__(self, channel_size, hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(channel_size * 2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, channel_size)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x.permute(0, 2, 1)

class GNO(MessagePassing):
    def __init__(self, in_channels):
        super().__init__(aggr='mean', node_dim=-3)
        hidden = in_channels * 4
        self.edge = Feedforward(in_channels, hidden)
        self.aggre = Feedforward_aggre(in_channels, hidden)
    @torch.compile
    def forward(self, x, edge_index, x_loc):
        return self.propagate(edge_index[:2, :].long(), x=x, xloc=edge_index[2:, :] if x_loc is None else x_loc)

    def message(self, x_i, x_j, xloc):
        loc_i = xloc[:2, :].T
        loc_j = xloc[2:4, :].T

        loc_i = loc_i.unsqueeze(2).repeat(1, 1, x_i.shape[-1])
        loc_j = loc_j.unsqueeze(2).repeat(1, 1, x_j.shape[-1])

        tmp = torch.cat([x_i, loc_i, x_j, loc_j], dim=1)
        return self.edge(tmp)

    def update(self, aggr_out, x):
        return self.aggre(torch.cat([aggr_out, x], dim=1))

class NeurOCAST(nn.Module):
    def __init__(self,
                 input_channels=8,
                 output_channels=1,
                 base_width=32,
                 base_modes=16,
                 num_layers=3,
                 width_scale=2,
                 mode_scale=1/2,
                 resample_strategy=None,
                 padding=10):
        super().__init__()

        self.padding = padding
        self.input_channels = input_channels + 1

        self.fc0 = nn.Linear(self.input_channels, base_width)

        self.fno_layers = nn.ModuleList()
        self.gno_layers = nn.ModuleList()

        # Convert scalar scales to list
        if not isinstance(width_scale, list):
            width_scale = [width_scale] * num_layers
        if not isinstance(mode_scale, list):
            mode_scale = [mode_scale] * num_layers

        width = base_width
        modes = base_modes

        for i in range(num_layers):
            out_width = int(width * width_scale[i])
            out_modes = max(1, int(modes * mode_scale[i]))

            self.fno_layers.append(FNO(
                in_channels=width,
                decoder_layer_size=out_width,
                out_channels=out_width,
                latent_channels=width,
                dimension=1,
                num_fno_layers=1,
                num_fno_modes=out_modes,
                coord_features = False,
            ))
            self.gno_layers.append(GNO(out_width))

            width = out_width
            modes = out_modes

        self.resample_strategy = resample_strategy or [None] * num_layers
        self.final_fno = FNO(
            in_channels=width,
            latent_channels=width,
            decoder_layer_size=out_width,
            out_channels=base_width,
            dimension=1,
            num_fno_layers=1,
            num_fno_modes=base_modes,
            coord_features = False,
        )

        self.fc1 = nn.Linear(base_width, base_width * 2)
        self.fc2 = nn.Linear(base_width * 2, output_channels)

    def forward(self, x, edge_index, x_loc):
        
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=1)
        x = F.pad(x, [self.padding, self.padding], mode='reflect')
        final_size = x.shape[2]
        x = x.squeeze().permute(0, 2, 1)
        x = self.fc0(x).permute(0, 2, 1)
        for fno, gno, target_size in zip(self.fno_layers, self.gno_layers, self.resample_strategy):
            x = fno(x)
            if target_size is not None:
                x = self.resample_tensor(x, target_size)
            x = gno(x, edge_index, x_loc)
    
        x = self.final_fno(x)
        x = self.resample_tensor(x, final_size/x.shape[2])
        x = x[..., self.padding:-self.padding]
        x = x.permute(0, 2, 1)
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        return x.permute(0, 2, 1).squeeze(1)

    @torch.compile
    def get_grid(self, shape, device):
        nstation, size_t = shape[-3], shape[-1]
        gridx = torch.linspace(0, 1, steps=size_t, dtype=torch.float).reshape(1, 1, size_t).repeat(nstation, 1, 1)
        return gridx.to(device)

    @torch.compile
    def resample_tensor(self, x, size):
        return F.interpolate(x, size=round(size*x.shape[2]), mode='linear', align_corners=True)
