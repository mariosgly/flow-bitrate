"""Additional spectrogram velocity networks for the Jamendo screen."""

import torch
import torch.nn.functional as F
from torch import nn

from .cyfm_unet import SinusoidalPositionEmbeddings


def reduced_ncsnpp(channels):
    from flowdec.backbones.ncsnpp import NCSNpp
    import yaml

    from .codec import UPSTREAM

    class RealNCSNpp(NCSNpp):
        def _preprocess(self, x, y):
            return x

        def _postprocess(self, h):
            return h

        def forward(self, x, time):
            return super().forward(x, None, time)

    config = yaml.safe_load((UPSTREAM / "config/model/backbone/ncsnpp_final_no_attn.yaml").read_text())
    config.pop("_target_")
    config.update(nf=32, ch_mult=[4, 5, 5, 3], num_channels=channels)
    return RealNCSNpp(**config)


class BandMixer(nn.Module):
    def __init__(self, width, layer, attention):
        super().__init__()
        self.band_norm = nn.LayerNorm(width)
        self.band_mix = nn.MultiheadAttention(width, 8, batch_first=True)
        self.time_norm = nn.LayerNorm(width)
        self.attention = attention
        self.shift = 64 if layer % 2 else 0
        dilation = 2 ** layer
        if attention:
            self.temporal = nn.MultiheadAttention(width, 8, batch_first=True)
        else:
            self.temporal = nn.Conv1d(width, width, 5, padding=2 * dilation, dilation=dilation, groups=width)
        self.ffn = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 2 * width),
                                 nn.GELU(), nn.Linear(2 * width, width))

    def forward(self, x):
        batch, frames, bands, width = x.shape
        across_bands = self.band_norm(x).reshape(batch * frames, bands, width)
        mixed, _ = self.band_mix(across_bands, across_bands, across_bands, need_weights=False)
        x = x + mixed.reshape_as(x)
        temporal = self.time_norm(x).transpose(1, 2).reshape(batch * bands, frames, width)
        if self.attention:
            # Shift by padding to connect adjacent windows in alternate layers.
            tail = (-(frames + self.shift)) % 128
            windows = F.pad(temporal, (0, 0, self.shift, tail)).reshape(-1, 128, width)
            valid = F.pad(torch.ones(frames, dtype=torch.bool, device=x.device), (self.shift, tail))
            padding = ~valid.reshape(-1, 128).repeat(batch * bands, 1)
            mixed, _ = self.temporal(windows, windows, windows, key_padding_mask=padding, need_weights=False)
            mixed = mixed.reshape(batch * bands, -1, width)[:, self.shift:self.shift + frames]
        else:
            mixed = self.temporal(temporal.transpose(1, 2)).transpose(1, 2)
        x = x + mixed.reshape(batch, bands, frames, width).transpose(1, 2)
        return x + self.ffn(x)


class BandSplitVelocity(nn.Module):
    """Project narrow low-frequency and wider high-frequency bands to tokens."""

    bands = (12,) * 16 + (24,) * 8 + (48,) * 8

    def __init__(self, in_channels, attention, width=256, depth=5):
        super().__init__()
        self.inputs = nn.ModuleList(nn.Linear(in_channels * band, width) for band in self.bands)
        self.outputs = nn.ModuleList(nn.Linear(width, 2 * band) for band in self.bands)
        self.time = nn.Sequential(SinusoidalPositionEmbeddings(width), nn.Linear(width, width),
                                  nn.SiLU(), nn.Linear(width, width))
        self.layers = nn.ModuleList(BandMixer(width, layer, attention) for layer in range(depth))

    def forward(self, x, time):
        batch, _, _, frames = x.shape
        tokens = [project(band.permute(0, 3, 1, 2).flatten(2))
                  for band, project in zip(x.split(self.bands, dim=2), self.inputs)]
        x = torch.stack(tokens, dim=2) + self.time(time)[:, None, None]
        for layer in self.layers:
            x = layer(x)
        outputs = [project(band).reshape(batch, frames, 2, bins).permute(0, 2, 3, 1)
                   for band, project, bins in zip(x.unbind(dim=2), self.outputs, self.bands)]
        return torch.cat(outputs, dim=2)
