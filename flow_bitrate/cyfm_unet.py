# Adapted from Complex-Flow-Matching; see licenses/CyFM-LICENSE.
"""The paper's U-Net, and the time-conditioned block it is built from.

One trunk serves both geometries. The only architectural difference between the
cylindrical and Cartesian arms is the width of the first convolution, which
follows ``Manifold.state_channels``: three for ``(m, cos phi, sin phi)``, two for
``(Re z, Im z)``. Everything after it is identical, which is what makes a
measured difference between the arms a difference in geometry.
"""

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

import math


class SinusoidalPositionEmbeddings(nn.Module):
    """Standard sinusoidal embedding of a scalar time.

    Args:
        dim: Embedding width, must be positive and even.

    Raises:
        ValueError: If ``dim`` is not a positive even number.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        if dim <= 0 or dim % 2 != 0:
            raise ValueError(f"embedding dim must be positive and even, got {dim}")
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        """Embed a batch of times.

        Args:
            time: Times of shape ``[B]``.

        Returns:
            Embedding of shape ``[B, dim]``.
        """
        half = self.dim // 2
        scale = math.log(10000.0) / max(half - 1, 1)
        frequencies = torch.exp(torch.arange(half, device=time.device) * -scale)
        angles = time.reshape(-1, 1) * frequencies.reshape(1, -1)
        return torch.cat([angles.sin(), angles.cos()], dim=-1)



class TimeConditionedBlock(nn.Module):
    """Residual convolutional block conditioned on diffusion time embedding.

    Args:
        in_channels: Input feature channels.
        out_channels: Output feature channels.
        time_emb_dim: Dimension of time embedding vector.
    """

    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int,
                 *, film=False, time_dilation=1) -> None:
        super().__init__()
        self.film = film
        self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, out_channels * (2 if film else 1)))
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               padding=(1, time_dilation), dilation=(1, time_dilation))
        self.norm2 = nn.GroupNorm(8, out_channels)

        self.silu = nn.SiLU()

        # Residual connection aligning channels if there was a change
        self.residual = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """Convolve, inject the time embedding, convolve again, add the residual."""
        # Main convolutional branch
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.silu(h)

        # Time injection (broadcasting over H and W dimensions)
        time_info = self.time_mlp(t_emb)
        time_info = time_info.unsqueeze(-1).unsqueeze(-1)
        if self.film:
            scale, shift = time_info.chunk(2, dim=1)
            h = h * (1 + scale) + shift
        else:
            h = h + time_info

        h = self.conv2(h)
        h = self.norm2(h)

        return self.silu(h + self.residual(x))


class DepthwiseResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.project = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.depthwise = nn.Conv2d(out_channels, out_channels, 7, padding=3, groups=out_channels)
        self.norm = nn.GroupNorm(8, out_channels)
        self.time = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, out_channels))
        self.mlp = nn.Sequential(nn.Conv2d(out_channels, 4 * out_channels, 1), nn.GELU(),
                                 nn.Conv2d(4 * out_channels, out_channels, 1))

    def forward(self, x, time):
        x = self.project(x)
        h = self.norm(self.depthwise(x)) + self.time(time)[:, :, None, None]
        return x + self.mlp(h)


class CylindricalUNet(nn.Module):
    """U-Net architecture for Continuous Flow Matching on complex MRI representations.

    Args:
        base_channels: Width of the first feature level.
        in_channels: Channels of input state tensor [B, C_in, H, W].
        out_channels: Channels of output velocity field [B, C_out, H, W].
    """

    def __init__(
        self,
        base_channels: int = 64,
        in_channels: int = 3,
        out_channels: int = 2,
        velocity_bound: Sequence[float] | None = None,
        time_embedding_dim: int | None = None,
        variant: str = "unet",
    ) -> None:
        super().__init__()

        # Per-channel ceiling on |velocity|, applied as bound * tanh(logit / bound) so the
        # map is the identity to first order and saturates only where the target cannot
        # go. Each geometry supplies its own from its own metric, so this is not an
        # advantage handed to one arm: on the cylinder the angular coordinate lives on a
        # circle of diameter pi and the bound binds; in the plane the bound is the support
        # diameter and is slack. The marginal field a flow-matching model converges to is
        # a conditional expectation of targets that already respect these limits, so the
        # optimum lies strictly inside and nothing representable is lost.
        if velocity_bound is None:
            self.register_buffer("_bound", None)
        else:
            bound = torch.tensor(list(velocity_bound), dtype=torch.float32)
            if bound.numel() != out_channels:
                raise ValueError(
                    f"velocity_bound has {bound.numel()} entries for {out_channels} channels"
                )
            if not bool((bound > 0).all()):
                raise ValueError(f"velocity_bound must be positive, got {velocity_bound}")
            self.register_buffer("_bound", bound.reshape(1, -1, 1, 1))

        time_emb_dim = base_channels * 4 if time_embedding_dim is None else time_embedding_dim
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(base_channels),
            nn.Linear(base_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        def block(in_width, out_width, time_dilation=1):
            if variant == "convnext":
                return DepthwiseResidualBlock(in_width, out_width, time_emb_dim)
            return TimeConditionedBlock(in_width, out_width, time_emb_dim,
                                        film=variant == "film", time_dilation=time_dilation)

        first_stride = (2, 1) if variant == "frequency-first" else 2
        # ENCODER (Downsampling, extracting global context)
        self.down1 = block(base_channels, base_channels * 2)
        self.pool1 = nn.MaxPool2d(first_stride)

        self.down2 = block(base_channels * 2, base_channels * 4)
        self.pool2 = nn.MaxPool2d(2)

        # BOTTLENECK (Deepest representation)
        self.bottleneck = block(base_channels * 4, base_channels * 4,
                                time_dilation=4 if variant == "dilated" else 1)

        # DECODER (Upsampling + Skip Connections)
        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        # Channels: 4*base (from upsample) + 4*base (from skip connection) = 8*base
        self.up_block1 = block(base_channels * 8, base_channels * 2)

        self.up2 = nn.Upsample(scale_factor=first_stride, mode="bilinear", align_corners=False)
        # Channels: 2*base (from upsample) + 2*base (from skip connection) = 4*base
        self.up_block2 = block(base_channels * 4, base_channels)

        # Final convolution outputting the velocity field
        self.heads = None
        if variant == "separate-heads":
            self.heads = nn.ModuleList(nn.Sequential(nn.Conv2d(base_channels, base_channels // 2, 3, padding=1),
                                                      nn.SiLU(), nn.Conv2d(base_channels // 2, 1, 1))
                                       for _ in range(out_channels))
        else:
            self.final_conv = nn.Conv2d(base_channels, out_channels, kernel_size=1)

        # Initial convolution, from the manifold's state channels. This is the one
        # module whose shape depends on the manifold, so constructing it last means
        # every other module has already drawn from the RNG at the same stream
        # position in both geometries: with one seed, the cylindrical and Euclidean
        # models start from element-wise identical weights everywhere except this
        # convolution. Moving it earlier silently reintroduces an initialisation
        # confound. Pinned by tests/test_manifolds/test_manifolds.py.
        self.init_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        self.coarse_projections = None
        if variant == "multiscale":
            self.coarse_projections = nn.ModuleList(nn.Conv2d(in_channels // 2, base_channels * scale, 1)
                                                    for scale in (2, 4, 8, 4))

    def forward(self, x: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        """Predict velocity field for state at given diffusion time.

        Args:
            x: Manifold state tensor [B, in_channels, H, W].
            time: Diffusion time values [B].

        Returns:
            Predicted velocity field [B, out_channels, H, W].
        """
        t_emb = self.time_mlp(time)
        condition = x[:, x.shape[1] // 2:]

        def condition_at(features, level):
            if self.coarse_projections is None:
                return features
            coarse = F.adaptive_avg_pool2d(condition, features.shape[-2:])
            return features + self.coarse_projections[level](coarse)

        x = self.init_conv(x)

        # Encoder
        d1 = self.down1(x, t_emb)
        p1 = self.pool1(d1)

        d2 = self.down2(condition_at(p1, 0), t_emb)
        p2 = self.pool2(d2)

        # Bottleneck
        b = self.bottleneck(condition_at(p2, 1), t_emb)

        # Decoder with Skip Connections
        u1 = self.up1(b)
        u1 = torch.cat([u1, d2], dim=1)
        u1 = self.up_block1(condition_at(u1, 2), t_emb)

        u2 = self.up2(u1)
        u2 = torch.cat([u2, d1], dim=1)
        u2 = self.up_block2(condition_at(u2, 3), t_emb)

        velocity = self.final_conv(u2) if self.heads is None else torch.cat([head(u2) for head in self.heads], dim=1)
        bound = self._bound
        if bound is None:
            return velocity
        assert isinstance(bound, torch.Tensor)
        return bound * torch.tanh(velocity / bound)
