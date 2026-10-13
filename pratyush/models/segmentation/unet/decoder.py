import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import ConvBlock


class DecoderBlock(nn.Module):
    """One step up the right half of the U-Net.

    The paper's "up-conv 2x2" both doubles the size and halves the channels.
    Interpolation only does the first -- it has no weights and cannot touch
    the channel count -- so a 3x3 convolution does the halving:

        1024 @ 36x36  --interpolate-->  1024 @ 72x72
                      --conv 3x3----->   512 @ 72x72
                      --cat SKIP[3]--->  1024 @ 72x72
                      --ConvBlock----->   512 @ 72x72

    Upsampling targets the skip's exact size rather than scale_factor=2, so a
    skip that came off an odd-sized level still lines up.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_convs: int = 3,
        mode: str = "bilinear",
    ):
        super().__init__()
        self.mode = mode

        self.reduce = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

        # concatenating the skip doubles the channels again
        self.conv_block = ConvBlock(out_channels * 2, out_channels, num_convs=num_convs)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        align = None if self.mode == "nearest" else False
        x = F.interpolate(x, size=skip.shape[-2:], mode=self.mode, align_corners=align)

        x = self.relu(self.reduce(x))
        x = torch.cat([skip, x], dim=1)

        return self.conv_block(x)


class Decoder(nn.Module):
    """The expanding half of the U-Net: one DecoderBlock per encoder level.

    Mirrors the encoder's channel list in reverse, so depth comes along with
    it and is never passed separately. Starting from the bottleneck:

        1024 -> 512 -> 256 -> 128 -> 64

    Each block consumes the skip from its own level, deepest first.
    """

    def __init__(
        self,
        channels: list[int],
        bottleneck_channels: int | None = None,
        num_convs: int = 3,
        mode: str = "bilinear",
    ):
        super().__init__()
        self.channels = list(channels)

        outs = list(reversed(self.channels))
        ins = [bottleneck_channels or self.channels[-1] * 2] + outs[:-1]

        self.blocks = nn.ModuleList(
            DecoderBlock(in_ch, out_ch, num_convs=num_convs, mode=mode)
            for in_ch, out_ch in zip(ins, outs)
        )

    @property
    def out_channels(self) -> int:
        """Channels handed to the output head."""
        return self.channels[0]

    def forward(self, x: torch.Tensor, skips: list[torch.Tensor]) -> torch.Tensor:
        # Checked up front: pairing the wrong number of skips would otherwise
        # fail deep inside a block with a confusing channel-mismatch error.
        if len(skips) != len(self.blocks):
            raise ValueError(
                f"expected {len(self.blocks)} skips, got {len(skips)} -- "
                f"encoder depth and decoder depth disagree"
            )

        for block, skip in zip(self.blocks, reversed(skips)):
            x = block(x, skip)

        return x
