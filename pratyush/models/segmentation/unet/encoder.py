import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """A run of 3x3 convolutions, each followed by a ReLU.

    `padding=1` keeps the spatial size intact, so a block of any `num_convs`
    leaves the map exactly as big as it arrived:

        574 -> 574 -> 574 -> 574

    That is what lets the decoder concatenate a skip without cropping it.
    The first convolution changes the channel count, the rest hold it steady.
    This is the shared body of the encoder, the bottleneck and the decoder.
    """

    def __init__(self, in_channels: int, out_channels: int, num_convs: int = 3):
        super().__init__()
        self.convs = nn.ModuleList(
            nn.Conv2d(
                in_channels if i == 0 else out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
            )
            for i in range(num_convs)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for conv in self.convs:
            x = self.relu(conv(x))
        return x


class EncoderBlock(nn.Module):
    """One step down the left half of the U-Net: a ConvBlock, then a 2x2 pool.

    The ConvBlock preserves the size, so only the pool changes it: 574 -> 287.

    Returns two tensors, because the decoder needs the map from *before*
    the pool for its skip connection.
    """

    def __init__(self, in_channels: int, out_channels: int, num_convs: int = 3):
        super().__init__()
        self.conv_block = ConvBlock(in_channels, out_channels, num_convs=num_convs)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        skip = self.conv_block(x)
        pooled = self.pool(skip)

        return skip, pooled


class Encoder(nn.Module):
    """The contracting half of the U-Net: `depth` encoder blocks in a row.

    The first block widens the grayscale input to `base_channels`, then every
    block after it doubles the channel count while the pool halves the size:

        1 -> 64 -> 128 -> 256 -> 512

    Convolutions are padded, so a skip comes out the same size as that
    level's input. Give it a size divisible by 2 ** depth and no pool will
    ever land on an odd number and drop a row.

    Collects one skip tensor per block on the way down. They come back in
    top-to-bottom order, so the decoder consumes them in reverse.
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 64,
        depth: int = 4,
        num_convs: int = 3,
    ):
        super().__init__()
        self.channels = [base_channels * 2 ** i for i in range(depth)]

        self.blocks = nn.ModuleList(
            EncoderBlock(
                in_channels if i == 0 else self.channels[i - 1],
                self.channels[i],
                num_convs=num_convs,
            )
            for i in range(depth)
        )

    @property
    def out_channels(self) -> int:
        """Channels handed to the bottleneck."""
        return self.channels[-1]

    def forward(self, x: torch.Tensor) -> tuple[list[torch.Tensor], torch.Tensor]:
        skips: list[torch.Tensor] = []

        for block in self.blocks:
            skip, x = block(x)
            skips.append(skip)

        return skips, x
