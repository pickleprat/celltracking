import torch
import torch.nn as nn

from .encoder import ConvBlock


class Bottleneck(nn.Module):
    """The bottom of the U: a ConvBlock with no pool, doubling the channels.

        512 @ 30x30  ->  1024 @ 24x24

    Nothing is downsampled here, so nothing is handed to a skip connection.
    The paper applies dropout at the end of the contracting path, which is
    what `dropout` is for; pass 0.0 to disable it.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int | None = None,
        num_convs: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.out_channels = in_channels * 2 if out_channels is None else out_channels

        self.conv_block = ConvBlock(in_channels, self.out_channels, num_convs=num_convs)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.conv_block(x))
