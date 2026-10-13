import torch
import torch.nn as nn


class SegmentationHead(nn.Module):
    """Maps the decoder's feature channels to one logit per class per pixel.

    A 1x1 convolution: a per-pixel linear classifier over the 64 decoder
    channels. The kernel is 1x1 because all the spatial reasoning already
    happened in the decoder -- this only has to weigh "given these features
    at this pixel, cell or background?".

        64 @ 576x576  ->  out_channels @ 576x576

    Returns raw logits, with no sigmoid or softmax. Squashing belongs in the
    loss (BCEWithLogitsLoss fuses sigmoid with log for numerical stability),
    and at inference you apply sigmoid yourself.

    out_channels=1 gives binary cell/background. Use 3 for the instance-aware
    variant that predicts interior / boundary / background, which is how you
    stop touching cells merging into one blob.
    """

    def __init__(self, in_channels: int, out_channels: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)
