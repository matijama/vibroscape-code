from collections import OrderedDict
import torch
from nemo.collections.asr.parts.submodules.jasper import init_weights
from nemo.core import NeuralModule, Exportable
from nemo.core.classes.common import typecheck
from nemo.core.neural_types import NeuralType, LogitsType, LengthsType, AcousticEncodedRepresentation
from torch import nn


def interpolate(x, ratio):
    """Interpolate data in time domain. This is used to compensate the
    resolution reduction in downsampling of a CNN.

    Args:
      x: (batch_size, time_steps, classes_num)
      ratio: int, ratio to interpolate

    Returns:
      upsampled: (batch_size, time_steps * ratio, classes_num)
    """
    (batch_size, time_steps, classes_num) = x.shape
    upsampled = x[:, :, None, :].repeat(1, 1, ratio, 1)
    upsampled = upsampled.reshape(batch_size, time_steps * ratio, classes_num)
    return upsampled


def pad_framewise_output(framewise_output, frames_num):
    """Pad framewise_output to the same length as input frames. The pad value
    is the same as the value of the last frame.

    Args:
      framewise_output: (batch_size, frames_num, classes_num)
      frames_num: int, number of frames to pad

    Outputs:
      output: (batch_size, frames_num, classes_num)
    """
    pad = framewise_output[:, -1 :, :].repeat(1, frames_num - framewise_output.shape[1], 1)
    """tensor for padding"""

    output = torch.cat((framewise_output, pad), dim=1)
    """(batch_size, frames_num, classes_num)"""

    return output


class LinearSigmoidDecoder(NeuralModule, Exportable):

    @property
    def input_types(self):
        return OrderedDict(
            {
                "encoder_output": NeuralType(('B', 'D', 'T'), AcousticEncodedRepresentation()),
                "length": NeuralType(('B',), LengthsType(), optional=True),
            }
        )

    @property
    def output_types(self):
        return OrderedDict(
            {
                "logits": NeuralType(('B', 'D'), LogitsType()),
                "logits_per_frame": NeuralType(('B', 'D', 'T'), LogitsType()),
            }
        )

    def __init__(
        self,
        feat_in: int,
        num_classes: int,
        interpolate_ratio: int=32,
        return_per_frame: bool = False,
        init_mode: str = "xavier_uniform",
    ):
        super().__init__()

        self._num_classes = num_classes
        self.interpolate_ratio = interpolate_ratio
        self.return_per_frame = return_per_frame
        self.fc_dec = nn.Linear(feat_in, num_classes, bias=True)
        self.apply(lambda x: init_weights(x, mode=init_mode))

    @typecheck()
    def forward(self, encoder_output, length=None):

        # input is BDT, flatten to BX, apply FC, then sigmoid
        # out = torch.sigmoid(self.fc_dec(torch.flatten(encoder_output, 1)))

        encoder_output = encoder_output.transpose(1, 2)
        segmentwise_output = torch.sigmoid(self.fc_dec(encoder_output))
        (clipwise_output, _) = torch.max(segmentwise_output, dim=1)

        if self.return_per_frame:
            framewise_output = interpolate(segmentwise_output, self.interpolate_ratio)
            framewise_output = pad_framewise_output(framewise_output, length)
            framewise_output = framewise_output.transpose(1, 2)
        else:
            framewise_output = None
        return clipwise_output, framewise_output

    @property
    def num_classes(self):
        return self._num_classes
