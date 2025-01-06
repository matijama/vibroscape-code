from collections import OrderedDict
from nemo.core import NeuralModule, Exportable
from nemo.core.classes.common import typecheck
from nemo.core.neural_types import NeuralType, LogitsType, LengthsType, AcousticEncodedRepresentation, SpectrogramType


class IdentityEncoder(NeuralModule, Exportable):

    @property
    def input_types(self):
        """Returns definitions of module input ports.
        """
        return OrderedDict(
            {
                "audio_signal": NeuralType(('B', 'D', 'T'), SpectrogramType()),
                "length": NeuralType(tuple('B'), LengthsType()),
            }
        )

    @property
    def output_types(self):
        """Returns definitions of module output ports.
        """
        return OrderedDict(
            {
                "outputs": NeuralType(('B', 'D', 'T'), AcousticEncodedRepresentation()),
                "encoded_lengths": NeuralType(tuple('B'), LengthsType()),
            }
        )

    def __init__(
        self,
    ):
        super().__init__()

    def forward(self, audio_signal, length=None):
        audio_signal.neural_type = NeuralType(axes=('B', 'D', 'T'), elements_type=AcousticEncodedRepresentation())
        return audio_signal, length


class IdentityDecoder(NeuralModule, Exportable):

    @property
    def input_types(self):
        return OrderedDict(
            {
                "encoder_output": NeuralType(('B', 'D'), AcousticEncodedRepresentation()),
                "length": NeuralType(('B',), LengthsType(), optional=True),
            }
        )

    @property
    def output_types(self):
        return OrderedDict(
            {
                "logits": NeuralType(('B', 'D'), LogitsType()),
            }
        )

    def __init__(
        self,
        num_classes: int,
    ):
        super().__init__()
        self._num_classes = num_classes

    @typecheck()
    def forward(self, encoder_output, length=None):

        # input is BDT, flatten to BX, apply FC, then sigmoid
        # out = torch.sigmoid(self.fc_dec(torch.flatten(encoder_output, 1)))

        return encoder_output

    @property
    def num_classes(self):
        return self._num_classes
