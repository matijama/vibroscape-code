from collections import OrderedDict
from nemo.core.classes.exportable import Exportable
from nemo.core.classes.module import NeuralModule
from nemo.core.neural_types import (
    AcousticEncodedRepresentation,
    LengthsType,
    NeuralType,
    SpectrogramType,
)

from models.PANNsCode import Cnn14


class PANNsEncoder(NeuralModule, Exportable):
    """
    Audio Spectrogram Transformer Encoder initialized from scratch
    """

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
        model_type='Cnn14',
        feat_in=128,
        feat_out=2048,
        keep_time=False
    ):
        super().__init__()
        if model_type == 'Cnn14':
            self._model = Cnn14(feat_in, feat_out, keep_time)

    def forward(self, audio_signal, length=None):

        res = self._model.forward(audio_signal)
        return res

