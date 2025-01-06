from collections import OrderedDict

import torch
from nemo.core.classes.exportable import Exportable
from nemo.core.classes.module import NeuralModule
from nemo.core.neural_types import (
    AcousticEncodedRepresentation,
    LengthsType,
    LogitsType,
    LogprobsType,
    NeuralType,
    SpectrogramType,
)
from transformers import ASTConfig, ASTModel
from transformers.models.audio_spectrogram_transformer.modeling_audio_spectrogram_transformer import ASTMLPHead

from PaSST.passt import get_model
from models.PANNsCode import Cnn14


class PaSSTEncoder(NeuralModule, Exportable):
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
        arch='passt_s_swa_p16_128_ap476',
        pretrained=True,
        num_classes=527,
        fstride=16,
        tstride=16,
        u_patchout=0,
        s_patchout_t=0,
        s_patchout_f=0,
    ):

        super().__init__()
        self._model = get_model(arch=arch,pretrained=pretrained, fstride=fstride, tstride=tstride, u_patchout=u_patchout, s_patchout_t=s_patchout_t, s_patchout_f=s_patchout_f)
        if self._model.num_classes != num_classes:
            # reset head
            self._model.num_classes = num_classes
            self._model.head[1] = torch.nn.Linear(self._model.num_features, num_classes, bias=True,device=self._model.head[1].weight.device, dtype=self._model.head[1].weight.dtype)
            torch.nn.init.xavier_uniform_(self._model.head[1].weight)

    def forward(self, audio_signal, length=None):

        if audio_signal.ndim == 3:
            audio_signal = audio_signal.unsqueeze(1)
        audio_signal = audio_signal.type(self._model.patch_embed.proj.weight.dtype)
        res = self._model.forward(audio_signal)
        return res

