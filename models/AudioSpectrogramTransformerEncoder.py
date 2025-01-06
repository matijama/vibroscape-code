from collections import OrderedDict
from nemo.core.classes.exportable import Exportable
from nemo.core.classes.module import NeuralModule
from nemo.core.neural_types import (
    AcousticEncodedRepresentation,
    LengthsType,
    LogitsType,
    NeuralType,
    SpectrogramType,
)
from transformers import ASTConfig, ASTModel
from transformers.models.audio_spectrogram_transformer.modeling_audio_spectrogram_transformer import ASTMLPHead


class AudioSpectrogramTransformerEncoder(NeuralModule, Exportable):
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
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        patch_size=16,
        qkv_bias=True,
        frequency_stride=10,
        time_stride=10,
        max_length=1024,
        num_mel_bins=128,
    ):
        super().__init__()
        cfg = ASTConfig(hidden_size, num_hidden_layers, num_attention_heads, intermediate_size, hidden_act, hidden_dropout_prob, attention_probs_dropout_prob,
                            initializer_range, layer_norm_eps, patch_size, qkv_bias, frequency_stride, time_stride, max_length, num_mel_bins)
        self._model = ASTModel(cfg)


    def forward(self, audio_signal, length=None):

        res = self._model.forward(audio_signal)
        return res


class AudioSpectrogramTransformerMLPDecoder(NeuralModule, Exportable):
    """
    Audio Spectrogram Transformer MLP Decoder
    """

    @property
    def input_types(self):
        return OrderedDict({"encoder_output": NeuralType(('B', 'T', 'D'), AcousticEncodedRepresentation())})

    @property
    def output_types(self):
        return OrderedDict({"logits": NeuralType(('B', 'D'), LogitsType())})

    def __init__(
        self,
        feat_in=768,
        layer_norm_eps=1e-12,
        num_classes=2,
    ):
        super().__init__()

        self._num_classes = num_classes

        cfg = ASTConfig(hidden_size=feat_in, layer_norm_eps=layer_norm_eps, num_labels=num_classes)
        self._model = ASTMLPHead(cfg)

    def forward(self, encoder_output):

        return self._model.forward(encoder_output)

    @property
    def num_classes(self):
        return self._num_classes