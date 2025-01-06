import functools

from nemo.collections.asr.modules.audio_preprocessing import AudioPreprocessor
from nemo.core import Exportable
from nemo.core.neural_types import (
    AudioSignal,
    LengthsType,
    NeuralType,
    SpectrogramType,
)
import torch
from torchopenl3.core import load_audio_embedding_model, get_audio_embedding


class AudioToOpenL3Preprocessor(AudioPreprocessor, Exportable):
    """Featurizer module that converts wavs to openl3 features.
    """

    def save_to(self, save_path: str):
        pass

    @classmethod
    def restore_from(cls, restore_path: str):
        pass

    @property
    def input_types(self):
        """Returns definitions of module input ports.
        """
        return {
            "input_signal": NeuralType(('B', 'T'), AudioSignal(freq=self._sample_rate)),
            "length": NeuralType(
                tuple('B'), LengthsType()
            ),  # Please note that length should be in samples not seconds.
        }

    @property
    def output_types(self):
        """Returns definitions of module output ports.

        processed_signal:
            0: AxisType(BatchTag)
            1: AxisType(MelSpectrogramSignalTag)
            2: AxisType(ProcessedTimeTag)
        processed_length:
            0: AxisType(BatchTag)
        """
        return {
            "processed_signal": NeuralType(('B', 'D', 'T'), SpectrogramType()),
            "processed_length": NeuralType(tuple('B'), LengthsType()),
        }

    def __init__(
        self,
        input_repr='mel128',
        content_type='music',
        features=6144,
        normalize='waveform',
        sample_rate=48000,
        window_size=1,
        window_stride=0.2,
    ):
        self._sample_rate = sample_rate
        self.normalize = normalize
        self.window_stride = window_stride

        n_window_size = int(window_size*sample_rate)
        n_window_stride = int(window_stride*sample_rate)
        super().__init__(n_window_size, n_window_stride)

        self.openl3_model = load_audio_embedding_model(input_repr, content_type, features)
        self.openl3_model.training = False
        for param in self.openl3_model.parameters():
            param.requires_grad = False

    def input_example(self, max_batch: int = 8, max_dim: int = 32000, min_length: int = 200):
        batch_size = torch.randint(low=1, high=max_batch, size=[1]).item()
        max_length = torch.randint(low=min_length, high=max_dim, size=[1]).item()
        signals = torch.rand(size=[batch_size, max_length]) * 2 - 1
        lengths = torch.randint(low=min_length, high=max_dim, size=[batch_size])
        lengths[0] = max_length
        return signals, lengths

    @torch.no_grad()
    def get_features(self, input_signal, length):

        seq_len = torch.round((length / self._sample_rate - 0.2) / self.window_stride)
        seq_len = seq_len.to(dtype=torch.long)

        # normalize batch if required
        if self.normalize is not None and self.normalize == 'waveform':
            maxv = torch.max(torch.abs(input_signal)) + 1e-6
            input_signal = input_signal / maxv

        x, ts_list = get_audio_embedding(input_signal, self._sample_rate, self.openl3_model, center=True, hop_size=self.window_stride, verbose=False)
        x = x.permute(0,2,1)

        # maybe we didn't exactly match the length, so we need to shorten it
        seq_len = torch.min(seq_len, torch.tensor(x.shape[-1]))

        return x, seq_len

    # @property
    # def filter_banks(self):
    #     return self.featurizer.filter_banks


