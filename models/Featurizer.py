import h5py
import torch
import torch.nn as nn
import numpy as np
from nemo.collections.asr.modules.audio_preprocessing import AudioPreprocessor
from nemo.collections.asr.parts.preprocessing import AudioSegment, WaveformFeaturizer
from nemo.collections.asr.parts.preprocessing.features import normalize_batch
from nemo.core.neural_types import NeuralType, AudioSignal, LengthsType, SpectrogramType


class Hdf5WaveformFeaturizer(WaveformFeaturizer):
    def __init__(self, sample_rate=16000, hop_size=0.02, int_values=False, augmentor=None):
        super().__init__(sample_rate=sample_rate, int_values=int_values, augmentor=augmentor)
        self.hop_size = hop_size

    def process(
        self,
        file_path,
        offset=0,
        duration=0,
        trim=False,
        trim_ref=np.max,
        trim_top_db=60,
        trim_frame_length=2048,
        trim_hop_length=512,
        orig_sr=None,
        channel_selector=None,
    ):
        if file_path.lower().endswith("hdf5"):
            chan = 0
            if channel_selector is not None:
                chan = channel_selector
            frame_start = int(np.round(offset / self.hop_size))
            frame_end = int(np.round((offset + duration) / self.hop_size)) + 1
            with h5py.File(file_path, 'r') as f:
                ds = f[f'pcen_{chan}']
                features = ds[frame_start:frame_end, :]

            return torch.tensor(features, dtype=torch.float)

        else:
            audio = AudioSegment.from_file(
                file_path,
                target_sr=self.sample_rate,
                int_values=self.int_values,
                offset=offset,
                duration=duration,
                trim=trim,
                trim_ref=trim_ref,
                trim_top_db=trim_top_db,
                trim_frame_length=trim_frame_length,
                trim_hop_length=trim_hop_length,
                orig_sr=orig_sr,
                channel_selector=channel_selector,
            )
            return self.process_segment(audio)


class Hdf5Preprocessor(AudioPreprocessor):

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
        sample_rate=16000,
        window_size=0.02,
        window_stride=0.01,
        normalize="per_feature",
        features=64,
        pad_value=0,
        pad_to=0,
    ):
        super().__init__(None, None)

        self._sample_rate = sample_rate
        self._features = features
        self._window_size = window_size
        self._window_stride = window_stride
        self.normalize = normalize
        self.pad_value = pad_value
        self.pad_to = pad_to


    def get_features(self, input_signal, length):
        x = input_signal
        seq_len = length

        maxseqlen = torch.max(seq_len)
        x = torch.reshape(x, (x.shape[0], maxseqlen, int(x.shape[1]/maxseqlen)))
        x = torch.permute(x, (0,2,1))

        # normalize if required
        if self.normalize:
            x, _, _ = normalize_batch(x, seq_len, normalize_type=self.normalize)

        # mask to zero any values beyond seq_len in batch, pad to multiple of `pad_to` (for efficiency)
        max_len = x.size(-1)
        mask = torch.arange(max_len).to(x.device)
        mask = mask.repeat(x.size(0), 1) >= seq_len.unsqueeze(1)
        x = x.masked_fill(mask.unsqueeze(1).type(torch.bool).to(device=x.device), self.pad_value)
        del mask
        pad_to = self.pad_to
        if pad_to == "max":
            x = nn.functional.pad(x, (0, self.max_length - x.size(-1)), value=self.pad_value)
        elif pad_to > 0:
            pad_amt = x.size(-1) % pad_to
            if pad_amt != 0:
                x = nn.functional.pad(x, (0, pad_to - pad_amt), value=self.pad_value)

        return x, seq_len

