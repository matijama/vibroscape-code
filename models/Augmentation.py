import random
from random import randint

import numpy as np
import scipy
from audiomentations.core.transforms_interface import BaseWaveformTransform
from nemo.collections.asr.parts.preprocessing import NoisePerturbation, AudioSegment
from nemo.core import NeuralModule, typecheck
from nemo.core.neural_types import NeuralType, LengthsType, SpectrogramType, AudioSignal
import audiomentations as AA
from torch_time_stretch import *


def remove_outliers(signals, lengths, sr, winlen=0.01):
    madfactor = 1.4826  # -1 / (sqrt(2) * erfcinv(3 / 2));
    winsize = int(sr*winlen)+1
    for i in range(signals.shape[0]):
        w = signals[i][:lengths[i]]
        center = scipy.ndimage.median_filter(w, winsize)
        a = np.abs(w - center)
        amovmad = madfactor * scipy.ndimage.median_filter(a, winsize)  # approximation of moving mad
        lowerbound = center - 3 * amovmad
        upperbound = center + 3 * amovmad
        tf = np.logical_or(a > upperbound, a < lowerbound)
        w[tf] = center[tf]
        signals[i][:lengths[i]] = w
    return signals


class AudioAugmentation(NeuralModule):

    def __init__(self, enabled, shuffle, modules):
        super(AudioAugmentation, self).__init__()
        self.enabled = enabled
        self.shuffle = shuffle
        self.mixup = None
        augs = []
        if 'add_gaussian_noise' in modules:
            t = modules['add_gaussian_noise']
            augs.append(AA.AddGaussianNoise(min_amplitude=t['min_amplitude'], max_amplitude=t['max_amplitude'], p=t['p']))
        if 'add_gaussian_snr' in modules:
            t = modules['add_gaussian_snr']
            augs.append(AA.AddGaussianSNR(min_snr_db=t['min_snr_db'], max_snr_db=t['max_snr_db'], p=t['p']))
        if 'band_pass_filter' in modules:
            t = modules['band_pass_filter']
            augs.append(AA.BandPassFilter(min_center_freq=t['min_center_freq'], max_center_freq=t['max_center_freq'], min_bandwidth_fraction=t['min_bandwidth_fraction'],
                                          max_bandwidth_fraction=t['max_bandwidth_fraction'], min_rolloff=t['min_rolloff'], max_rolloff=t['max_rolloff'], zero_phase=t['zero_phase'], p=t['p']))
        if 'band_stop_filter' in modules:
            t = modules['band_stop_filter']
            augs.append(AA.BandStopFilter(min_center_freq=t['min_center_freq'], max_center_freq=t['max_center_freq'], min_bandwidth_fraction=t['min_bandwidth_fraction'],
                                          max_bandwidth_fraction=t['max_bandwidth_fraction'], min_rolloff=t['min_rolloff'], max_rolloff=t['max_rolloff'], zero_phase=t['zero_phase'], p=t['p']))
        if 'gain' in modules:
            t = modules['gain']
            augs.append(AA.Gain(min_gain_db=t['min_gain_db'], max_gain_db=t['max_gain_db'], p=t['p']))
        if 'high_pass_filter' in modules:
            t = modules['high_pass_filter']
            augs.append(AA.HighPassFilter(min_cutoff_freq=t['min_cutoff_freq'], max_cutoff_freq=t['max_cutoff_freq'], min_rolloff=t['min_rolloff'], max_rolloff=t['max_rolloff'], zero_phase=t['zero_phase'], p=t['p']))
        if 'limiter' in modules:
            t = modules['limiter']
            augs.append(AA.Limiter(min_threshold_db=t['min_threshold_db'], max_threshold_db=t['max_threshold_db'], min_attack=t['min_attack'], max_attack=t['max_attack'],
                                   min_release=t['min_release'], max_release=t['max_release'], threshold_mode=t['threshold_mode'], p=t['p']))
        if 'low_pass_filter' in modules:
            t = modules['low_pass_filter']
            augs.append(AA.LowPassFilter(min_cutoff_freq=t['min_cutoff_freq'], max_cutoff_freq=t['max_cutoff_freq'], min_rolloff=t['min_rolloff'], max_rolloff=t['max_rolloff'], zero_phase=t['zero_phase'], p=t['p']))
        if 'pitch_shift' in modules:
            t = modules['pitch_shift']
            augs.append(AA.PitchShift(min_semitones=t['min_semitones'], max_semitones=t['max_semitones'], p=t['p']))
        if 'seven_band_parametric_eq' in modules:
            t = modules['seven_band_parametric_eq']
            augs.append(AA.SevenBandParametricEQ(min_gain_db=t['min_gain_db'], max_gain_db=t['max_gain_db'], p=t['p']))
        if 'time_stretch' in modules:
            t = modules['time_stretch']
            augs.append(PyTimeStretch(min_rate=t['min_rate'], max_rate=t['max_rate'], leave_length_unchanged=t['leave_length_unchanged'], p=t['p']))
        if 'mixup' in modules:
            t = modules['mixup']
            self.mixup = BatchDataMixup(alpha=t['alpha'], beta=t['beta'], p=t['p'])
        if 'noise' in modules:
            t = modules['noise']
            augs.append(TemplateNoise(
                t.get('p', 0.2),
                manifest_path=t.get('manifest_path', None),
                min_snr_db=t.get('min_snr_db', 10),
                max_snr_db=t.get('max_snr_db', 50),
                max_gain_db=t.get('max_gain_db', 300),
                audio_tar_filepaths=t.get('audio_tar_filepaths', None),
                shuffle_n=t.get('shuffle_n', 100),
                orig_sr=t.get('orig_sr', 16000),
            ))

        self.audio_augmentations = AA.Compose(augs, shuffle=shuffle)

    @typecheck()
    @torch.no_grad()
    def forward(self, input_signal, length, sample_rate):
        if not self.enabled:
            return input_signal, length

        isig_cpu = input_signal.cpu().numpy()
        for i in range(isig_cpu.shape[0]):
            t = isig_cpu[i][:length[i]]
            # mixup here
            if self.mixup:
                if not self.mixup.are_parameters_frozen or self.mixup.parameters["should_apply"] is None:
                    self.mixup.randomize_parameters(input_signal, sample_rate)
                if self.mixup.parameters["should_apply"]:
                    t = self.mixup.apply(samples=t, batch=isig_cpu, length=length)

            t = self.audio_augmentations(samples=t, sample_rate=sample_rate)
            isig_cpu[i][:length[i]] = t

        input_signal = torch.from_numpy(isig_cpu).to(input_signal.device)

        # stretch in the end
        aug_stretch = next((x for x in self.audio_augmentations.transforms if type(x) is PyTimeStretch), None)
        if aug_stretch:
            for i in range(input_signal.shape[0]):
                if not aug_stretch.are_parameters_frozen or aug_stretch.parameters["should_apply"] is None:
                    aug_stretch.randomize_parameters(input_signal, sample_rate)
                if aug_stretch.parameters["should_apply"]:
                    t = aug_stretch.apply(samples=input_signal[i][None, None, :length[i]], sample_rate=sample_rate)
                    input_signal[i][:t.shape[-1]] = t[0, 0, :]
                    length[i] = t.shape[-1]

        return input_signal, length

    @property
    def input_types(self):
        """Returns definitions of module output ports.
        """
        return {
            "input_signal": NeuralType(('B', 'T'), AudioSignal()),
            "length": NeuralType(tuple('B'), LengthsType()),
            "sample_rate": NeuralType(tuple('T')),
        }

    @property
    def output_types(self):
        """Returns definitions of module output ports.
        """
        return {
            "augmented_signal": NeuralType(('B', 'T'), AudioSignal()),
            "length": NeuralType(tuple('B'), LengthsType()),
        }

    def save_to(self, save_path: str):
        pass

    @classmethod
    def restore_from(cls, restore_path: str):
        pass


class PyTimeStretch(BaseWaveformTransform):
    """Time stretch the signal without changing the pitch"""

    supports_multichannel = True

    def __init__(
        self,
        min_rate: float = 0.8,
        max_rate: float = 1.25,
        leave_length_unchanged: bool = True,
        p: float = 0.5,
    ):
        super().__init__(p)
        assert min_rate >= 0.1
        assert max_rate <= 10
        assert min_rate <= max_rate
        self.min_rate = min_rate
        self.max_rate = max_rate
        self.leave_length_unchanged = leave_length_unchanged

    def randomize_parameters(self, samples: np.ndarray, sample_rate: int):
        super().randomize_parameters(samples, sample_rate)
        if self.parameters["should_apply"]:
            """
            If rate > 1, then the signal is sped up.
            If rate < 1, then the signal is slowed down.
            """
            self.parameters["rate"] = random.uniform(self.min_rate, self.max_rate)

    def apply(self, samples: np.ndarray, sample_rate: int):

        # do nothing if ndarray, only if torch tensor
        if type(samples) is not torch.Tensor:
            return samples

        slen = samples.shape[-1]
        rate = int(np.round(self.parameters["rate"] * 100))
        if rate != 100:
            time_stretched_samples = time_stretch(samples, Fraction(rate, 100), sample_rate)
        else:
            time_stretched_samples = samples

        if time_stretched_samples.shape[-1] > slen:
            time_stretched_samples = time_stretched_samples[..., :slen]
        return time_stretched_samples


class TemplateNoise(BaseWaveformTransform):
    """Time stretch the signal without changing the pitch"""

    supports_multichannel = True

    def __init__(
        self,
        p: float = 0.5,
        manifest_path: str = None,
        min_snr_db: float = 10,
        max_snr_db: float = 50,
        max_gain_db: float = 300.0,
        audio_tar_filepaths: str = None,
        shuffle_n: int = 100,
        orig_sr: float = 16000,
    ):
        super().__init__(p)
        if p > 0:
            self.augmentor = NoisePerturbation(manifest_path, min_snr_db, max_snr_db, max_gain_db, None, audio_tar_filepaths, shuffle_n, orig_sr)

    def randomize_parameters(self, samples: np.ndarray, sample_rate: int):
        super().randomize_parameters(samples, sample_rate)
        if self.parameters["should_apply"]:
            self.parameters["snr_db"] = random.uniform(self.augmentor._min_snr_db, self.augmentor._max_snr_db)

    def apply(self, samples: np.ndarray, sample_rate: int):

        audio_record = random.sample(self.augmentor._manifest.data, 1)[0]
        audio_file = audio_record.audio_file
        offset = 0 if audio_record.offset is None else audio_record.offset

        data_duration = len(samples) / sample_rate

        # calculate noise segment to use
        offset = random.uniform(offset, offset + max(0, audio_record.duration - data_duration))
        duration = data_duration

        noise = AudioSegment.from_file(audio_file, target_sr=sample_rate, offset=offset, duration=duration)

        snr_db = self.parameters["snr_db"]
        data_rms_db = 10 * np.log10(np.mean(samples ** 2))

        noise_gain_db = data_rms_db - noise.rms_db - snr_db
        noise_gain_db = min(noise_gain_db, self.augmentor._max_gain_db)

        # calculate noise segment to use
        start_time = random.uniform(0.0, noise.duration - len(samples) / sample_rate)
        if noise.duration > (start_time + data_duration):
            noise.subsegment(start_time=start_time, end_time=start_time + data_duration)

        # adjust gain for snr purposes and superimpose
        noise.gain_db(noise_gain_db)

        if noise._samples.shape[0] < samples.shape[0]:
            noise_idx = random.randint(0, samples.shape[0] - noise._samples.shape[0])
            samples[noise_idx: noise_idx + noise._samples.shape[0]] += noise._samples
        else:
            samples += noise._samples

        return samples


class BatchDataMixup(BaseWaveformTransform):
    """ adds random sample from a batch to the target sample"""

    supports_multichannel = True

    def __init__(
        self,
        alpha: float = 8,
        beta: float = 1,
        p: float = 0.5,
    ):
        super().__init__(p)
        self.alpha = alpha
        self.beta = beta

    def randomize_parameters(self, samples: np.ndarray, sample_rate: int):
        super().randomize_parameters(samples, sample_rate)
        if self.parameters["should_apply"]:
            self.parameters["lambda"] = np.random.beta(self.alpha, self.beta)

    def apply(self, samples: np.ndarray, batch: np.ndarray, length):

        ix = np.random.randint(0, batch.shape[0])
        minlen = min(samples.shape[-1], length[ix])
        lam = self.parameters["lambda"]
        samples[:minlen] = lam * samples[:minlen] + (1 - lam) * batch[ix, :minlen]
        return samples


class SpectrogramStretch(NeuralModule):

    def __init__(self, enabled: bool, n_freq: int, min_rate: float, max_rate: float, p: float):
        super(SpectrogramStretch, self).__init__()
        self.enabled = enabled
        self.min_rate = min_rate
        self.max_rate = max_rate
        self.p = p
        self.stretch = T.TimeStretch(n_freq=n_freq)

    @typecheck()
    @torch.no_grad()
    def forward(self, input_spectrogram, length):
        if not self.enabled or random.random() > self.p:
            return input_spectrogram, length

        for i in range(input_spectrogram.shape[0]):
            t = input_spectrogram[i][:,:length[i]]
            rate = random.uniform(self.min_rate, self.max_rate)
            t = torch.real(self.stretch(t, rate))
            t = t[:,:min(input_spectrogram.shape[-1], t.shape[-1])]
            input_spectrogram[i][:,:t.shape[1]] = t
            length[i] = t.shape[1]

        return input_spectrogram, length

    @property
    def input_types(self):
        """Returns definitions of module output ports.
        """
        return {
            "input_spectrogram": NeuralType(('B', 'D', 'T'), SpectrogramType()),
            "length": NeuralType(tuple('B'), LengthsType()),
        }

    @property
    def output_types(self):
        """Returns definitions of module output ports.
        """
        return {
            "stretched_signal": NeuralType(('B', 'D', 'T'), SpectrogramType()),
            "length": NeuralType(tuple('B'), LengthsType()),
        }

    def save_to(self, save_path: str):
        pass

    @classmethod
    def restore_from(cls, restore_path: str):
        pass


class CropOrPadAugmentation(NeuralModule):
    """
    Pad or Crop the incoming Spectrogram to a certain shape.
    Args:
        audio_length (int): the final number of timesteps that is required.
            The signal will be either padded or cropped temporally to this
            size.
    """

    def __init__(self, audio_length, random_crop = False):
        super(CropOrPadAugmentation, self).__init__()
        self.audio_length = audio_length
        self.random_crop = random_crop

    @typecheck()
    @torch.no_grad()
    def forward(self, input_signal, length):
        return input_signal, length


    @typecheck()
    @torch.no_grad()
    def forward(self, input_signal, length):
        image = input_signal
        image_lengths = length.cpu().detach().numpy()
        num_images = image.shape[0]

        audio_length = self.audio_length

        offsets = []
        cutout_images = []
        # Crop long signal
        for idx in range(0, num_images):
            image_len = image_lengths[idx]
            if image_len > audio_length:  # randomly slice
                if self.random_crop:
                    offset = randint(0, image_len - audio_length)
                else:
                    offset = 0
                cutout_images.append(image[idx : idx + 1, :, offset : offset + audio_length])
                offsets.append(offset)

            else:  # pad only right
                pad_right = (audio_length - image_len)
                cutout_images.append(torch.nn.functional.pad(image[idx : idx+1, :, 0:image_len], [0, pad_right], mode="constant", value=0))
                offsets.append(0)

        # Replace dynamic length sequences with static number of timesteps
        image = torch.cat(cutout_images, dim=0)
        del cutout_images

        length = (length * 0) + audio_length

        return image, length, torch.tensor(offsets, device=length.device)

    @property
    def input_types(self):
        """Returns definitions of module output ports.
        """
        return {
            "input_signal": NeuralType(('B', 'D', 'T'), SpectrogramType()),
            "length": NeuralType(tuple('B'), LengthsType()),
        }

    @property
    def output_types(self):
        """Returns definitions of module output ports.
        """
        return {
            "processed_signal": NeuralType(('B', 'D', 'T'), SpectrogramType()),
            "processed_length": NeuralType(tuple('B'), LengthsType()),
            "offsets": NeuralType(tuple('B'), LengthsType()),
        }

    def save_to(self, save_path: str):
        pass

    @classmethod
    def restore_from(cls, restore_path: str):
        pass
