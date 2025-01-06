import json
import logging
import os
from collections import namedtuple, Counter
from typing import Optional, Dict, List, Union, Any
import numpy as np
import torch
from nemo.collections.common.parts.preprocessing import manifest
from nemo.collections.common.parts.preprocessing.collections import _Collection
from nemo.core import Dataset
from nemo.core.neural_types import NeuralType, AudioSignal, LengthsType, LabelsType, ProbsType


class AudioLabelsDataset(Dataset):
    """
    Dataset that loads tensors via a json file containing paths to audio files,labels, and durations and offsets(in seconds). Each new line is a different sample.
    JSON files should be of the following format::
        {"audio_filepath": "/path/to/audio_wav_0.wav", "duration": time_in_sec_0, "label": target_label_0, "offset": offset_in_sec_0, "labels": sequence_of_target_labels}
        ...
        {"audio_filepath": "/path/to/audio_wav_n.wav", "duration": time_in_sec_n, "label": target_label_n, "offset": offset_in_sec_n, , "labels": sequence_of_target_labels}
    Args:
        manifest_filepath (str): Dataset parameter. Path to JSON containing data.
        labels (list): Dataset parameter. List of target classes that can be output by the speaker recognition model.
        featurizer
        min_duration (float): Dataset parameter. All training files which have a duration less than min_duration are dropped. Note: Duration is read from the manifest JSON.
        max_duration (float): Dataset parameter. All training files which have a duration more than max_duration are dropped. Note: Duration is read from the manifest JSON.
        trim (bool): Whether to use trim silence from beginning and end of audio signal using librosa.effects.trim(). Defaults to False.
    """

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:

        output_types = {
            'audio_signal': NeuralType(('B', 'T'), AudioSignal(freq=self._sample_rate) if self is not None and hasattr(self, '_sample_rate') else AudioSignal(), ),
            'a_sig_length': NeuralType(tuple('B'), LengthsType()),
            'label': NeuralType(tuple('B'), LabelsType()),
            'label_length': NeuralType(tuple('B'), LengthsType()),
            'labels': NeuralType(('B', 'T'), LabelsType()),
            'labels_length': NeuralType(tuple('B'), LengthsType()),
            'y': NeuralType(('B', 'D'), ProbsType()),
        }
        return output_types

    def __init__(
            self,
            *,
            manifest_filepath: str,
            labels: List[str],
            featurizer,
            min_duration: Optional[float] = 0.1,
            max_duration: Optional[float] = None,
            trim: bool = False,
            normalize_audio: bool = False,
            one_clip_length_in_sec: float = None,
            nwin_clip_length_in_sec: float = None,
            subsample_each_epoch: float = 0
    ):
        super().__init__()
        self.collection = AudioLabels(
            manifests_files=manifest_filepath.split(','),
            min_duration=min_duration,
            max_duration=max_duration,
        )

        self.one_clip_length_in_sec = one_clip_length_in_sec
        self.nwin_clip_length_in_sec = nwin_clip_length_in_sec
        self.featurizer = featurizer
        self.trim = trim
        self.normalize_audio = normalize_audio
        self.subsample_each_epoch = subsample_each_epoch

        self.labels = labels if labels else self.collection.uniq_labels
        self.num_classes = len(self.labels) if self.labels is not None else 1
        self.label2id, self.id2label = {}, {}
        for label_id, label in enumerate(self.labels):
            self.label2id[label] = label_id
            self.id2label[label_id] = label

        self.index_map = None

        for idx in range(len(self.labels[:5])):
            logging.debug(" label id {} and its mapped label {}".format(idx, self.id2label[idx]))

    def new_subset(self):
        # add all not from bgrd
        if self.subsample_each_epoch == 0:
            return

        if self.subsample_each_epoch <= 1:
            # subsample the bgrd class as given by the subsample_each_epoch ratio

            fgix = [x for x in range(len(self.collection)) if self.collection[x].label != 'bgrd']

            lbls = [x.orig_label for x in self.collection if x.label != 'bgrd']
            c_lbls = Counter(lbls)
            if len(c_lbls) > 1 and False:  # disable idf sampling of fg categories
                c_lbls = {k: np.log(len(lbls) / v) for k, v in c_lbls.items()}  # idf
                lbl_p = [c_lbls[x.orig_label] for x in self.collection if x.label != 'bgrd']
                lbl_p = np.divide(lbl_p, np.sum(lbl_p))

                fgix = list(np.random.choice(fgix, len(fgix) // 2, replace=True, p=lbl_p))

            fgcnt = len(fgix)
            bgcnt = int(fgcnt / self.subsample_each_epoch) - fgcnt

            if False:  # sampling according to https://arxiv.org/pdf/2102.01243.pdf
                bgix = [x for x in range(len(self.collection)) if self.collection[x].label == 'bgrd']
                fgw = np.array([x.frequency for x in self.collection if x.label != 'bgrd'])
                ixmap = np.random.choice(fgix, fgcnt, replace=True, p=fgw / sum(fgw)).tolist()
                bgw = np.array([x.frequency for x in self.collection if x.label == 'bgrd'])
                ixmap = ixmap + np.random.choice(bgix, bgcnt, replace=True, p=bgw / sum(bgw)).tolist()
            else:
                ixmap = fgix
                truebgix = [x for x in range(len(self.collection)) if self.collection[x].label == 'bgrd' and self.collection[x].orig_label.startswith('BG')]
                restbgix = [x for x in range(len(self.collection)) if self.collection[x].label == 'bgrd' and not self.collection[x].orig_label.startswith('BG')]
                if len(truebgix)>len(restbgix):
                    ixmap = ixmap + np.random.choice(restbgix, min(len(restbgix), bgcnt // 2), replace=False).tolist()
                    ixmap = ixmap + np.random.choice(truebgix, min(len(truebgix), bgcnt+fgcnt-len(ixmap)), replace=False).tolist()
                else:
                    ixmap = ixmap + np.random.choice(truebgix, min(len(truebgix), bgcnt // 2), replace=False).tolist()
                    ixmap = ixmap + np.random.choice(restbgix, min(len(restbgix), bgcnt+fgcnt-len(ixmap)), replace=False).tolist()

                # if len(ixmap) < fgcnt + bgcnt:
                #     ixmap = ixmap + np.random.choice(truebgix + restbgix, fgcnt + bgcnt - len(ixmap), replace=True).tolist()
        else:
            # select subsample_each_epoch items from the dataset, keep all labels, subsample according to probabilities
            c = Counter([x.label for x in self.collection])
            p = np.array(list(c.values()))
            p = np.sqrt(p)
            p = p / np.sum(p)
            n = np.round(self.subsample_each_epoch * p).astype(int)
            n[n<1] = 1  # ensure at least one per class
            n[n.argmax()] += self.subsample_each_epoch - n.sum()  # ensure correct total count
            ixmap = []
            for i,k in enumerate(c.keys()):
                ixs = [i for i, x in enumerate(self.collection) if x.label == k]
                ixmap = ixmap + np.random.choice(ixs, n[i], replace=True).tolist()

        ixmap.sort()
        self.index_map = ixmap

    def __len__(self):
        if self.index_map is not None:
            return len(self.index_map)
        return len(self.collection)

    def __getitem__(self, index):
        if self.index_map is not None:
            index = self.index_map[index]

        sample = self.collection[index]

        offset = sample.offset

        if offset is None:
            offset = 0
        duration = sample.duration

        if self.nwin_clip_length_in_sec is not None and duration > self.nwin_clip_length_in_sec:
            offset = offset + np.random.uniform(0, sample.duration-self.nwin_clip_length_in_sec, 1)
            duration = self.nwin_clip_length_in_sec

        features = self.featurizer.process(sample.audio_file, offset=offset, duration=duration, trim=self.trim, channel_selector=sample.channel_selector)
        f, fl = features, torch.tensor(features.shape[0]).long()

        t = torch.tensor(self.label2id[sample.label]).long()

        tl = torch.tensor(1).long()  # For compatibility with collate_fn used later

        sl = None
        if sample.labels is not None:
            sl = torch.tensor([self.label2id[x] for x in sample.labels.split(' \t\n\r')]).long()

        sllen = None
        if sample.labels_len is not None:
            sllen = torch.tensor(sample.labels_len).long()

        y = None
        if sample.y is not None:
            y = torch.tensor(sample.y).float()

        return f, fl, t, tl, sl, sllen, y

    def _collate_fn(self, batch):
        return _speech_collate_fn(self, batch, pad_id=0)


def _speech_collate_fn(self, batch, pad_id):
    """collate batch of audio sig, audio len, tokens, tokens len
    Args:
        batch (Optional[FloatTensor], Optional[LongTensor], LongTensor,
               LongTensor):  A tuple of tuples of signal, signal lengths,
               encoded tokens, and encoded tokens length.  This collate func
               assumes the signals are 1d torch tensors (i.e. mono audio).
    """
    _, audio_lengths, _, tokens_lengths, _, labels_len, _ = zip(*batch)
    max_audio_len = 0
    has_audio = audio_lengths[0] is not None
    if has_audio:
        max_audio_len = max(audio_lengths).item()
    max_tokens_len = max(tokens_lengths).item()
    max_labels_len = 0
    if labels_len[0] is not None:
        max_labels_len = max(labels_len).item()

    audio_signal, tokens, labels, ys = [], [], [],  []

    is_hdf = (len(batch[0][0].shape) > 1 and batch[0][0].shape[1] > 2)
    for sig, sig_len, tokens_i, tokens_i_len, labels_i, labels_i_len, y_i in batch:
        if has_audio:
            sig_len = sig_len.item()
            if sig_len < max_audio_len:
                if is_hdf:
                    pad = (0, 0, 0, max_audio_len - sig_len)
                else:
                    pad = (0, max_audio_len - sig_len)
                sig = torch.nn.functional.pad(sig, pad)
            if not is_hdf and self.normalize_audio:
                sig = normalize(sig)
            if is_hdf:
                audio_signal.append(torch.reshape(sig, (sig.shape[0] * sig.shape[1],)))
            else:
                audio_signal.append(sig)

        tokens_i_len = tokens_i_len.item()
        if tokens_i_len < max_tokens_len:
            pad = (0, max_tokens_len - tokens_i_len)
            tokens_i = torch.nn.functional.pad(tokens_i, pad, value=pad_id)
        tokens.append(tokens_i)
        labels_i_len = labels_i_len if labels_i_len is not None else 0
        if labels_i_len < max_labels_len:
            pad = (0, max_labels_len - labels_i_len)
            labels_i = torch.nn.functional.pad(labels_i, pad, value=0)
        labels.append(labels_i)
        ys.append(y_i)

    if has_audio:
        audio_signal = torch.stack(audio_signal)
        audio_lengths = torch.stack(audio_lengths)
    else:
        audio_signal, audio_lengths = None, None
    tokens = torch.stack(tokens)
    tokens_lengths = torch.stack(tokens_lengths)

    if max_labels_len > 0:
        labels = torch.stack(labels)
        labels_lengths = torch.stack(labels_len)
    else:
        labels, labels_lengths = None, None
    ys = torch.stack(ys)

    return audio_signal, audio_lengths, tokens, tokens_lengths, labels, labels_lengths, ys


class AudioLabels(_Collection):
    OUTPUT_TYPE = namedtuple(typename='SpeechLabelsEntity', field_names='audio_file duration label orig_label offset labels labels_len channel_selector y frequency', )

    """`SpeechLabels` collector from structured json files."""
    """Instantiates audio-label manifest with filters and preprocessing.
    Args:
        audio_files: List of audio files.
        durations: List of float durations.
        labels: List of labels.
        offsets: List of offsets or None.
        min_duration: Minimum duration to keep entry with (default: None).
        max_duration: Maximum duration to keep entry with (default: None).
        max_number: Maximum number of samples to collect.
        do_sort_by_duration: True if sort samples list by duration.
        index_by_file_id: If True, saves a mapping from filename base (ID) to index in data.
    """

    def __init__(self, manifests_files: Union[str, List[str]], *args, **kwargs):
        """Parse lists of audio files, durations and transcripts texts.
        Args:
            manifests_files: Either single string file or list of such -
                manifests to yield items from.
            *args: Args to pass to `SpeechLabel` constructor.
            **kwargs: Kwargs to pass to `SpeechLabel` constructor.
        """
        audio_files, durations, labels, orig_labels, offsets, labelss, labelss_len, channels, y, frequency = [], [], [], [], [], [], [], [], [], []

        for item in manifest.item_iter(manifests_files, parse_func=self.__parse_item):
            audio_files.append(item['audio_file'])
            durations.append(item['duration'])
            labels.append(item['label'])
            orig_labels.append(item['orig_label']),
            offsets.append(item['offset'])
            labelss.append(item['labels'])
            labelss_len.append(item['labels_len'])
            channels.append(item['channel_selector'])
            y.append(item['y'])
            frequency.append(item['frequency'])

        data = self.process(audio_files, durations, labels, orig_labels, offsets, labelss, labelss_len, channels, y, frequency, *args, **kwargs)
        super().__init__(data)

    def process(self, audio_files: List[str], durations: List[float], labels: List[Union[int, str]], orig_labels: List[Union[int, str]], offsets: List[Optional[float]],
                labelss: List[str], labelss_len: List[int], channels: List[int], ys: List[List[float]], frequencies: List[float], min_duration: Optional[float] = None, max_duration: Optional[float] = None,
                max_number: Optional[int] = None, do_sort_by_duration: bool = False, index_by_file_id: bool = False):

        if index_by_file_id:
            self.mapping = {}
        output_type = self.OUTPUT_TYPE
        data, duration_filtered = [], 0.0
        for audio_file, duration, label, orig_label, offset, label_s, label_s_len, channel, y, frequency \
                in zip(audio_files, durations, labels, orig_labels, offsets, labelss, labelss_len, channels, ys, frequencies):
            # Duration filters.
            if min_duration is not None and duration < min_duration:
                duration_filtered += duration
                continue

            if max_duration is not None and max_duration > 0 and duration > max_duration:
                duration_filtered += duration
                continue

            data.append(output_type(audio_file, duration, label, orig_label, offset, label_s, label_s_len, channel, y, frequency))

            if index_by_file_id:
                file_id, _ = os.path.splitext(os.path.basename(audio_file))
                self.mapping[file_id] = len(data) - 1

            # Max number of entities filter.
            if len(data) == max_number:
                break

        if do_sort_by_duration:
            if index_by_file_id:
                logging.warning("Tried to sort dataset by duration, but cannot since index_by_file_id is set.")
            else:
                data.sort(key=lambda entity: entity.duration)

        logging.info(
            "Filtered duration for loading collection is %f.", duration_filtered,
        )
        self.uniq_labels = sorted(set(map(lambda x: x.label, data)))
        logging.info("# {} files loaded accounting to # {} labels".format(len(data), len(self.uniq_labels)))

        return data

    def __parse_item(self, line: str, manifest_file: str) -> Dict[str, Any]:
        item = json.loads(line)

        # Audio file
        if 'audio_filename' in item:
            item['audio_file'] = item.pop('audio_filename')
        elif 'audio_filepath' in item:
            item['audio_file'] = item.pop('audio_filepath')
        else:
            raise ValueError(
                f"Manifest file has invalid json line " f"structure: {line} without proper audio file key."
            )
        item['audio_file'] = os.path.expanduser(item['audio_file'])

        # Duration.
        if 'duration' not in item:
            raise ValueError(f"Manifest file has invalid json line " f"structure: {line} without proper duration key.")

        if not 'label' in item:
            raise ValueError(f"Manifest file has invalid json line " f"structure: {line} without proper label key.")

        item = dict(
            audio_file=item['audio_file'],
            duration=item['duration'],
            label=item['label'],
            orig_label=str(item.get('text', None)),
            offset=item.get('offset', None),
            labels=item.get('labels', None),
            labels_len=item.get('labels_len', None),
            channel_selector=item.get('channel_selector', 0),
            y=item.get('y', None),
            frequency=item.get('frequency', 1.0),
        )

        return item


class AudioStreamLabelsDataset(AudioLabelsDataset):
    """
    Dataset that loads tensors via a json file containing paths to audio
    files, command class, and durations (in seconds). Each new line is a
    different sample. Example below:
    {"audio_filepath": "/path/to/audio_wav_0.wav", "duration": time_in_sec_0, "label": \
        target_label_0, "offset": offset_in_sec_0}
    ...
    {"audio_filepath": "/path/to/audio_wav_n.wav", "duration": time_in_sec_n, "label": \
        target_label_n, "offset": offset_in_sec_n}
    Args:
        manifest_filepath (str): Path to manifest json as described above. Can
            be comma-separated paths.
        labels (Optional[list]): String containing all the possible labels to map to
            if None then automatically picks from ASRSpeechLabel collection.
        min_duration (float): Dataset parameter.
            All training files which have a duration less than min_duration
            are dropped. Note: Duration is read from the manifest JSON.
            Defaults to 0.1.
        max_duration (float): Dataset parameter.
            All training files which have a duration more than max_duration
            are dropped. Note: Duration is read from the manifest JSON.
            Defaults to None.
        trim (bool): Whether to use trim silence from beginning and end
            of audio signal using librosa.effects.trim().
            Defaults to False.
        window_length_in_sec (float): length of window/slice (in seconds)
            Use this for speaker recognition and VAD tasks.
        shift_length_in_sec (float): amount of shift of window for generating the frame for VAD task in a batch
            Use this for VAD task during inference.
        normalize_audio (bool): Whether to normalize audio signal.
            Defaults to False.
        is_regression_task (bool): Whether the dataset is for a regression task instead of classification
    """

    def __init__(
            self,
            *,
            manifest_filepath: str,
            labels: List[str],
            featurizer,
            min_duration: Optional[float] = 0.1,
            max_duration: Optional[float] = None,
            trim: bool = False,
            window_length_in_sec: Optional[float] = 8,
            shift_length_in_sec: Optional[float] = 1,
            normalize_audio: bool = False,
            one_clip_length_in_sec: float = None,
            nwin_clip_length_in_sec: float = None,
            subsample_each_epoch: float = 0,
    ):
        self.window_length_in_sec = window_length_in_sec
        self.shift_length_in_sec = shift_length_in_sec
        self.normalize_audio = normalize_audio
        self.one_clip_length_in_sec = one_clip_length_in_sec
        self.nwin_clip_length_in_sec = nwin_clip_length_in_sec

        logging.debug("Window/slice length considered for collate func is {}".format(self.window_length_in_sec))
        logging.debug("Shift length considered for collate func is {}".format(self.shift_length_in_sec))

        super().__init__(
            manifest_filepath=manifest_filepath,
            labels=labels,
            featurizer=featurizer,
            min_duration=min_duration,
            max_duration=max_duration,
            trim=trim,
            one_clip_length_in_sec=one_clip_length_in_sec,
            nwin_clip_length_in_sec=nwin_clip_length_in_sec,
            subsample_each_epoch=subsample_each_epoch
        )

    def fixed_seq_collate_fn(self, batch):
        return _fixed_seq_collate_fn(self, batch)

    def vad_frame_seq_collate_fn(self, batch):
        return _vad_frame_seq_collate_fn(self, batch)


def _fixed_seq_collate_fn(self, batch):
    """collate batch of audio sig, audio len, tokens, tokens len
        Args:
            batch (Optional[FloatTensor], Optional[LongTensor], LongTensor,
                LongTensor):  A tuple of tuples of signal, signal lengths,
                encoded tokens, and encoded tokens length.  This collate func
                assumes the signals are 1d torch tensors (i.e. mono audio).
        """
    _, audio_lengths, _, tokens_lengths = zip(*batch)

    has_audio = audio_lengths[0] is not None
    fixed_length = int(max(audio_lengths))

    audio_signal, tokens, new_audio_lengths = [], [], []
    for sig, sig_len, tokens_i, _ in batch:
        if has_audio:
            sig_len = sig_len.item()
            chunck_len = sig_len - fixed_length

            if chunck_len < 0:
                repeat = fixed_length // sig_len
                rem = fixed_length % sig_len
                sub = sig[-rem:] if rem > 0 else torch.tensor([])
                rep_sig = torch.cat(repeat * [sig])
                sig = torch.cat((rep_sig, sub))
            new_audio_lengths.append(torch.tensor(fixed_length))

            audio_signal.append(sig)

        tokens.append(tokens_i)

    if has_audio:
        audio_signal = torch.stack(audio_signal)
        audio_lengths = torch.stack(new_audio_lengths)
    else:
        audio_signal, audio_lengths = None, None
    tokens = torch.stack(tokens)
    tokens_lengths = torch.stack(tokens_lengths)

    return audio_signal, audio_lengths, tokens, tokens_lengths


def _vad_frame_seq_collate_fn(self, batch):
    """collate batch of audio sig, audio len, tokens, tokens len
    Args:
        batch (Optional[FloatTensor], Optional[LongTensor], LongTensor,
            LongTensor):  A tuple of tuples of signal, signal lengths,
            encoded tokens, and encoded tokens length.  This collate func
            assumes the signals are 1d torch tensors (i.e. mono audio).
            batch size equals to 1.
    """
    if len(batch[0][0].shape) > 1 and batch[0][0].shape[1] > 2:
        #  hdf5 stored pcen or other spectrograms

        slice_length = int(np.round(self.window_length_in_sec / self.featurizer.hop_size))
        _, audio_lengths, _, tokens_lengths, _, labels_len, _ = zip(*batch)
        shift = int(np.round(self.shift_length_in_sec / self.featurizer.hop_size))
        has_audio = audio_lengths[0] is not None

        audio_signal, num_slices, tokens, audio_lengths, ys = [], [], [], [], []

        for sig, sig_len, tokens_i, tokens_i_len, labels_i, labels_i_len, y_i in batch:
            if sig_len < slice_length:
                right_padded = slice_length - sig_len
                sig = torch.nn.functional.pad(sig, (right_padded, sig.shape[1]))

            sig_len = sig.shape[0]

            if has_audio:
                slices = (sig_len - slice_length) // shift + 1
                for slice_id in range(slices):
                    start_idx = slice_id * shift
                    end_idx = start_idx + slice_length
                    signal = sig[start_idx:end_idx]
                    audio_signal.append(torch.reshape(signal, (signal.shape[0]*signal.shape[1],)))

                num_slices.append(slices)
                tokens.extend([tokens_i] * slices)
                audio_lengths.extend([slice_length] * slices)
                ys.extend([y_i] * slices)

        if has_audio:
            audio_signal = torch.stack(audio_signal)
            audio_lengths = torch.tensor(audio_lengths)
        else:
            audio_signal, audio_lengths = None, None

        tokens = torch.stack(tokens)
        tokens_lengths = torch.tensor(num_slices)
        ys = torch.stack(ys)
        return audio_signal, audio_lengths, tokens, tokens_lengths, ys

    else:
        #  audio
        slice_length = int(self.featurizer.sample_rate * self.one_clip_length_in_sec)
        _, audio_lengths, _, tokens_lengths, _, labels_len, _ = zip(*batch)
        shift = int(self.featurizer.sample_rate * self.shift_length_in_sec)
        has_audio = audio_lengths[0] is not None

        append_len_start = 0  # slice_length // 2
        append_len_end = 0  # slice_length - slice_length // 2

        audio_signal, num_slices, tokens, audio_lengths, ys = [], [], [], [], []

        for sig, sig_len, tokens_i, tokens_i_len, labels_i, labels_i_len, y_i in batch:
            right_padded = 0
            if sig_len < slice_length:
                right_padded = slice_length - sig_len
                sig = torch.nn.functional.pad(sig, (0, right_padded))

            if self.normalize_audio:
                sig = normalize(sig)
            # start = torch.zeros(append_len_start)
            # end = torch.zeros(append_len_end)
            # sig = torch.cat((start, sig, end))
            sig = torch.nn.functional.pad(sig, (append_len_start, max(append_len_end - right_padded, 0)))
            sig_len = sig.shape[0]

            if has_audio:
                slices = (sig_len - slice_length) // shift + 1
                for slice_id in range(slices):
                    start_idx = slice_id * shift
                    end_idx = start_idx + slice_length
                    signal = sig[start_idx:end_idx]
                    audio_signal.append(signal)

                num_slices.append(slices)
                tokens.extend([tokens_i] * slices)
                audio_lengths.extend([slice_length] * slices)
                ys.extend([y_i] * slices)

        if has_audio:
            audio_signal = torch.stack(audio_signal)
            audio_lengths = torch.tensor(audio_lengths)
        else:
            audio_signal, audio_lengths = None, None

        tokens = torch.stack(tokens)
        tokens_lengths = torch.tensor(num_slices)
        ys = torch.stack(ys)
        return audio_signal, audio_lengths, tokens, tokens_lengths, ys


def normalize(signal):
    """normalize signal
    Args:
        signal(FloatTensor): signal to be normalized.
    """
    signal_minusmean = signal - signal.mean()
    return signal_minusmean / signal_minusmean.abs().max()
