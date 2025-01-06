import argparse

import librosa
import progressbar
import torch
import numpy as np
from scipy.ndimage import median_filter
from pathlib import Path

from models import EncoderDecoderClassificationModel
from infer import get_overlapping_probs


def get_model(m: str, runon: str):

    if str(m).lower().endswith('nemo'):
        model = EncoderDecoderClassificationModel.restore_from(m, strict=False)
    else:
        model = EncoderDecoderClassificationModel.load_from_checkpoint(m)

    model = model.to(runon)
    model.eval()
    return model


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description='Label file(s) with specified models and saves to Raven compatible annotation files.')
    parser.add_argument("audio_files", type=str, nargs='+', help='Any list of audio files or directories to process.')
    parser.add_argument('--models', required=False, type=str, nargs='+', default=None,
                        help='A list of models used for inference.')
    parser.add_argument('--channel', required=False, default=None, type=int, choices=range(1, 10),
                        help='Audio channel to process. If not given, all will be processed, otherwise only the specified channel. Count starts with 1!')
    parser.add_argument('--step_size', required=False, default=0.5, type=float, help='Period (in seconds) of stepping through the audio file. ')
    parser.add_argument('--median_filter', required=False, default=3, type=float, help='The width of smoothing median filter (in seconds) to be applied to labels.')
    parser.add_argument('--run_on', required=False, default='cuda', type=str, help='Run processing or cpu or cuda (gpu).')
    parser.add_argument('--batch_size', required=False, default=200, type=int, help='Batch size. If you have more GPU memory, increase this for faster processing.')
    parser.add_argument('--chunk_len', required=False, default=700, type=int, help='Audio chunk size. If audio file is larger, it will be cut into chunks of this size.')
    parser.add_argument('--suffix', required=False, default=None, type=str, help='Suffix to add to the resulting .txt file.')
    parser.add_argument('--true_thresh', required=False, default=None, type=float, help='Threshold for active class. Default is to take the max activation.')

    args = parser.parse_args()
    runon = args.run_on

    seg_models = {}
    for m in args.models:
        cls = m.split('-')[2]
        mod = get_model(m, runon)
        if cls in seg_models:
            seg_models[cls].append(mod)
        else:
            seg_models[cls] = [mod]

    files = []
    for filename in args.audio_files:
        p = Path(filename)
        if p.is_dir():
            files += [str(x) for x in p.glob('**/*.*') if x.suffix.lower() in ['.wav', '.mp3', '.flac']]
        else:
            files.append(str(p))

    med_filter = int(args.median_filter / args.step_size)
    sr = [x for x in seg_models.values()][0][0].cfg.sample_rate
    total_duration = 0
    for file in files:
        total_duration += librosa.get_duration(filename=file)

    widgets = [' [', progressbar.Timer(format='elapsed time: %(elapsed)s'), '] ', progressbar.Bar('*'), ' (', progressbar.ETA(), ') ', ]
    bar = progressbar.ProgressBar(max_value=total_duration, widgets=widgets).start()
    processed_duration = 0

    for file in files:
        print(f'Processing {file} ...')

        audio_len_s = librosa.get_duration(filename=file)
        audio_loc_s = 0
        is_last = False
        chunk_steps = int(np.round(args.chunk_len / args.step_size))
        olap = 0

        all_probs = {}
        while audio_loc_s < audio_len_s and is_last is False:

            dur_s = args.chunk_len + 10
            if audio_len_s - audio_loc_s < dur_s * 1.5:
                dur_s = audio_len_s - audio_loc_s + 10
                is_last = True

            x, sr = librosa.load(file, sr=sr, mono=False, offset=audio_loc_s, duration=dur_s)
            if x.ndim == 1:
                channels = [0]
            else:
                channels = range(0, x.shape[0])
            if args.channel is not None:
                channels = [int(args.channel)-1]

            for km in seg_models:
                for c in channels:
                    probs = get_overlapping_probs(x, c, seg_models[km], sr, args.step_size, args.batch_size, runon)

                    key = km + '_' + str(c)
                    if audio_loc_s == 0:
                        all_probs[key] = probs
                    else:
                        all_probs[key] = np.concatenate((all_probs[key][:-olap, ], probs))

            olap = probs.shape[0] - chunk_steps
            audio_loc_s += args.chunk_len
            processed_duration += dur_s - 10
            bar.update(processed_duration)

        all_labels = []

        for km in seg_models:
            for c in channels:
                key = km + '_' + str(c)

                probs = all_probs[key]
                if args.median_filter is not None and args.median_filter > 0:
                    probs = median_filter(probs, size=(med_filter, 1))
                    # lbl_idx = median_filter(lbl_idx, size=med_filter)

                if args.true_thresh is None:
                    lbl_idx = probs.argmax(axis=1)
                    probs_bin = np.zeros_like(probs)
                    probs_bin[np.arange(probs.shape[0]), lbl_idx] = 1
                else:
                    probs_bin = (probs > args.true_thresh) + 0
                for target_ix in np.where([x != 'bgrd' for x in seg_models[km][0].cfg.labels])[0]:
                    probs_target = probs_bin[:,target_ix]
                    probs_target = np.concatenate(([0], probs_target + 0, [0]))
                    positions = np.where(np.diff(probs_target) != 0)[0]
                    for i in range(0, len(positions),2):
                        all_labels.append((positions[i]*args.step_size, positions[i+1]*args.step_size, seg_models[km][0].cfg.labels[int(target_ix)], c+1))

        all_labels = sorted(all_labels, key=lambda x: (x[3]-1)*1000000 + x[0])
        lbls = "_".join(seg_models.keys())

        if args.suffix is not None:
            out_file = str(Path(file).with_suffix(f'.{lbls}.{args.suffix}.txt'))
        else:
            out_file = str(Path(file).with_suffix(f'.{lbls}.txt'))
        with open(out_file, 'w', encoding='utf8') as f:
            f.write(f'Selection\tView\tChannel\tBegin Time (s)\tEnd Time (s)\tLow Freq (Hz)\tHigh Freq (Hz)\tAnnotation\n')
            for (i, x) in enumerate(all_labels):
                f.write(f'{i+1}\tSpectrogram 1\t{x[3]}\t{x[0]}\t{x[1]}\t0\t{sr/2}\t{x[2]}\n')

    return

# example Raven output. First column is a sequence. Channels start with 1
# Selection	View	Channel	Begin Time (s)	End Time (s)	Low Freq (Hz)	High Freq (Hz)	Annotation
# 1	Spectrogram 1	1	9.470096799	9.786469743	1758.418	9908.544	one


if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter
