import argparse
import random
import librosa
import matplotlib.pyplot as plt
import numpy as np
import librosa.display
import h5py
import os
import pandas
from pathlib import Path
from outliers import smirnov_grubbs as grubbs


def pcen(db_root, filename, hop_size=0.0125, win_size=0.0250, n_mels=64, min_freq=70, max_freq=6000, sr=16000, time_constant=0.1, alpha=0.98, delta=2, r=0.25):
    y, sr = librosa.load(os.path.join(db_root, filename), sr=sr, mono=False)
    fp = Path(filename)
    file = db_root / fp.with_suffix(f'.{n_mels}.hdf5')
    if not file.exists():
      print(filename)
      with h5py.File(str(file), "w") as f:
        for c in range(y.shape[0]):

            yc = y[c, :]
            pcen_s = calc_pcen(yc, hop_size, win_size, n_mels, min_freq, max_freq, sr, time_constant, alpha, delta, r)

            f.create_dataset(f"pcen_{c}", data=pcen_s)
        f.close()

    return str(file.relative_to(db_root))


def calc_pcen(yc, hop_size=0.0125, win_size=0.0250, n_mels=64, min_freq=70, max_freq=6000, sr=16000, time_constant=0.1, alpha=0.98, delta=2, r=0.25):
    yc = yc / max(np.abs(yc))
    outlier_size = int(30 / hop_size)  # half minute step, one minute size
    hop_size = int(sr*hop_size)
    win_size = int(sr*win_size)
    fft_size = int(2**(np.ceil(np.log2(win_size))))

    # remove large peaks (noise)
    mw = np.mean(yc)
    wn = np.where(np.abs(np.diff(yc)) > 0.1)[0]
    while len(wn) > 0:
        yc[wn] = mw
        yc[wn + 1] = mw
        wn = np.where(np.abs(np.diff(yc)) > 0.1)[0]

    M = librosa.feature.melspectrogram(y=yc * (2 ** 31), sr=sr, power=1, hop_length=hop_size, win_length=win_size, n_fft=fft_size, n_mels=n_mels, fmin=min_freq, fmax=max_freq, htk=True)

    Mlog = np.log(M+1e-5)

    # remove outlier peaks (noise)
    MDN = M
    for m in range(MDN.shape[0]):
        mn = np.median(MDN[m, :])
        i = 0
        while i < MDN.shape[1] - outlier_size - 1:
            ix = np.array(grubbs.max_test_indices(Mlog[m, i: min(MDN.shape[1], i + 2 * outlier_size)], alpha=.05))
            if len(ix) > 0:
                MDN[m, i + ix] = mn
            i = i + outlier_size

    pcen_s = librosa.pcen(MDN, sr=sr, hop_length=hop_size, gain=alpha, bias=delta, power=r, time_constant=time_constant)
    pcen_s = np.transpose(pcen_s)
    return pcen_s

