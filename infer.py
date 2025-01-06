import librosa
import numpy as np
import torch
from nemo.collections.asr.parts.preprocessing.features import normalize_batch
from models.pcen_encode import pcen, calc_pcen


def get_overlapping_probs(x, channel: int, seg_models, sr: float, step_size: float, batch_size: int, runon: str):

    all_probs = np.empty((0, len(seg_models[0].cfg.labels)))
    for model in seg_models:
        mprobs = np.empty((0, len(seg_models[0].cfg.labels)))
        if 'pcen' in model.cfg.db_file:
            hop_size_s = model.cfg.preprocessor.window_stride
            pcen_s = calc_pcen(x[channel, :], hop_size=hop_size_s, win_size=hop_size_s*2, n_mels=model.cfg.preprocessor.features, sr=sr, min_freq=70, max_freq=6000, time_constant=0.1, alpha=0.98, delta=2, r=0.25)
            win_len = int(model.cfg.window_length_in_sec / hop_size_s)
            hop_size = int(step_size / hop_size_s)

            frames = librosa.util.frame(pcen_s, frame_length=win_len, hop_length=hop_size, axis=0)   # time, win_len, n_mel
            for r in range(0, frames.shape[0], batch_size):
                test_batch = torch.tensor(frames[r:min(r + batch_size, frames.shape[0]), :, :],dtype=torch.float).to(runon)
                test_batch = torch.reshape(test_batch,[test_batch.shape[0], test_batch.shape[1]*test_batch.shape[2]])
                lens = torch.tensor([win_len] * test_batch.shape[0],dtype=torch.int).to(runon)
                log_probs = model.infer(input_signal=test_batch, input_signal_length=lens)
                if type(log_probs) is tuple:
                    log_probs = log_probs[0]
                probs = torch.softmax(log_probs, dim=-1).cpu().detach().numpy()
                mprobs = np.concatenate((mprobs, probs))
        elif False and hasattr(model,'pcen') and model.pcen.enabled:  # make pcen on whole signal instead of individual frames
            w = x[channel, :]
            if model.cfg.train_ds.normalize_audio:  # normalize_audio to 1 if normalized during training
                w = w - np.mean(w)
                w = w / (np.max(np.abs(w))+1e-10)
            feats, feat_len = model.preprocessor(input_signal=torch.tensor(w)[None,].to(runon), length=torch.tensor(len(w))[None,].to(runon))
            feats, feat_len = model.pcen(feats, feat_len)
            if feats.shape[2] < model.crop_or_pad.audio_length:
                feats = torch.nn.functional.pad(feats,(1, model.crop_or_pad.audio_length - feats.shape[2]),mode='replicate')
            stride = int(model.cfg.test_step_size / model.cfg.preprocessor.window_stride)
            feats = feats[0,].unfold(1, model.crop_or_pad.audio_length, stride).permute(1, 0, 2)
            for r in range(0, feats.shape[0], batch_size):
                test_batch = feats[r:min(r + batch_size, feats.shape[0]), :, :]
                lens = torch.tensor([test_batch.shape[2]] * test_batch.shape[0]).to(runon)
                if model.pcen.normalize:
                    test_batch, _, _ = normalize_batch(test_batch, lens, normalize_type=model.pcen.normalize)

                logits, _ = model.infer(processed_signal=test_batch, processed_signal_length=lens)

                probs = torch.softmax(logits, dim=-1).cpu().detach().numpy()
                mprobs = np.concatenate((mprobs, probs))

        else:
            win_len = int(model.cfg.one_clip_length_in_sec * sr)
            hop_size = int(step_size * sr)
            if x.ndim > 1:
                t = np.array(x[channel, :])
            else:
                t = x
            if t.shape[0] < win_len:
                t = np.concatenate((t, np.zeros(win_len - t.shape[0])))
            frames = librosa.util.frame(t, frame_length=win_len, hop_length=hop_size)
            frames = np.transpose(frames)
            if model.cfg.train_ds.normalize_audio:  # normalize_audio to 1 if normalized during training
                frames = frames - np.mean(frames, axis=1)[:, None]
                frames = frames / (np.max(np.abs(frames), axis=1)[:, None]+1e-10)

            for r in range(0, frames.shape[0], batch_size):
                test_batch = torch.tensor(frames[r:min(r + batch_size, frames.shape[0]), :]).to(runon)
                lens = torch.tensor([win_len] * test_batch.shape[0]).to(runon)
                log_probs = model.infer(input_signal=test_batch, input_signal_length=lens)
                if type(log_probs) is tuple:
                    log_probs = log_probs[0]
                probs = torch.softmax(log_probs, dim=-1).cpu().detach().numpy()
                mprobs = np.concatenate((mprobs, probs))
        if all_probs.shape[0] == 0:
            all_probs = mprobs
        else:
            all_probs += mprobs

    all_probs = all_probs / len(seg_models)
    return all_probs

