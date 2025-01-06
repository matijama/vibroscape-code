import json
from omegaconf import OmegaConf, ListConfig
import pytorch_lightning as pl
from nemo.core.config import hydra_runner
from nemo.utils.exp_manager import exp_manager
import os
import torch
import numpy as np



@hydra_runner(config_path="nemo_config", config_name="config")
def main(cfg):
    name = cfg.name

    #nemo.collections.asr.parts.preprocessing.features.normalize_batch = models.Normalization.normalize_batch

    OmegaConf.register_new_resolver("dividei", lambda x, y: int(x/y))
    torch.set_float32_matmul_precision('high')

    winsize = cfg.model.window_length_in_sec
    nwins = cfg.model.n_windows
    winstep = cfg.model.shift_length_in_sec

    if 'n_fft' in cfg.model.preprocessor:
        wl = cfg.model.preprocessor.n_fft/cfg.model.preprocessor.sample_rate
    else:
        wl = cfg.model.preprocessor.window_size

    # minimum clip length is at least one window
    cfg.model.one_clip_length_in_sec = winsize + max(wl, cfg.model.preprocessor.window_size)
    # clip length in one window + nwins * stepsize
    cfg.model.nwin_clip_length_in_sec = winsize + (nwins-1) * winstep + max(wl, cfg.model.preprocessor.window_size) + 0.02  # 0.02 added for augmentation time shifts

    # setup vad_streaming if needed
    cfg.model.train_ds.vad_stream = (nwins > 1)
    cfg.model.validation_ds.vad_stream = (nwins > 1)

    # we need this to crop oversized nwin_clip_length_in_sec clips to the correct size
    cfg.model.crop_or_pad_augment.audio_length = int(np.ceil(winsize / cfg.model.preprocessor.window_stride))

    version = cfg.model.get('dataset_version', '')
    cfg.model.train_ds.manifest_filepath = os.path.join(cfg.db_root, f"training_manifest{version}.json")
    if 'validation_ds' in cfg.model:
        cfg.model.validation_ds.manifest_filepath = os.path.join(cfg.db_root, f"validation_manifest{version}.json")
        with open(cfg.model.validation_ds.manifest_filepath, 'r', encoding='utf8') as f:
            ls = [json.loads(x)['label'] for x in f.readlines()]
            labels_val =  list(sorted(set(ls)))
    else:
        labels_val = []
    if 'test_ds' in cfg.model:
        cfg.model.test_ds.manifest_filepath = os.path.join(cfg.db_root, f"test_manifest{version}.json")
        with open(cfg.model.test_ds.manifest_filepath, 'r', encoding='utf8') as f:
            test_files = list(set([json.loads(x)['audio_filepath'] for x in f.readlines()]))
    with open(cfg.model.train_ds.manifest_filepath, 'r', encoding='utf8') as f:
        ls = [json.loads(x)['label'] for x in f.readlines()]
    labels = list(sorted(set(ls)))
    class_weights = np.ones(len(labels))/len(labels)

    trainer = pl.Trainer(**cfg.trainer)

    # change labels in config if necessary
    cfg.model.num_classes = len(labels)
    new_labels = ListConfig(labels)
    if cfg.model.labels != new_labels:
        cfg.model.labels = new_labels

        if 'params' in cfg.model.decoder:
            cfg.model.decoder.params.num_classes = len(labels)
        else:
            cfg.model.decoder.num_classes = len(labels)

        if 'train_ds' in cfg.model and cfg.model.train_ds is not None:
            cfg.model.train_ds.labels = new_labels
        if 'validation_ds' in cfg.model and cfg.model.validation_ds is not None:
            cfg.model.validation_ds.labels = ListConfig(labels_val)
        if 'test_ds' in cfg.model and cfg.model.test_ds is not None:
            cfg.model.test_ds.labels = new_labels

    log_dir = exp_manager(trainer, cfg.get("exp_manager", None))
    if cfg.model.encoder.get('_target_','xxx') == 'models.PANNsEncoder':
        from models.PANNsModel import PANNsModel
        model = PANNsModel(cfg=cfg.model, trainer=trainer)
    else:
        from models.EncoderDecoderClassificationModel import EncoderDecoderClassificationModel
        model = EncoderDecoderClassificationModel(cfg=cfg.model, trainer=trainer, loss_weights=class_weights)

    model.to('cuda')

    from nemo.collections.nlp.metrics import ClassificationReport
    model.classification_report = ClassificationReport(num_classes=len(labels), dist_sync_on_step=True, mode='all')

    if cfg.get('init_from_nemo_model',None) is not None:
        if cfg.init_from_nemo_model.get('model',None) is not None:
            if cfg.init_from_nemo_model.model.get('path',None) is not None:
                model.maybe_init_from_pretrained_checkpoint(cfg)
    if cfg.get('init_from_panns_ckpt', None) is not None:
        model.maybe_init_from_pretrained_checkpoint(cfg)

    with torch.autograd.set_detect_anomaly(True):
        trainer.fit(model)

    model.pre_save_hook()

    saved_model_path = os.path.join(log_dir, name + '_model.nemo')
    model.save_to(saved_model_path)
    model.eval()

    if 'test_ds' in cfg.model and cfg.model.test_ds.manifest_filepath is not None:
        trainer = pl.Trainer()
        if model.prepare_test(trainer):
            trainer.test(model)



if __name__ == '__main__':

    main()  # noqa pylint: disable=no-value-for-parameter
