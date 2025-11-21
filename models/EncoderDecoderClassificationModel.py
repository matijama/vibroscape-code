import logging
import nemo
import scipy
import torch
from typing import Optional, Dict
from nemo.collections.asr.data import audio_to_label_dataset
from nemo.collections.asr.parts.preprocessing import process_augmentations, WaveformFeaturizer
from nemo.collections.nlp.metrics import ClassificationReport
from nemo.core.classes.common import typecheck
from nemo.core.neural_types import NeuralType, LogitsType, LengthsType
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from sklearn import metrics

from models.Losses import CrossEntropyLoss
from models.Dataset import AudioLabelsDataset, AudioStreamLabelsDataset
from models.SpeakerDecoder import SpeakerDecoder
from nemo.collections.asr.models.asr_model import ASRModel


class EncoderDecoderClassificationModel(nemo.collections.asr.models.classification_models.EncDecClassificationModel):
    """Encoder decoder Classification models."""

    def __init__(self, cfg: DictConfig, trainer: Trainer = None, loss_weights = None):

        self.classification_report: ClassificationReport = None
        self.validation_logits = []
        self.validation_labels = []
        self.loss_weights = loss_weights

        super().__init__(cfg=cfg, trainer=trainer)

        if hasattr(self._cfg, 'pcen') and self._cfg.pcen is not None:
            self.pcen = ASRModel.from_config_dict(self._cfg.pcen)
        else:
            self.pcen = None

        if hasattr(self._cfg, 'audio_augment') and self._cfg.audio_augment is not None:
            self.audio_augment = ASRModel.from_config_dict(self._cfg.audio_augment)
        else:
            self.audio_augment = None

        if hasattr(self._cfg, 'spec_stretch') and self._cfg.spec_stretch is not None:
            self.spec_stretch = ASRModel.from_config_dict(self._cfg.spec_stretch)
        else:
            self.spec_stretch = None

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        return {"outputs": NeuralType(('B', 'D'), LogitsType()),
                "offsets": NeuralType(tuple('B'), LengthsType()),
                }

    def preprocess_and_augment(self, input_signal, input_signal_length):

        sample_rate = self.preprocessor._sample_rate
        # isig_cpu = input_signal.cpu().numpy()
        # isig_cpu = remove_outliers(isig_cpu, input_signal_length, sample_rate, 0.01)

        if self.audio_augment is not None and self.audio_augment.enabled and self.training:
            input_signal, input_signal_length = self.audio_augment(input_signal=input_signal, length=input_signal_length, sample_rate=sample_rate)

        # input_signal = torch.from_numpy(isig_cpu).to(input_signal.device)

        processed_signal, processed_signal_len = self.preprocessor(
            input_signal=input_signal, length=input_signal_length,
        )

        if self.pcen is not None and self.pcen.enabled:
            processed_signal, processed_signal_len = self.pcen(processed_signal, processed_signal_len)

        if self.spec_stretch is not None and self.spec_stretch.enabled and self.training:
            processed_signal, processed_signal_len = self.spec_stretch(input_spectrogram=processed_signal, length=processed_signal_len)

        processed_offsets = None
        # Crop or pad is always applied
        if self.crop_or_pad is not None:
            processed_signal, processed_signal_len, processed_offsets = self.crop_or_pad(input_signal=processed_signal, length=processed_signal_len)

        # Spec augment is not applied during evaluation/testing
        if self.spec_augmentation is not None and self.training:
            processed_signal = self.spec_augmentation(input_spec=processed_signal, length=processed_signal_len)
        return processed_signal, processed_signal_len, processed_offsets

    @typecheck()
    def forward(self, input_signal=None, input_signal_length=None, processed_signal=None, processed_signal_length=None):

        processed_offsets = None
        if processed_signal is None:
            input_signal = input_signal.type(self.dtype)
            processed_signal, processed_signal_length, processed_offsets = self.preprocess_and_augment(input_signal, input_signal_length)

        encoded, encoded_len = self.encoder(audio_signal=processed_signal, length=processed_signal_length)
        if type(self.decoder) is SpeakerDecoder:
            logits, _ = self.decoder(encoder_output=encoded)
        else:
            logits = self.decoder(encoder_output=encoded)
        return logits, processed_offsets


    def infer(self, input_signal=None, input_signal_length=None, processed_signal=None, processed_signal_length=None):

        return self.forward(input_signal=input_signal, input_signal_length=input_signal_length, processed_signal=processed_signal, processed_signal_length=processed_signal_length)


    # PTL-specific methods
    def training_step(self, batch, batch_nb):
        #self.training_step_end()
        if 'vad_stream' in self._cfg.train_ds and self._cfg.train_ds.vad_stream is True:
            audio_signal, audio_signal_len, labels, _, ys = batch
        else:
            audio_signal, audio_signal_len, labels, _, labelss, labelss_len, ys = batch

        logits, offsets = self.forward(input_signal=audio_signal, input_signal_length=audio_signal_len)
        loss_value = self.loss(logits=logits, labels=labels, ys=ys)

        self.log('train_loss', loss_value)
        self.log('learning_rate', self._optimizer.param_groups[0]['lr'])
        self.log('global_step', self.trainer.global_step)

        self._accuracy(logits=logits, labels=labels)
        topk_scores = self._accuracy.compute()
        self._accuracy.reset()

        for top_k, score in zip(self._accuracy.top_k, topk_scores):
            self.log('training_batch_accuracy_top_{}'.format(top_k), score)

        if self.classification_report:
            preds = torch.argmax(logits, dim=1)
            self.classification_report(predictions=preds, labels=labels)
            scores = self.classification_report.compute()
            self.classification_report.reset()

            for i in range(len(scores[0])):
                self.log(f'training_batch_precision_{self._cfg.labels[i]}', scores[0][i])
                self.log(f'training_batch_recall_{self._cfg.labels[i]}', scores[1][i])
                self.log(f'training_batch_f1_{self._cfg.labels[i]}', scores[2][i])

        return {
            'loss': loss_value,
        }

    # def _setup_metrics(self):
    #     self._accuracy = TopKClassificationAccuracy(dist_sync_on_step=True)

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        if 'vad_stream' in self._cfg.validation_ds and self._cfg.validation_ds.vad_stream is True:
            audio_signal, audio_signal_len, labels, _, ys = batch
        else:
            audio_signal, audio_signal_len, labels, _, labelss, labelss_len, ys = batch

        logits, offsets = self.forward(input_signal=audio_signal, input_signal_length=audio_signal_len)
        loss_value = self.loss(logits=logits, labels=labels, ys=ys)
        acc = self._accuracy(logits=logits, labels=labels)
        correct_counts, total_counts = self._accuracy.correct_counts_k, self._accuracy.total_counts_k

        if self.classification_report:
            self.validation_logits += logits.cpu().detach().numpy().tolist()
            self.validation_labels += labels.cpu().detach().numpy().tolist()

            preds = torch.argmax(logits, dim=1)
            self.classification_report(predictions=preds, labels=labels)

        loss = {
            'val_loss': loss_value,
            'val_correct_counts': correct_counts,
            'val_total_counts': total_counts,
            'val_acc': acc,
        }

        if type(self.trainer.val_dataloaders) == list and len(self.trainer.val_dataloaders) > 1:
            self.validation_step_outputs[dataloader_idx].append(loss)
        else:
            self.validation_step_outputs.append(loss)

        return loss


    def multi_validation_epoch_end(self, outputs, dataloader_idx: int = 0):
        tensorboard_log = super().multi_validation_epoch_end(outputs, dataloader_idx)['log']

        if self.classification_report:
            scores = self.classification_report.compute()
            z = list(self._cfg.labels)
            if len(z) == 2:
                z[0] = 'fg'
            for i in range(len(scores[0])):
                tensorboard_log[f'val_precision_{z[i]}'] = scores[0][i]  # _{self._cfg.labels[i]}
                tensorboard_log[f'val_recall_{z[i]}'] = scores[1][i]
                tensorboard_log[f'val_f1_{z[i]}'] = scores[2][i]

            if self.validation_logits:
                probs = scipy.special.softmax(self.validation_logits, axis=1)
                tensorboard_log['val_map'] = metrics.average_precision_score(self.validation_labels, probs[:,0], pos_label = 0)
                self.validation_logits = []
                self.validation_labels = []

            # for k, v in tensorboard_log.items():
            #     print(k, v)
            self.classification_report.reset()
            #self.log('training_batch_summary', scores[3])

        return {'log': tensorboard_log}


    def test_step(self, batch, batch_idx, dataloader_idx=0):
        if 'vad_stream' in self._cfg.test_ds and self._cfg.test_ds.vad_stream is True:
            audio_signal, audio_signal_len, labels, _, ys = batch
        else:
            audio_signal, audio_signal_len, labels, _, labelss, labelss_len, ys = batch
        logits, offsets = self.forward(input_signal=audio_signal, input_signal_length=audio_signal_len)
        loss_value = self.loss(logits=logits, labels=labels, ys=ys)
        acc = self._accuracy(logits=logits, labels=labels)
        correct_counts, total_counts = self._accuracy.correct_counts_k, self._accuracy.total_counts_k
        loss =  {
            'test_loss': loss_value,
            'test_correct_counts': correct_counts,
            'test_total_counts': total_counts,
            'test_acc': acc,
        }
        if type(self.trainer.test_dataloaders) == list and len(self.trainer.test_dataloaders) > 1:
            self.test_step_outputs[dataloader_idx].append(loss)
        else:
            self.test_step_outputs.append(loss)
        return loss

    def on_train_epoch_start(self):
        self._train_dl.dataset.new_subset()

    def on_validation_epoch_start(self):
        self._validation_dl.dataset.new_subset()

    def _setup_dataloader_from_config(self, config: DictConfig):

        

        if 'augmentor' in config:
            augmentor = process_augmentations(config['augmentor'])
        else:
            augmentor = None

        featurizer =  WaveformFeaturizer(
            sample_rate=config['sample_rate'], int_values=config.get('int_values', False), augmentor=augmentor
        )
        shuffle = config['shuffle']

        if 'manifest_filepath' in config and config['manifest_filepath'] is None:
            logging.warning(f"Could not load dataset as `manifest_filepath` is None. Provided config : {config}")
            return None

        cnf = OmegaConf.to_container(config)
        if 'vad_stream' in config and config['vad_stream']:
            logging.info("Perform streaming frame-level classification")
            dataset = AudioStreamLabelsDataset(
                manifest_filepath=cnf['manifest_filepath'],
                labels=cnf['labels'],
                featurizer=featurizer,
                max_duration=cnf.get('max_duration', None),
                min_duration=cnf.get('min_duration', None),
                trim=cnf.get('trim_silence', False),
                normalize_audio=cnf.get('normalize_audio', False),
                window_length_in_sec=cnf.get('window_length_in_sec', 0.31),
                shift_length_in_sec=cnf.get('shift_length_in_sec', 0.01),
                nwin_clip_length_in_sec=cnf.get('nwin_clip_length_in_sec', None),
                one_clip_length_in_sec=cnf.get('one_clip_length_in_sec', None),
                subsample_each_epoch=cnf.get('subsample_each_epoch', 0),
            )
            if cnf.get('subsample_each_epoch', 0) > 0:
                dataset.new_subset()

            batch_size = config['batch_size'] // cnf.get('n_windows', 1)
            collate_func = dataset.vad_frame_seq_collate_fn
        else:

            dataset = AudioLabelsDataset(
                manifest_filepath=cnf['manifest_filepath'],
                labels=cnf['labels'],
                featurizer=featurizer,
                max_duration=cnf.get('max_duration', None),
                min_duration=cnf.get('min_duration', None),
                trim=cnf.get('trim_silence', False),
                normalize_audio=cnf.get('normalize_audio', False),
                nwin_clip_length_in_sec=cnf.get('nwin_clip_length_in_sec', None),
                one_clip_length_in_sec=cnf.get('one_clip_length_in_sec', None),
                subsample_each_epoch=cnf.get('subsample_each_epoch', 0),
            )
            if cnf.get('subsample_each_epoch', 0) > 0:
                dataset.new_subset()

            batch_size = config['batch_size']
            collate_func = dataset._collate_fn 

        return self._setup_dataloader(config, dataset, batch_size, collate_func, shuffle)


    def _setup_dataloader(self, config: DictConfig, dataset, batch_size, collate_func, shuffle):

        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            collate_fn=collate_func,
            drop_last=config.get('drop_last', False),
            shuffle=shuffle,
            num_workers=config.get('num_workers', 0),
            pin_memory=config.get('pin_memory', False),
        )


    def _setup_loss(self):
        """
        Setup loss function for training
        Returns: Loss function
        """
        return CrossEntropyLoss(2, weight=self.loss_weights)


    def pre_save_hook(self):
        pass
