import logging
from typing import Optional, Dict
from collections import OrderedDict
import torch
from lightning_utilities.core.rank_zero import rank_zero_only
from nemo.core import typecheck
from nemo.core.neural_types import NeuralType, LogitsType, LengthsType
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer

from models import EncoderDecoderClassificationModel, SpeakerDecoder, LinearSigmoidDecoder
from models.AudioSpectrogramTransformerModel import AudioSpectrogramTransformerModel


class PANNsModel(EncoderDecoderClassificationModel):
    """Encoder decoder Classification models."""

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):

        super().__init__(cfg=cfg, trainer=trainer)

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        return {"outputs": NeuralType(('B', 'D'), LogitsType()),
                "offsets": NeuralType(tuple('B'), LengthsType()),
                }

    @typecheck()
    def forward(self, input_signal=None, input_signal_length=None, processed_signal=None, processed_signal_length=None):

        processed_offsets = None
        if processed_signal_length is None:
            input_signal = input_signal.type(self.encoder._model.fc1.weight.dtype)  # cast to prevent fp issues
            processed_signal, processed_signal_length, processed_offsets = self.preprocess_and_augment(input_signal, input_signal_length)

        processed_signal = processed_signal.permute(0, 2, 1)
        output, length = self.encoder(audio_signal=processed_signal, length=processed_signal_length)
        # ignored: head_mask = head_mask, output_attentions = output_attentions, output_hidden_states = output_hidden_states, return_dict = return_dict,

        if type(self.decoder) is SpeakerDecoder:
            logits, _ = self.decoder(encoder_output=output)
        elif type(self.decoder) is LinearSigmoidDecoder:
            logits, logits_per_frame = self.decoder(encoder_output=output, length=processed_signal.shape[1])
        else:
            logits = self.decoder(encoder_output=output)

        # loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        return logits, processed_offsets

    @rank_zero_only
    def maybe_init_from_pretrained_checkpoint(self, cfg: OmegaConf, map_location: str = 'cpu'):
        """
                hydra usage example:

                init_from_nemo_model:
                    model0:
                        path:<path/to/model1>
                        include:["encoder"]
                    model1:
                        path:<path/to/model2>
                        include:["decoder"]
                        exclude:["embed"]

            init_from_panns_ckpt: inits model from PANNs checkpoint
            init_from_nemo_model: inits model from another NeMo model checkpoint
            init_from_pretrained_model: Str name of a pretrained model checkpoint (obtained via cloud).
            init_from_ptl_ckpt: Str name of a Pytorch Lightning checkpoint file. It will be loaded and
        """

        if 'init_from_panns_ckpt' in cfg:
            if isinstance(cfg.init_from_panns_ckpt, str):
                model_name = cfg.init_from_panns_ckpt
                include = ['encoder', 'decoder']
                exclude = []
            elif isinstance(cfg.init_from_panns_ckpt, (DictConfig, dict)):
                model_load_dict = cfg.init_from_panns_ckpt
                for model_load_cfg in model_load_dict.values():
                    model_name = model_load_cfg.path
                    include = model_load_cfg.get('include', ['encoder', 'decoder'])
                    exclude = model_load_cfg.get('exclude', [])
                    break

            checkpoint = torch.load(model_name, map_location=map_location)

            state_dict = self.state_dict()

            for x in exclude:
                include.remove(x)

            new_state_dict = OrderedDict()
            for ed in include:
                if ed == 'encoder':
                    prefix = 'encoder._model.'
                else:
                    prefix = 'decoder.'
                for key in [x for x in state_dict.keys() if x.startswith(prefix)]:
                    new_key = key.replace(prefix, '')
                    new_state_dict[key] = checkpoint['model'][new_key]

            # Restore checkpoint into current model
            self.load_state_dict(new_state_dict, strict=False)

            logging.info(f'Model checkpoint restored from pretrained checkpoint with name : `{model_name}`')
            del checkpoint
        else:
            super().maybe_init_from_pretrained_checkpoint(cfg, map_location)


def from_pretrained_checkpoint(model_path, device, cfg, trainer=None):
    model = PANNsModel(cfg=cfg.model, trainer=trainer)

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    #self.base.load_state_dict(checkpoint['model'])

    for k in cfg.model.encoder:
        try:
            z = model.config.__getattribute__(k)
            cfg.model.encoder[k] = z
        except:
            continue
    cfg.model.labels = list(model.config.id2label.values())
    cfg.model.train_ds.manifest_filepath = "nemo_config/dummy.nemo"
    cfg.model.validation_ds.manifest_filepath = "nemo_config/dummy.nemo"
    if 'test_ds' in cfg.model:
        cfg.model.test_ds.manifest_filepath = "nemo_config/dummy.nemo"
    nemo_model = AudioSpectrogramTransformerModel(cfg.model, trainer)

    state_dict = model.state_dict()

    keys = [x for x in state_dict.keys() if not x.startswith("classifier")]
    to_replace = keys[0].split('.')[0]
    replacement = list(nemo_model.encoder.state_dict().keys())[0].split('.')[0]
    new_state_dict = OrderedDict()
    for key in keys:
        new_key = key.replace(to_replace, replacement)
        new_state_dict[new_key] = state_dict[key]
    nemo_model.encoder.load_state_dict(new_state_dict, strict=False)

    keys = [x for x in state_dict.keys() if x.startswith("classifier")]
    to_replace = keys[0].split('.')[0]
    replacement = list(nemo_model.decoder.state_dict().keys())[0].split('.')[0]
    new_state_dict = OrderedDict()
    for key in keys:
        new_key = key.replace(to_replace, replacement)
        new_state_dict[new_key] = state_dict[key]
    nemo_model.decoder.load_state_dict(new_state_dict, strict=False)

    return nemo_model
