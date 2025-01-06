from typing import Optional, Dict
from collections import OrderedDict
from nemo.core import typecheck
from nemo.core.neural_types import NeuralType, LogitsType, LengthsType
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from transformers import AutoModelForAudioClassification

from models import EncoderDecoderClassificationModel, SpeakerDecoder, AudioSpectrogramTransformerMLPDecoder


class AudioSpectrogramTransformerModel(EncoderDecoderClassificationModel):
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
        if processed_signal is None:
            processed_signal, processed_signal_length, processed_offsets = self.preprocess_and_augment(input_signal, input_signal_length)

        processed_signal = processed_signal.permute(0, 2, 1)
        outputs = self.encoder(audio_signal=processed_signal, length=processed_signal_length)
        # ignored: head_mask = head_mask, output_attentions = output_attentions, output_hidden_states = output_hidden_states, return_dict = return_dict,

        encoded = outputs.pooler_output  # outputs[1]

        if type(self.decoder) is AudioSpectrogramTransformerMLPDecoder:
            logits = self.decoder(encoder_output=encoded)
        elif type(self.decoder) is SpeakerDecoder:
            logits, _ = self.decoder(encoder_output=encoded)
        else:
            logits = self.decoder(encoder_output=encoded)

        # loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        return logits, processed_offsets


def from_huggingface(model_name, cfg, trainer=None):
    model = AutoModelForAudioClassification.from_pretrained(model_name)
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
