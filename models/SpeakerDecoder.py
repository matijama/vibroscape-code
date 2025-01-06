from collections import OrderedDict
import torch
from typing import Optional, Union
from nemo.collections.asr.parts.submodules.jasper import init_weights
from nemo.collections.asr.parts.submodules.tdnn_attention import StatsPoolLayer, AttentivePoolLayer
from nemo.core import NeuralModule, Exportable
from nemo.core.classes.common import typecheck
from nemo.core.neural_types import NeuralType, LogitsType, LengthsType, AcousticEncodedRepresentation
from torch import nn
import torch.nn.functional as F

from models.SelfAttentionPooling import SelfAttentionPooling, DoubleMHA, AvgAttnPooling2dS


class SpeakerDecoder(NeuralModule, Exportable):
    """
    Speaker Decoder creates the final neural layers that maps from the outputs
    of Jasper Encoder to the embedding layer followed by speaker based softmax loss.
    Args:
        feat_in (int): Number of channels being input to this module
        num_classes (int): Number of unique speakers in dataset
        emb_sizes (list) : shapes of intermediate embedding layers (we consider speaker embbeddings from 1st of this layers)
                Defaults to [128]
        pool_mode (str) : Pooling stratergy type. options are 'xvector','tap', 'attention', 'selfattentionpool'
                Defaults to 'xvector (mean and variance)'
                tap (temporal average pooling: just mean)
                attention (attention based pooling)

        init_mode (str): Describes how neural network parameters are
            initialized. Options are ['xavier_uniform', 'xavier_normal',
            'kaiming_uniform','kaiming_normal'].
            Defaults to "xavier_uniform".
    """

    def input_example(self, max_batch=1, max_dim=256):
        """
        Generates input examples for tracing etc.
        Returns:
            A tuple of input examples.
        """
        input_example = torch.randn(max_batch, self.input_feat_in, max_dim).to(next(self.parameters()).device)
        return tuple([input_example])

    @property
    def input_types(self):
        return OrderedDict(
            {
                "encoder_output": NeuralType(('B', 'D', 'T'), AcousticEncodedRepresentation()),
                "length": NeuralType(('B',), LengthsType(), optional=True),
            }
        )

    @property
    def output_types(self):
        return OrderedDict(
            {
                "logits": NeuralType(('B', 'D'), LogitsType()),
                "embs": NeuralType(('B', 'D'), AcousticEncodedRepresentation()),
            }
        )

    def __init__(
        self,
        feat_in: int,
        num_classes: int,
        emb_sizes: Optional[Union[int, list]] = 128,
        pool_mode: str = 'xvector',
        angular: bool = False,
        attention_channels: int = 128,
        init_mode: str = "xavier_uniform",
    ):
        super().__init__()
        self._pooling = None
        self.angular = angular
        self.emb_id = 2
        self.num_classes = num_classes
        bias = False if self.angular else True
        emb_sizes = [emb_sizes] if type(emb_sizes) is int else emb_sizes
        print(self)
        print(feat_in)
        print(num_classes)
        print(pool_mode)

        self._num_classes = num_classes
        self.pool_mode = pool_mode.lower()
        if self.pool_mode == 'xvector' or self.pool_mode == 'tap':
            self._pooling = StatsPoolLayer(feat_in=feat_in, pool_mode=self.pool_mode)
            affine_type = 'linear'
        elif self.pool_mode == 'attention':
            self._pooling = AttentivePoolLayer(inp_filters=feat_in, attention_channels=attention_channels)
            affine_type = 'conv'
        elif self.pool_mode =='selfattentionpool':
            self._pooling = SelfAttentionPooling(input_dim=feat_in)
            affine_type = 'linear'
        elif self.pool_mode =='doublemha':
            self._pooling = DoubleMHA(feat_in=feat_in, heads_number=4)
            affine_type = 'linear'
        elif self.pool_mode =='avgattnpool2d':
            self._pooling = AvgAttnPooling2dS(feat_in=feat_in)
            affine_type = 'linear'

        shapes = [self._pooling.feat_in]
        for size in emb_sizes:
            shapes.append(int(size))

        emb_layers = []
        for shape_in, shape_out in zip(shapes[:-1], shapes[1:]):
            layer = self.affine_layer(shape_in, shape_out, learn_mean=False, affine_type=affine_type)
            emb_layers.append(layer)

        self.emb_layers = nn.ModuleList(emb_layers)

        self.final = nn.Linear(shapes[-1], self._num_classes, bias=bias)

        self.apply(lambda x: init_weights(x, mode=init_mode))

    def affine_layer(
        self, inp_shape, out_shape, learn_mean=True, affine_type='conv',
    ):
        if affine_type == 'conv':
            layer = nn.Sequential(
                nn.BatchNorm1d(inp_shape, affine=True, track_running_stats=True),
                nn.Conv1d(inp_shape, out_shape, kernel_size=1),
            )

        else:
            layer = nn.Sequential(
                nn.Linear(inp_shape, out_shape),
                nn.BatchNorm1d(out_shape, affine=learn_mean, track_running_stats=True),
                nn.ReLU(),
            )

        return layer

    @typecheck()
    def forward(self, encoder_output, length=None):
        pool = self._pooling(encoder_output, length)
        embs = []

        for layer in self.emb_layers:
            pool, emb = layer(pool), layer[: self.emb_id](pool)
            embs.append(emb)

        pool = pool.squeeze(-1)
        if self.angular:
            for W in self.final.parameters():
                W = F.normalize(W, p=2, dim=1)
            pool = F.normalize(pool, p=2, dim=1)

        out = self.final(pool)

        return out, embs[-1].squeeze(-1)

