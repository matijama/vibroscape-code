import torch
from torch import nn
import torch.nn.functional as F

from PANNs.pytorch.models import ConvBlock, init_bn, init_layer


class Cnn14(nn.Module):
    def __init__(self, feat_in: int, feat_out: int, keep_time: bool):

        super(Cnn14, self).__init__()

        self.keep_time = keep_time
        self.bn0 = nn.BatchNorm2d(feat_in)

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.conv_block5 = ConvBlock(in_channels=512, out_channels=1024)
        self.conv_block6 = ConvBlock(in_channels=1024, out_channels=2048)

        self.fc1 = nn.Linear(2048, feat_out, bias=True)

        self.init_weight()

    def init_weight(self):
        init_bn(self.bn0)
        init_layer(self.fc1)

    def forward(self, input, length=None):  # , mixup_lambda=None
        """
        Input: (batch_size, data_length)"""

        # expects (batch_size, 1, time_steps, mel_bins)
        x = torch.unsqueeze(input, 1)

        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        # if self.training:
        #     x = self.spec_augmenter(x)
        #
        # # Mixup on spectrogram
        # if self.training and mixup_lambda is not None:
        #     x = do_mixup(x, mixup_lambda)

        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block5(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block6(x, pool_size=(1, 1), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = torch.mean(x, dim=3)

        if self.keep_time:
            x1 = F.max_pool1d(x, kernel_size=3, stride=1, padding=1)
            x2 = F.avg_pool1d(x, kernel_size=3, stride=1, padding=1)
            x = x1 + x2
            x = F.dropout(x, p=0.5, training=self.training)
            x = x.transpose(1, 2)
            x = F.relu_(self.fc1(x))
            x = x.transpose(1, 2)
        else:
            (x1, _) = torch.max(x, dim=2)
            x2 = torch.mean(x, dim=2)
            x = x1 + x2
            x = F.dropout(x, p=0.5, training=self.training)
            x = F.relu_(self.fc1(x))
            # add time, beacuse we need B,D,T output
            x = torch.unsqueeze(x, 2)

        return x, length
