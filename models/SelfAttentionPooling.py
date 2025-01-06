from copy import copy
from math import sqrt
from typing import Callable

import torch
from torch import nn, Tensor
import torch.nn.functional as F


class SelfAttentionPooling(nn.Module):
    """
    Implementation of SelfAttentionPooling
    Original Paper: Self-Attention Encoding and Pooling for Speaker Recognition
    https://arxiv.org/pdf/2008.01077v1.pdf
    """

    def __init__(self, input_dim):
        self.feat_in = input_dim
        super(SelfAttentionPooling, self).__init__()
        self.W = nn.Linear(input_dim, 1)

    def forward(self, batch_rep, length=None):
        """
        input:
            batch_rep : size (N, H, T), N: batch size, T: sequence length, H: Hidden dimension

        attention_weight:
            att_w : size (N, T, 1)

        return:
            utter_rep: size (N, H)
        """
        softmax = nn.functional.softmax
        batch_rep = batch_rep.permute(0, 2, 1)
        att_w = softmax(self.W(batch_rep).squeeze(-1),dim=1).unsqueeze(-1)
        utter_rep = torch.sum(batch_rep * att_w, dim=1)

        return utter_rep


def new_parameter(*size):
    out = torch.nn.Parameter(torch.FloatTensor(*size))
    torch.nn.init.xavier_normal_(out)
    return out


class Attention(nn.Module):

    def __init__(self, embedding_size):
        super(Attention, self).__init__()
        self.embedding_size = embedding_size
        self.att = new_parameter(self.embedding_size, 1)

    def forward(self, ht):
        attention_score = torch.matmul(ht, self.att).squeeze()
        attention_score = F.softmax(attention_score, dim=-1).view(ht.size(0), ht.size(1), 1)
        ct = torch.sum(ht * attention_score, dim=1)

        return ct, attention_score


class HeadAttention(nn.Module):

    def __init__(self, feat_in, heads_number, mask_prob=0.25, attentionSmoothing=False):

        super(HeadAttention, self).__init__()
        self.embedding_size = feat_in // heads_number
        self.att = new_parameter(self.embedding_size, 1)
        self.mask_prob = int(1 / mask_prob)
        self.attentionSmoothing = attentionSmoothing

    def __maskAttention(self, attention_score, mask_value=-float('inf')):

        mask = torch.cuda.FloatTensor(attention_score.size()).random_(self.mask_prob) > 0
        attention_score[~mask] = mask_value
        return attention_score

    def __narrowAttention(self, new_ht):

        attention_score = torch.matmul(new_ht, self.att).squeeze()
        if self.training:
            attention_score = self.__maskAttention(attention_score)
        attention_score = F.softmax(attention_score, dim=-1).view(new_ht.size(0), new_ht.size(1), 1)
        return attention_score

    def __wideAttention(self, new_ht):

        attention_score = torch.matmul(new_ht, self.att).squeeze()
        if self.training:
            attention_score = self.__maskAttention(attention_score, mask_value=-1)
        attention_score /= torch.sum(attention_score, dim=1).unsqueeze(1)
        return attention_score.view(new_ht.size(0), new_ht.size(1), 1)

    def forward(self, ht):

        if self.attentionSmoothing:
            attention_score = self.__wideAttention(ht)
        else:
            attention_score = self.__narrowAttention(ht)

        weighted_ht = ht * attention_score
        ct = torch.sum(weighted_ht, dim=1)

        return ct, attention_score


def innerKeyValueAttention(query, key, value):
    d_k = query.size(-1)
    scores = torch.diagonal(torch.matmul(key, query) / sqrt(d_k), dim1=-2, dim2=-1).view(value.size(0), value.size(1), value.size(2))
    p_attn = F.softmax(scores, dim=-2)
    weighted_vector = value * p_attn.unsqueeze(-1)
    ct = torch.sum(weighted_vector, dim=1)
    return ct, p_attn


class MultiHeadAttention(nn.Module):
    def __init__(self, feat_in, heads_number):
        super(MultiHeadAttention, self).__init__()
        self.feat_in = feat_in
        assert self.feat_in % heads_number == 0  # d_model
        self.head_size = self.feat_in // heads_number
        self.heads_number = heads_number
        self.query = new_parameter(self.head_size, self.heads_number)
        self.aligmment = None

    def getAlignments(self, ht):
        batch_size = ht.size(0)
        key = ht.view(batch_size * ht.size(1), self.heads_number, self.head_size)
        value = ht.view(batch_size, -1, self.heads_number, self.head_size)
        headsContextVectors, self.alignment = innerKeyValueAttention(self.query, key, value)
        return self.alignment

    def getHeadsContextVectors(self, ht):
        batch_size = ht.size(0)
        key = ht.reshape(batch_size * ht.size(1), self.heads_number, self.head_size)
        value = ht.reshape(batch_size, -1, self.heads_number, self.head_size)
        headsContextVectors, self.alignment = innerKeyValueAttention(self.query, key, value)
        return headsContextVectors

    def forward(self, ht):
        headsContextVectors = self.getHeadsContextVectors(ht)
        return headsContextVectors.view(headsContextVectors.size(0), -1), copy(self.alignment)


class DoubleMHA(nn.Module):
    def __init__(self, feat_in, heads_number, mask_prob=0.2):
        super(DoubleMHA, self).__init__()
        self.heads_number = heads_number
        self.utteranceAttention = MultiHeadAttention(feat_in, heads_number)
        self.heads_size = feat_in // heads_number
        self.headsAttention = HeadAttention(feat_in, heads_number, mask_prob=mask_prob, attentionSmoothing=False)
        self.feat_in = self.heads_size

    def getAlignments(self, x):
        utteranceRepresentation, alignment = self.utteranceAttention(x)
        headAlignments = self.headsAttention(utteranceRepresentation.view(utteranceRepresentation.size(0), self.heads_number, self.heads_size))[1]
        return alignment, headAlignments

    def forward(self, x, length=None):
        x = x.permute([0, 2, 1])
        utteranceRepresentation, alignment = self.utteranceAttention(x)
        compressedRepresentation = self.headsAttention(utteranceRepresentation.view(utteranceRepresentation.size(0), self.heads_number, self.heads_size))[0]
        return compressedRepresentation  # , alignment


class AttentionPool2d(nn.Module):
    "Attention for Learned Aggregation"
    def __init__(self,
        ni:int,
        bias:bool=True,
        norm:Callable[[int], nn.Module]=nn.LayerNorm
    ):
        super().__init__()
        self.norm = norm(ni)
        self.q = nn.Linear(ni, ni, bias=bias)
        self.vk = nn.Linear(ni, ni*2, bias=bias)
        self.proj = nn.Linear(ni, ni)

    def forward(self, x:Tensor, cls_q:Tensor):
        x = self.norm(x.flatten(2).transpose(1,2))
        B, N, C = x.shape

        q = self.q(cls_q.expand(B, -1, -1))
        k, v = self.vk(x).reshape(B, N, 2, C).permute(2, 0, 1, 3).chunk(2, 0)

        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, C)
        return self.proj(x)


class AvgAttnPooling2dS(nn.Module):
    def __init__(self,
                 feat_in:int,
                 attn_bias:bool=True,
                 ffn_expand=3,
                 norm:Callable[[int], nn.Module]=nn.LayerNorm,
                 act_cls:Callable[[None], nn.Module]=nn.GELU,
                 ):
        super().__init__()
        self.feat_in = feat_in
        self.cls_q = nn.Parameter(torch.zeros([1, feat_in]))
        self.attn = AttentionPool2d(feat_in, attn_bias, norm)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.norm = norm(feat_in)
        self.act = act_cls()
        nn.init.trunc_normal_(self.cls_q, std=0.02)
        self.apply(self._init_weights)

    def forward(self, x:Tensor, length:Tensor=None):
        return self.act(self.norm(self.pool(x).flatten(1) + self.attn(x, self.cls_q)))

    @torch.no_grad()
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


