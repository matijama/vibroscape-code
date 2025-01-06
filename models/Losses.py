import torch
import torch.nn as nn
import torch.nn.functional as F
from nemo.core import Serialization, Typing, typecheck
from nemo.core.neural_types import NeuralType, LogitsType, LabelsType, MaskType, ProbsType, LossType
from nemo.utils import logging
from torch.nn.modules import Module

class AdMSoftmaxLoss(nn.Module):

    def __init__(self, in_features, out_features, s=30.0, m=0.2):
        '''
        AM Softmax Loss
        '''
        super(AdMSoftmaxLoss, self).__init__()
        self.s = s
        self.m = m
        self.in_features = in_features
        self.out_features = out_features
        self.fc = nn.Linear(in_features, out_features, bias=False)

    def forward(self, logits, labels):
        '''
        input shape (N, in_features)
        '''
        assert len(logits) == len(labels)
        assert torch.min(labels) >= 0
        assert torch.max(labels) < self.out_features

        for W in self.fc.parameters():
            W = F.normalize(W, dim=1)

        x = F.normalize(logits, dim=1)

        wf = self.fc(x)
        numerator = self.s * (torch.diagonal(wf.transpose(0, 1)[labels]) - self.m)
        excl = torch.cat([torch.cat((wf[i, :y], wf[i, y + 1:])).unsqueeze(0) for i, y in enumerate(labels)], dim=0)
        denominator = torch.exp(numerator) + torch.sum(torch.exp(self.s * excl), dim=1)
        L = numerator - torch.log(denominator)
        return -torch.mean(L)



class CrossEntropyLoss(nn.CrossEntropyLoss, Serialization, Typing):
    """
    CrossEntropyLoss
    """

    @property
    def input_types(self):
        """Returns definitions of module input ports.
        """
        return {
            "logits": NeuralType(['B'] + ['ANY'] * (self._logits_dim - 1), LogitsType()),
            "labels": NeuralType(['B'] + ['ANY'] * (self._logits_dim - 2), LabelsType()),
            "ys": NeuralType(['B'] + ['ANY'] * (self._logits_dim - 1), ProbsType()),
            "loss_mask": NeuralType(['B'] + ['ANY'] * (self._logits_dim - 2), MaskType(), optional=True),
        }

    @property
    def output_types(self):
        """Returns definitions of module output ports.
        """
        return {"loss": NeuralType(elements_type=LossType())}

    def __init__(self, logits_ndim=2, weight=None, reduction='mean', ignore_index=-100):
        """
        Args:
            logits_ndim (int): number of dimensions (or rank) of the logits tensor
            weight (list): list of rescaling weight given to each class
            reduction (str): type of the reduction over the batch
        """
        if weight is not None and not torch.is_tensor(weight):
            weight = torch.FloatTensor(weight)
            logging.info(f"Weighted Cross Entropy loss with weight {weight}")
        super().__init__(weight=weight, reduction=reduction, ignore_index=ignore_index)
        self._logits_dim = logits_ndim

    @typecheck()
    def forward(self, logits, labels, ys=None, loss_mask=None):
        """
        Args:
            logits (float): output of the classifier
            labels (long): ground truth labels
            loss_mask (bool/float/int): tensor to specify the masking
        """
        logits_flatten = torch.flatten(logits, start_dim=0, end_dim=-2)
        labels_flatten = torch.flatten(labels, start_dim=0, end_dim=-1)
        if ys is not None:
            ys_flatten = torch.flatten(ys, start_dim=0, end_dim=-2)

        if loss_mask is not None:
            if loss_mask.dtype is not torch.bool:
                loss_mask = loss_mask > 0.5
            loss_mask_flatten = torch.flatten(loss_mask, start_dim=0, end_dim=-1)
            logits_flatten = logits_flatten[loss_mask_flatten]
            labels_flatten = labels_flatten[loss_mask_flatten]
            if ys is not None:
                ys_flatten = ys_flatten[loss_mask_flatten]

        if len(labels_flatten) == 0:
            return super().forward(logits, torch.argmax(logits, dim=-1))

        if ys is not None and logits.shape[1] <= 2:  # ys only supported with binary problems
            loss = super().forward(logits_flatten, ys_flatten)
        else:
            loss = super().forward(logits_flatten, labels_flatten)
        return loss


class PrototypicalLoss(Module):
    '''
    Loss class deriving from Module for the prototypical loss function defined below
    '''
    def __init__(self, n_support):
        super(PrototypicalLoss, self).__init__()
        self.n_support = n_support

    def forward(self, input, target):
        return self.prototypical_loss(input, target)


    def prototypical_loss(self, input, target):
        '''
        Inspired by https://github.com/jakesnell/prototypical-networks/blob/master/protonets/models/few_shot.py

        Compute the barycentres by averaging the features of n_support
        samples for each class in target, computes then the distances from each
        samples' features to each one of the barycentres, computes the
        log_probability for each n_query samples for each one of the current
        classes, of appartaining to a class c, loss and accuracy are then computed
        and returned
        Args:
        - input: the model output for a batch of samples
        - target: ground truth for the above batch of samples
        - n_support: number of samples to keep in account when computing barycentres, for each one of the current classes
        '''

        def supp_idxs(c):
            # FIXME when torch will support where as np
            return target.eq(c).nonzero()[:self.n_support].squeeze(1)

        # FIXME when torch.unique will be available on cuda too
        classes = torch.unique(target)
        n_classes = len(classes)
        # FIXME when torch will support where as np
        # assuming n_query, n_target constants
        n_query = target.eq(classes[0].item()).sum().item() - self.n_support

        support_idxs = list(map(supp_idxs, classes))

        prototypes = torch.stack([input[idx_list].mean(0) for idx_list in support_idxs])
        # FIXME when torch will support where as np
        query_idxs = torch.stack(list(map(lambda c: target.eq(c).nonzero()[self.n_support:], classes))).view(-1)

        query_samples = input[query_idxs]
        dists = euclidean_dist(query_samples, prototypes)

        log_p = F.log_softmax(-dists, dim=1)
        log_p_y = log_p.view(n_classes, n_query, -1)

        target_inds = torch.arange(0, n_classes).to(input.device)
        target_inds = target_inds.view(n_classes, 1, 1)
        target_inds = target_inds.expand(n_classes, n_query, 1).long()

        loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
        _, y_hat = log_p_y.max(2)
        acc_val = y_hat.eq(target_inds.squeeze(2)).float().mean()

        return loss_val,  acc_val, log_p, target_inds.flatten(), prototypes, classes # target_cpu[query_idxs], classes[y_hat.view(-1)], log_p


def euclidean_dist(x, y):
    '''
    Compute euclidean distance between two tensors
    '''
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    if d != y.size(1):
        raise Exception

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)
