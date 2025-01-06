import torch
import torch.nn as nn
from nemo.collections.asr.parts.preprocessing.features import normalize_batch

try:
    import torchaudio

    HAVE_TORCHAUDIO = True
except ModuleNotFoundError:
    HAVE_TORCHAUDIO = False

CONSTANT = 1e-5


# def normalize_batch(x, seq_len, normalize_type):
#     x_mean = None
#     x_std = None
#     if normalize_type == "per_feature":
#         x_mean = torch.zeros((seq_len.shape[0], x.shape[1]), dtype=x.dtype, device=x.device)
#         x_std = torch.zeros((seq_len.shape[0], x.shape[1]), dtype=x.dtype, device=x.device)
#         for i in range(x.shape[0]):
#             if x[i, :, : seq_len[i]].shape[1] == 1:
#                 raise ValueError(
#                     "normalize_batch with `per_feature` normalize_type received a tensor of length 1. This will result "
#                     "in torch.std() returning nan. Make sure your audio length has enough samples for a single "
#                     "feature (ex. at least `hop_length` for Mel Spectrograms)."
#                 )
#             x_mean[i, :] = x[i, :, : seq_len[i]].mean(dim=1)
#             x_std[i, :] = x[i, :, : seq_len[i]].std(dim=1)
#         # make sure x_std is not zero
#         x_std += CONSTANT
#         return (x - x_mean.unsqueeze(2)) / x_std.unsqueeze(2), x_mean, x_std
#     elif normalize_type == "all_features":
#         x_mean = torch.zeros(seq_len.shape, dtype=x.dtype, device=x.device)
#         x_std = torch.zeros(seq_len.shape, dtype=x.dtype, device=x.device)
#         for i in range(x.shape[0]):
#             x_mean[i] = x[i, :, : seq_len[i].item()].mean()
#             x_std[i] = x[i, :, : seq_len[i].item()].std()
#         # make sure x_std is not zero
#         x_std += CONSTANT
#         return (x - x_mean.view(-1, 1, 1)) / x_std.view(-1, 1, 1), x_mean, x_std
#     elif "fixed_mean" in normalize_type and "fixed_std" in normalize_type:
#         x_mean = torch.tensor(normalize_type["fixed_mean"], device=x.device)
#         x_std = torch.tensor(normalize_type["fixed_std"], device=x.device)
#         return (
#             (x - x_mean.expand(x.shape[0], x.shape[1]).unsqueeze(2)) / x_std.expand(x.shape[0], x.shape[1]).unsqueeze(2),
#             x_mean,
#             x_std,
#         )
#     else:
#         return x, x_mean, x_std


class ExponentialMovingAverage(nn.Module):
    """
    Applies learnable exponential moving average, as required by learnable PCEN layer

    Arguments
    ---------
    input_size : int
        The expected size of the input.
    coeff_init: float
        Initial smoothing coefficient value
    per_channel: bool
        Controls whether every smoothing coefficients are learned
        independently for every input channel
    trainable: bool
        whether to learn the PCEN parameters or use fixed

    Example
    -------
    >>> inp_tensor = torch.rand([10, 50, 40])
    >>> pcen = ExponentialMovingAverage(40)
    >>> out_tensor = pcen(inp_tensor)
    >>> out_tensor.shape
    torch.Size([10, 50, 40])
    """

    def __init__(
        self,
        input_size: int,
        coeff_init: float = 0.04,
        per_channel: bool = False,
        trainable: bool = True,
    ):
        super(ExponentialMovingAverage, self).__init__()
        self._coeff_init = coeff_init
        self._per_channel = per_channel
        self.trainable = trainable
        weights = (
            torch.ones(input_size,) if self._per_channel else torch.ones(1,)
        )
        self._weights = nn.Parameter(
            weights * self._coeff_init, requires_grad=trainable
        )

    def forward(self, x):
        """Returns the normalized input tensor.

       Arguments
        ---------
        x : torch.Tensor (batch, channels, time)
            input to normalize.
        """
        w = torch.clamp(self._weights, min=0.0, max=1.0)
        initial_state = x[:, :, 0]

        def scan(init_state, x, w):
            """Loops and accumulates."""
            x = x.permute(2, 0, 1)
            acc = init_state
            results = []
            for ix in range(x.shape[0]):
                acc = (w * x[ix]) + ((1.0 - w) * acc)
                results.append(acc.unsqueeze(0))
            results = torch.cat(results, dim=0)
            results = results.permute(1, 2, 0)
            return results

        output = scan(initial_state, x, w)
        return output


class PCEN(nn.Module):
    """
    This class implements a learnable Per-channel energy normalization (PCEN) layer, supporting both
    original PCEN as specified in [1] as well as sPCEN as specified in [2]

    [1] Yuxuan Wang, Pascal Getreuer, Thad Hughes, Richard F. Lyon, Rif A. Saurous, "Trainable Frontend For
    Robust and Far-Field Keyword Spotting", in Proc of ICASSP 2017 (https://arxiv.org/abs/1607.05666)

    [2] Neil Zeghidour, Olivier Teboul, F{\'e}lix de Chaumont Quitry & Marco Tagliasacchi, "LEAF: A LEARNABLE FRONTEND
    FOR AUDIO CLASSIFICATION", in Proc of ICLR 2021 (https://arxiv.org/abs/2101.08596)

    The default argument values correspond with those used by [2].

    Arguments
    ---------
    input_size : int
        The expected size of the input.
    alpha: float
        specifies alpha coefficient for PCEN
    smooth_coef: float
        specified smooth coefficient for PCEN
    delta: float
        specifies delta coefficient for PCEN
    root: float
        specifies root coefficient for PCEN
    floor: float
        specifies floor coefficient for PCEN
    trainable: bool
        whether to learn the PCEN parameters or use fixed
    per_channel_smooth_coef: bool
        whether to learn independent smooth coefficients for every channel.
        when True, essentially using sPCEN from [2]

    Example
    -------
    >>> inp_tensor = torch.rand([10, 50, 40])
    >>> pcen = PCEN(40, alpha=0.96)         # sPCEN
    >>> out_tensor = pcen(inp_tensor)
    >>> out_tensor.shape
    torch.Size([10, 50, 40])
    """

    def __init__(
        self,
        input_size: int,
        enabled: bool = True,
        alpha: float = 0.96,
        smooth_coef: float = 0.04,
        delta: float = 2.0,
        root: float = 2.0,
        floor: float = 1e-12,
        trainable: bool = True,
        per_channel_smooth_coef: bool = True,
        normalize: str = "none"
    ):
        super(PCEN, self).__init__()
        self.enabled = enabled
        self._smooth_coef = smooth_coef
        self._floor = floor
        self._per_channel_smooth_coef = per_channel_smooth_coef
        self.normalize = normalize
        self.alpha = nn.Parameter(
            torch.ones(input_size) * alpha, requires_grad=trainable
        )
        self.delta = nn.Parameter(
            torch.ones(input_size) * delta, requires_grad=trainable
        )
        self.root = nn.Parameter(
            torch.ones(input_size) * root, requires_grad=trainable
        )

        self.ema = ExponentialMovingAverage(
            input_size,
            coeff_init=self._smooth_coef,
            per_channel=self._per_channel_smooth_coef,
            trainable=trainable,
        )

    def forward(self, x, x_length):
        """Returns the normalized input tensor.

        Arguments
        ---------
        x : torch.Tensor (batch, time, channels)
            input to normalize.
        x_length: torch.Tensor (batch)
            length of each input sequence
        """
        if not self.enabled:
            return x

        alpha = torch.min(
            self.alpha, torch.tensor(1.0, dtype=x.dtype, device=x.device)
        )
        root = torch.max(
            self.root, torch.tensor(1.0, dtype=x.dtype, device=x.device)
        )
        ema_smoother = self.ema(x)
        one_over_root = 1.0 / root
        output = (
            x / (self._floor + ema_smoother) ** alpha.view(1, -1, 1)
            + self.delta.view(1, -1, 1)
        ) ** one_over_root.view(1, -1, 1) - self.delta.view(
            1, -1, 1
        ) ** one_over_root.view(
            1, -1, 1
        )

        # normalize if required
        if self.normalize:
            output, _, _ = normalize_batch(output, x_length, normalize_type=self.normalize)

        return output, x_length
