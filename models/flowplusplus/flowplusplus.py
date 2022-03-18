import math
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from models.flowplusplus.act_norm import ActNorm
from models.flowplusplus.inv_conv import InvConv
from models.flowplusplus.nn import GatedConv
from models.flowplusplus.coupling import Coupling
from util import channelwise, checkerboard, Flip, safe_log, squeeze, unsqueeze

def Conv2D(c_in, c_out, kernel_size=3, stride=1, padding=0):
    n = float(c_in * kernel_size * kernel_size)
    stdv = 1. / math.sqrt(n)
    initializer = nn.initializer.Uniform(low= -stdv, high=stdv)
    param_attr =  paddle.ParamAttr(initializer=initializer)
    return nn.Conv2D(c_in, c_out, kernel_size, stride, padding, weight_attr=param_attr, bias_attr=param_attr) 

class FlowPlusPlus(nn.Layer):
    """
    Pytorch implementation of Flow++ Model

    Based on the paper:
    "Flow++: Improving Flow-Based Generative Models
        with Variational Dequantization and Architecture Design"
    by Jonathan Ho, Xi Chen, Aravind Srinivas, Yan Duan, Pieter Abbeel
    (https://openreview.net/forum?id=Hyg74h05tX).

    Args:
        scales (tuple or list): Number of each type of coupling layer in each
            scale. Each scale is a 2-tuple of the form
            (num_channelwise, num_checkerboard).
        in_channels (int): Number of channels in the input.
        mid_channels (int): Number of channels in the intermediate layers.
        num_blocks (int): Number of residual blocks in the s and t network of
            `Coupling` layers.
        num_dequant_blocks (int): Number of blocks in the dequantization flows.
    """
    def __init__(self,
                 scales=((0, 4), (2, 3)),
                 in_shape=(3, 32, 32),
                 mid_channels=96,
                 num_blocks=10,
                 num_dequant_blocks=2,
                 num_components=32,
                 use_attn=True,
                 drop_prob=0.2):
        super(FlowPlusPlus, self).__init__()
        # Register bounds to pre-process images, not learnable
        self.register_buffer('bounds', paddle.to_tensor([0.9], dtype='float32'))
        if num_dequant_blocks > 0:
            self.dequant_flows = _Dequantization(in_shape=in_shape,
                                                 mid_channels=mid_channels,
                                                 num_blocks=num_dequant_blocks,
                                                 use_attn=use_attn,
                                                 drop_prob=drop_prob)
        else:
            self.dequant_flows = None
        self.flows = _FlowStep(scales=scales,
                               in_shape=in_shape,
                               mid_channels=mid_channels,
                               num_blocks=num_blocks,
                               num_components=num_components,
                               use_attn=use_attn,
                               drop_prob=drop_prob)

    def forward(self, x, reverse=False):  
        sldj = paddle.zeros([x.shape[0]])
        if not reverse:
            x = (x + 1.0) / 2.0
            x, sldj = self.dequantize(x, sldj)
            x, sldj = self.to_logits(x, sldj)
        x, sldj = self.flows(x, sldj, reverse)

        return x, sldj

    def dequantize(self, x, sldj):
        if self.dequant_flows is not None:
            x, sldj = self.dequant_flows(x, sldj)
        else:
            x = (x * 255. + paddle.rand_like(x)) / 256.

        return x, sldj

    def to_logits(self, x, sldj):
        """Convert the input image `x` to logits.

        Args:
            x (Paddle.Tensor): Input image.
            sldj (Paddle.Tensor): Sum log-determinant of Jacobian.

        Returns:
            y (Paddle.Tensor): Dequantized logits of `x`.

        See Also:
            - Dequantization: https://arxiv.org/abs/1511.01844, Section 3.1
            - Modeling logits: https://arxiv.org/abs/1605.08803, Section 4.1
        """

        y = (2 * x - 1) * self.bounds
        y = (y + 1) / 2
        y = y.log() - (1. - y).log()
        ldj = F.softplus(y) + F.softplus(-y) \
            - F.softplus((1. - self.bounds).log() - self.bounds.log())
        sldj = sldj + ldj.flatten(1).sum(-1)

        return y, sldj


class _FlowStep(nn.Layer):
    """Recursive builder for a Flow++ model.

    Each `_FlowStep` corresponds to a single scale in Flow++.
    The constructor is recursively called to build a full model.

    Args:
        scales (tuple): Number of each type of coupling layer in each scale.
            Each scale is a 2-tuple of the form (num_channelwise, num_checkerboard).
        in_channels (int): Number of channels in the input.
        mid_channels (int): Number of channels in the intermediate layers.
        num_blocks (int): Number of residual blocks in the s and t network of
            `Coupling` layers.
        num_components (int): Number of components in the mixture.
        use_attn (bool): Use attention in the coupling layers.
        drop_prob (float): Dropout probability.
    """
    def __init__(self, scales, in_shape, mid_channels, num_blocks, num_components, use_attn, drop_prob):
        super(_FlowStep, self).__init__()
        in_channels, in_height, in_width = in_shape
        num_channelwise, num_checkerboard = scales[0]
        channels = []
        for i in range(num_channelwise):
            channels += [ActNorm(in_channels // 2),
                         InvConv(in_channels // 2),
                         Coupling(in_channels=in_channels // 2,
                                  mid_channels=mid_channels,
                                  num_blocks=num_blocks,
                                  num_components=num_components,
                                  use_attn=use_attn,
                                  drop_prob=drop_prob),
                         Flip()]

        checkers = []
        for i in range(num_checkerboard):
            checkers += [ActNorm(in_channels),
                         InvConv(in_channels),
                         Coupling(in_channels=in_channels,
                                  mid_channels=mid_channels,
                                  num_blocks=num_blocks,
                                  num_components=num_components,
                                  use_attn=use_attn,
                                  drop_prob=drop_prob),
                         Flip()]
        self.channels = nn.LayerList(channels) if channels else None
        self.checkers = nn.LayerList(checkers) if checkers else None

        if len(scales) <= 1:
            self.next = None
        else:
            next_shape = (2 * in_channels, in_height // 2, in_width // 2)
            self.next = _FlowStep(scales=scales[1:],
                                  in_shape=next_shape,
                                  mid_channels=mid_channels,
                                  num_blocks=num_blocks,
                                  num_components=num_components,
                                  use_attn=use_attn,
                                  drop_prob=drop_prob)

    def forward(self, x, sldj, reverse=False):
        if reverse:
            if self.next is not None:
                x = squeeze(x)
                x, x_split = x.chunk(2, axis=1)
                x, sldj = self.next(x, sldj, reverse)
                x = paddle.concat((x, x_split), axis=1)
                x = unsqueeze(x)

            if self.checkers:
                x = checkerboard(x)
                #print(len(self.checkers))
                for flow in reversed(self.checkers):
                    #print(type(flow))
                    x, sldj = flow(x, sldj, reverse)
                    #print(type(x))
                x = checkerboard(x, reverse=True)

            if self.channels:
                x = channelwise(x)
                for flow in reversed(self.channels):
                    x, sldj = flow(x, sldj, reverse)
                x = channelwise(x, reverse=True)
        else:
            if self.channels:
                x = channelwise(x)
                for flow in self.channels:
                    x, sldj = flow(x, sldj, reverse)
                x = channelwise(x, reverse=True)

            if self.checkers:
                x = checkerboard(x)
                for flow in self.checkers:
                    x, sldj = flow(x, sldj, reverse)
                x = checkerboard(x, reverse=True)

            if self.next is not None:
                x = squeeze(x)
                x, x_split = x.chunk(2, axis=1)
                x, sldj = self.next(x, sldj, reverse)
                x = paddle.concat((x, x_split), axis=1)
                x = unsqueeze(x)

        return x, sldj


class _Dequantization(nn.Layer):
    """Paddle implementation of Dequantization Network for Flow++

    Args:
        in_shape (int): Shape of the input.
        mid_channels (int): Number of channels in the intermediate layers.
        num_blocks (int): Number of residual blocks in the s and t network of
            `Coupling` layers.
        use_attn (bool): Use attention in the coupling layers.
        drop_prob (float): Dropout probability.
        num_flows (int): Number of InvConv+MLCoupling flows to use.
        aux_channels (int): Number of channels in auxiliary input to couplings.
        num_components (int): Number of components in the mixture.
    """
    def __init__(self, in_shape, mid_channels, num_blocks, use_attn, drop_prob,
                 num_flows=4, aux_channels=32, num_components=32):
        super(_Dequantization, self).__init__()
        in_channels, in_height, in_width = in_shape
        self.aux_conv = nn.Sequential(
            Conv2D(2 * in_channels, aux_channels, kernel_size=3, padding=1),
            GatedConv(aux_channels, drop_prob),
            GatedConv(aux_channels, drop_prob),
            GatedConv(aux_channels, drop_prob))

        flows = []
        for _ in range(num_flows):
            flows += [ActNorm(in_channels),
                      InvConv(in_channels),
                      Coupling(in_channels, mid_channels, num_blocks,
                               num_components, drop_prob,
                               use_attn=use_attn,
                               aux_channels=aux_channels),
                      Flip()]
        self.flows = nn.LayerList(flows)

    def forward(self, x, sldj):
        u = paddle.randn(shape=x.shape, dtype=x.dtype)
        eps_nll = 0.5 * (u ** 2 + math.log(2 * math.pi))

        aux = self.aux_conv(paddle.concat(checkerboard(x - 0.5), axis=1))
        u = checkerboard(u)
        for i, flow in enumerate(self.flows):
            u, sldj = flow(u, sldj, aux=aux) if i % 4 == 2 else flow(u, sldj)
        u = checkerboard(u, reverse=True)

        u = F.sigmoid(u)
        x = (x * 255. + u) / 256.

        sigmoid_ldj = safe_log(u) + safe_log(1. - u)
        sldj = sldj + (eps_nll + sigmoid_ldj).flatten(1).sum(-1)

        return x, sldj
