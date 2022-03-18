import math
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from paddle.nn.utils import weight_norm
from util import concat_elu, WNConv2d


class NN(nn.Layer):
    """Neural network used to parametrize the transformations of an MLCoupling.

    An `NN` is a stack of blocks, where each block consists of the following
    two layers connected in a residual fashion:
      1. Conv: input -> nonlinearit -> conv3x3 -> nonlinearity -> gate
      2. Attn: input -> conv1x1 -> multihead self-attention -> gate,
    where gate refers to a 1×1 convolution that doubles the number of channels,
    followed by a gated linear unit (Dauphin et al., 2016).
    The convolutional layer is identical to the one used by PixelCNN++
    (Salimans et al., 2017), and the multi-head self attention mechanism we
    use is identical to the one in the Transformer (Vaswani et al., 2017).

    Args:
        in_channels (int): Number of channels in the input.
        num_channels (int): Number of channels in each block of the network.
        num_blocks (int): Number of blocks in the network.
        num_components (int): Number of components in the mixture.
        drop_prob (float): Dropout probability.
        use_attn (bool): Use attention in each block.
        aux_channels (int): Number of channels in optional auxiliary input.
    """
    def __init__(self, in_channels, num_channels, num_blocks, num_components, drop_prob, use_attn=True, aux_channels=None):
        super(NN, self).__init__()
        self.k = num_components  # k = number of mixture components
        self.in_conv = WNConv2d(in_channels, num_channels, kernel_size=3, padding=1)
        self.mid_convs = nn.LayerList([ConvAttnBlock(num_channels, drop_prob, use_attn, aux_channels)
                                        for _ in range(num_blocks)])
        self.out_conv = WNConv2d(num_channels, in_channels * (2 + 3 * self.k),
                                 kernel_size=3, padding=1)
        self.rescale = weight_norm(Rescale(in_channels))

    def forward(self, x, aux=None):
        b, c, h, w = x.shape
        x = self.in_conv(x)
        for conv in self.mid_convs:
            x = conv(x, aux)
        x = self.out_conv(x)

        # Split into components and post-process
        x = paddle.reshape(x, [b, -1, c, h, w])
        s, t, pi, mu, scales = x.split([1, 1, self.k, self.k, self.k], axis=1)
        s = self.rescale(paddle.tanh(s.squeeze(1)))
        t = t.squeeze(1)
        scales = scales.clip(min=-7)  # From the code in original Flow++ paper

        return s, t, pi, mu, scales


class ConvAttnBlock(nn.Layer):
    def __init__(self, num_channels, drop_prob, use_attn, aux_channels):
        super(ConvAttnBlock, self).__init__()
        self.conv = GatedConv(num_channels, drop_prob, aux_channels)
        self.norm_1 = nn.LayerNorm(num_channels)
        if use_attn:
            self.attn = GatedAttn(num_channels, drop_prob=drop_prob)
            self.norm_2 = nn.LayerNorm(num_channels)
        else:
            self.attn = None

    def forward(self, x, aux=None):
        #assert False
        x = self.conv(x, aux) + x
        x = paddle.transpose(x, [0, 2, 3, 1])  # (b, h, w, c)
        x = self.norm_1(x)
        
        if self.attn:
            x = self.attn(x) + x
            x = self.norm_2(x)
        x = paddle.transpose(x, [0, 3, 1, 2])  # (b, c, h, w)

        return x


def Linear(d_in, d_out, with_bias=True):
    stdv = 1. / math.sqrt(d_in)
    initializer = nn.initializer.Uniform(low= -stdv, high=stdv)
    param_attr =  paddle.ParamAttr(initializer=initializer)
    if with_bias:
        return nn.Linear(d_in, d_out, weight_attr=param_attr, bias_attr=param_attr)
    else:
        return nn.Linear(d_in, d_out, weight_attr=param_attr, bias_attr=False)

class GatedAttn(nn.Layer):
    """
    Paddle implementation of Gated Multi-Head Self-Attention Block

    Based on the paper:
    "Attention Is All You Need"
    by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones,
        Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin
    (https://arxiv.org/abs/1706.03762).

    Args:
        d_model (int): Number of channels in the input.
        num_heads (int): Number of attention heads.
        drop_prob (float): Dropout probability.
    """
    def __init__(self, d_model, num_heads=4, drop_prob=0.):
        super(GatedAttn, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.drop_prob = drop_prob
        self.in_proj = weight_norm(Linear(d_model, 3 * d_model, with_bias=False))
        self.gate = weight_norm(Linear(d_model, 2 * d_model))

    def forward(self, x):
        # Flatten and encode position
        #print(x.shape)
        b, h, w, c = x.shape
        x = paddle.reshape(x, [b, h * w, c])
        _, seq_len, num_channels = x.shape
        pos_encoding = self.get_pos_enc(seq_len, num_channels)
        x = x + pos_encoding

        # Compute q, k, v
        memory, query = paddle.split(self.in_proj(x), (2 * c, c), axis=-1)
        #print('memory query.shape', self.in_proj(x).shape, memory.shape,  query.shape)
        q = self.split_last_dim(query, self.num_heads)
        '''
        print('Memory shape', memory.shape, self.d_model)
        for tensor in paddle.split(memory, self.d_model, axis=2):
            print(tensor.shape)
        assert False
        '''
        num_split = math.ceil(float(memory.shape[2]) / self.d_model)
        k, v = [self.split_last_dim(tensor, self.num_heads)
                for tensor in paddle.split(memory, num_split, axis=2)]

        # Compute attention and reshape
        key_depth_per_head = self.d_model // self.num_heads
        q *= key_depth_per_head ** -0.5
        x = self.dot_product_attention(q, k, v)
        x = self.combine_last_two_dim(paddle.transpose(x, [0, 2, 1, 3]))
        x = x.transpose([0, 2, 1]).reshape([b, c, h, w]).transpose([0, 2, 3, 1])  # (b, h, w, c)

        x = self.gate(x)
        a, b = x.chunk(2, axis=-1)
        x = a * F.sigmoid(b)

        return x

    def dot_product_attention(self, q, k, v, bias=False):
        """Dot-product attention.

        Args:
            q (paddle.Tensor): Queries of shape (batch, heads, length_q, depth_k)
            k (paddle.Tensor): Keys of shape (batch, heads, length_kv, depth_k)
            v (paddle.Tensor): Values of shape (batch, heads, length_kv, depth_v)
            bias (bool): Use bias for attention.

        Returns:
            attn (paddle.Tensor): Output of attention mechanism.
        """
        weights = paddle.matmul(q, k.transpose([0, 1, 3, 2]))
        if bias:
            weights += self.bias
        weights = F.softmax(weights, axis=-1)
        weights = F.dropout(weights, self.drop_prob, self.training)
        attn = paddle.matmul(weights, v)

        return attn

    @staticmethod
    def split_last_dim(x, n):
        """Reshape x so that the last dimension becomes two dimensions.
        The first of these two dimensions is n.
        Args:
            x (paddle.Tensor): Tensor with shape (..., m)
            n (int): Size of second-to-last dimension.
        Returns:
            ret (paddle.Tensor): Resulting tensor with shape (..., n, m/n)
        """
        old_shape = list(x.shape)
        last = old_shape[-1]
        new_shape = old_shape[:-1] + [n] + [last // n if last else None]
        ret = x.reshape(new_shape)

        return ret.transpose([0, 2, 1, 3])

    @staticmethod
    def combine_last_two_dim(x):
        """Merge the last two dimensions of `x`.

        Args:
            x (torch.Tensor): Tensor with shape (..., m, n)

        Returns:
            ret (torch.Tensor): Resulting tensor with shape (..., m * n)
        """
        old_shape = list(x.shape)
        a, b = old_shape[-2:]
        new_shape = old_shape[:-2] + [a * b if a and b else None]
        ret = x.reshape(new_shape)

        return ret

    @staticmethod
    def get_pos_enc(seq_len, num_channels, device=None):
        position = paddle.arange(seq_len, dtype='float32')
        num_timescales = num_channels // 2
        log_timescale_increment = math.log(10000.) / (num_timescales - 1)
        inv_timescales = paddle.arange(num_timescales, dtype='float32')
        inv_timescales *= -log_timescale_increment
        inv_timescales = inv_timescales.exp_()
        scaled_time = position.unsqueeze(1) * inv_timescales.unsqueeze(0)
        encoding = paddle.concat([paddle.sin(scaled_time), paddle.cos(scaled_time)], axis=1)
        encoding = F.pad(encoding, [0, num_channels % 2, 0, 0])
        encoding = encoding.reshape([1, seq_len, num_channels])

        return encoding


class GatedConv(nn.Layer):
    """Gated Convolution Block

    Originally used by PixelCNN++ (https://arxiv.org/pdf/1701.05517).

    Args:
        num_channels (int): Number of channels in hidden activations.
        drop_prob (float): Dropout probability.
        aux_channels (int): Number of channels in optional auxiliary input.
    """
    def __init__(self, num_channels, drop_prob=0., aux_channels=None):
        super(GatedConv, self).__init__()
        self.nlin = concat_elu
        self.conv = WNConv2d(2 * num_channels, num_channels, kernel_size=3, padding=1)
        self.drop = nn.Dropout2D(drop_prob)
        self.gate = WNConv2d(2 * num_channels, 2 * num_channels, kernel_size=1, padding=0)
        if aux_channels is not None:
            self.aux_conv = WNConv2d(2 * aux_channels, num_channels, kernel_size=1, padding=0)
        else:
            self.aux_conv = None

    def forward(self, x, aux=None):
        x = self.nlin(x)
        x = self.conv(x)
        if aux is not None:
            aux = self.nlin(aux)
            x = x + self.aux_conv(aux)
        x = self.nlin(x)
        x = self.drop(x)
        x = self.gate(x)
        a, b = x.chunk(2, axis=1)
        x = a * F.sigmoid(b)

        return x


class Rescale(nn.Layer):
    """
    Paddle implementation of Per-channel rescaling. Need a proper `nn.Layer` so we can wrap it
    with `torch.nn.utils.weight_norm`.
    Args:
        num_channels (int): Number of channels in the input.
    """

    def __init__(self, num_channels):
        super(Rescale, self).__init__()
        weight = paddle.static.create_parameter(shape=[num_channels, 1, 1], dtype='float32', default_initializer=nn.initializer.Constant(value=1.0)) 
        self.add_parameter("weight", weight)

    def forward(self, x):
        x = self.weight * x
        return x
