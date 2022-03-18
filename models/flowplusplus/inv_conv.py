import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class InvConv(nn.Layer):
    """
    Paddle Implementation of Invertible 1x1 Convolution for 2D inputs. Originally described in Glow
    (https://arxiv.org/abs/1807.03039). Does not support LU-decomposed version.

    Args:
        num_channels (int): Number of channels in the input and output.
        random_init (bool): Initialize with a random orthogonal matrix.
            Otherwise initialize with noisy identity.
    """
    def __init__(self, num_channels, random_init=False):
        super(InvConv, self).__init__()
        self.num_channels = 2 * num_channels

        if random_init:
            # Initialize with a random orthogonal matrix
            w_init = np.random.randn(self.num_channels, self.num_channels)
            w_init = np.linalg.qr(w_init)[0]
        else:
            # Initialize as identity permutation with some noise
            w_init = np.eye(self.num_channels, self.num_channels) \
                     + 1e-3 * np.random.randn(self.num_channels, self.num_channels)
        w_init = w_init.astype(np.float32)
        weight = paddle.static.create_parameter(shape=w_init.shape, dtype='float32',\
            default_initializer=paddle.nn.initializer.Assign(w_init))
        self.add_parameter("weight", weight)

    def forward(self, x, sldj, reverse=False):
        x = paddle.concat(x, axis=1)

        ldj = paddle.linalg.slogdet(self.weight)[1] * x.shape[2] * x.shape[3]

        if reverse:
            weight = paddle.cast(paddle.linalg.inv(paddle.cast(self.weight, 'float64')), self.weight.dtype)
            sldj = sldj - ldj
        else:
            weight = self.weight
            sldj = sldj + ldj

        weight = weight.reshape([self.num_channels, self.num_channels, 1, 1])
        x = F.conv2d(x, weight)
        x = x.chunk(2, axis=1)

        return x, sldj
