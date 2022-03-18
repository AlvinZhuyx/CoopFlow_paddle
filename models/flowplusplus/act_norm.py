import paddle
import paddle.nn as nn

from util import mean_dim


class _BaseNorm(nn.Layer):
    """Paddle implementation for the base class for ActNorm (Glow) and PixNorm (Flow++).

    The mean and inv_std get initialized using the mean and variance of the
    first mini-batch. After the init, mean and inv_std are trainable parameters.

    Adapted from:
        > https://github.com/openai/glow
    """
    def __init__(self, num_channels, height, width):
        super(_BaseNorm, self).__init__()

        # Input gets concatenated along channel axis
        num_channels *= 2

        self.register_buffer('is_initialized', paddle.zeros([1]))
        mean = paddle.static.create_parameter(shape=[1, num_channels, height, width], dtype='float32', default_initializer=nn.initializer.Constant(value=0.0)) 
        inv_std = paddle.static.create_parameter(shape=[1, num_channels, height, width], dtype='float32', default_initializer=nn.initializer.Constant(value=0.0)) 
        self.add_parameter("mean", mean)
        self.add_parameter("inv_std", inv_std)
        self.eps = 1e-6

    def initialize_parameters(self, x):
        if not self.training:
            return

        with paddle.no_grad():
            mean, inv_std = self._get_moments(x)
            paddle.assign(mean.detach().clone(), self.mean)
            paddle.assign(inv_std.detach().clone(), self.inv_std)
            self.is_initialized += 1.

    def _center(self, x, reverse=False):
        if reverse:
            return x + self.mean
        else:
            return x - self.mean

    def _get_moments(self, x):
        raise NotImplementedError('Subclass of _BaseNorm must implement _get_moments')

    def _scale(self, x, sldj, reverse=False):
        raise NotImplementedError('Subclass of _BaseNorm must implement _scale')

    def forward(self, x, ldj=None, reverse=False):
        x = paddle.concat(x, axis=1)
        if not self.is_initialized:
            #print("initializing act norm")
            self.initialize_parameters(x)

        if reverse:
            x, ldj = self._scale(x, ldj, reverse)
            x = self._center(x, reverse)
        else:
            x = self._center(x, reverse)
            x, ldj = self._scale(x, ldj, reverse)
        x = x.chunk(2, axis=1)

        return x, ldj


class ActNorm(_BaseNorm):
    """
    Paddle implementation of Activation Normalization used in Glow
    The mean and inv_std get initialized using the mean and variance of the
    first mini-batch. After the init, mean and inv_std are trainable parameters.
    """
    def __init__(self, num_channels):
        super(ActNorm, self).__init__(num_channels, 1, 1)

    def _get_moments(self, x):
        mean = mean_dim(x.clone(), dim=[0, 2, 3], keepdims=True)
        var = mean_dim((x.clone() - mean) ** 2, dim=[0, 2, 3], keepdims=True)
        inv_std = 1. / (var.sqrt() + self.eps)

        return mean, inv_std

    def _scale(self, x, sldj, reverse=False):
        if reverse:
            x = x / self.inv_std
            sldj = sldj - self.inv_std.log().sum() * x.shape[2] * x.shape[3]
        else:
            x = x * self.inv_std
            sldj = sldj + self.inv_std.log().sum() * x.shape[2] * x.shape[3]

        return x, sldj


class PixNorm(_BaseNorm):
    """
    Paddle implementation of Pixel-wise Activation Normalization used in Flow++

    Normalizes every activation independently (note this differs from the variant
    used in in Glow, where they normalize each channel). The mean and stddev get
    initialized using the mean and stddev of the first mini-batch. After the
    initialization, `mean` and `inv_std` become trainable parameters.
    """
    def _get_moments(self, x):
        mean = paddle.mean(x.clone(), axis=0, keepdim=True)
        var = paddle.mean((x.clone() - mean) ** 2, axis=0, keepdim=True)
        inv_std = 1. / (var.sqrt() + self.eps)

        return mean, inv_std

    def _scale(self, x, sldj, reverse=False):
        if reverse:
            x = x / self.inv_std
            sldj = sldj - self.inv_std.log().sum()
        else:
            x = x * self.inv_std
            sldj = sldj + self.inv_std.log().sum()

        return x, sldj
