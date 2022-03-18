import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class Flip(nn.Layer):
    def forward(self, x, sldj, reverse=False):
        #print(type(x), len(x))
        assert (isinstance(x, tuple) or isinstance(x, list)) and len(x) == 2
        return (x[1], x[0]), sldj


def mean_dim(tensor, dim=None, keepdims=False):
    """Take the mean along multiple dimensions.

    Args:
        tensor (paddle.Tensor): Tensor of values to average.
        dim (list): List of dimensions along which to take the mean.
        keepdims (bool): Keep dimensions rather than squeezing.

    Returns:
        mean (paddle.Tensor): New tensor of mean value(s).
    """
    if dim is None:
        return tensor.mean()
    else:
        if isinstance(dim, int):
            dim = [dim]
        dim = sorted(dim)
        for d in dim:
            tensor = tensor.mean(axis=d, keepdim=True)
        if not keepdims:
            for i, d in enumerate(dim):
                tensor.squeeze_(d-i)
        return tensor


def checkerboard(x, reverse=False):
    """Split x in a checkerboard pattern. Collapse horizontally."""
    # Get dimensions
    if reverse:
        b, c, h, w = x[0].shape
        w *= 2
    else:
        b, c, h, w = x.shape

    # Get list of indices in alternating checkerboard pattern
    y_idx = []
    z_idx = []
    for i in range(h):
        for j in range(w):
            if (i % 2) == (j % 2):
                y_idx.append(i * w + j)
            else:
                z_idx.append(i * w + j)
    y_idx = paddle.to_tensor(y_idx, dtype='int64')
    z_idx = paddle.to_tensor(z_idx, dtype='int64')

    if reverse:
        y, z = (paddle.reshape(t, [b, c, h * w // 2]) for t in x)
        # x = paddle.zeros(shape=[b, c, h * w], dtype=y.dtype)
        # x[:, :, y_idx] += y
        # x[:, :, z_idx] += z
        
        x = paddle.zeros(shape=[h * w, b, c], dtype=y.dtype)
        x[y_idx] += y.transpose([2, 0, 1])
        x[z_idx] += z.transpose([2, 0, 1])
        x = x.transpose([1, 2, 0])
        
        x = paddle.reshape(x, [b, c, h, w])

        return x
    else:
        if w % 2 != 0:
            raise RuntimeError('Checkerboard got odd width input: {}'.format(w))
         
        x = paddle.reshape(x, [b, c, h * w])
        '''
        import numpy as np
        
        x_trans = x.transpose([2, 0, 1])
        tmp_y = (x_trans[y_idx]).transpose([1, 2, 0])
        tmp_z = (x_trans[z_idx]).transpose([1, 2, 0])
        y = paddle.reshape(tmp_y, [b, c, h, w // 2])
        z = paddle.reshape(tmp_z, [b, c, h, w // 2])
        # y = paddle.reshape(x[:, :, y_idx], [b, c, h, w // 2])
        # z = paddle.reshape(x[:, :, z_idx], [b, c, h, w // 2])
        '''
        #print(paddle.gather(x, y_idx, axis=2).shape)
        y = paddle.reshape(paddle.gather(x, y_idx, axis=2), [b, c, h, w // 2])
        z = paddle.reshape(paddle.gather(x, z_idx, axis=2), [b, c, h, w // 2])
        return y, z


def channelwise(x, reverse=False):
    """Split x channel-wise."""
    if reverse:
        x = paddle.concat(x, axis=1)
        return x
    else:
        y, z = x.chunk(2, axis=1)
        return y, z


def squeeze(x):
    """Trade spatial extent for channels. I.e., convert each
    1x4x4 volume of input into a 4x1x1 volume of output.

    Args:
        x (paddle.Tensor): Input to squeeze.

    Returns:
        x (paddle.Tensor): Squeezed or unsqueezed tensor.
    """
    b, c, h, w = x.shape
    x = paddle.reshape(x, [b, c, h // 2, 2, w // 2, 2])
    x = paddle.transpose(x, [0, 1, 3, 5, 2, 4])
    x = paddle.reshape(x, [b, c * 2 * 2, h // 2, w // 2])

    return x


def unsqueeze(x):
    """Trade channels channels for spatial extent. I.e., convert each
    4x1x1 volume of input into a 1x4x4 volume of output.

    Args:
        x (paddle.Tensor): Input to unsqueeze.

    Returns:
        x (paddle.Tensor): Unsqueezed tensor.
    """
    b, c, h, w = x.shape
    x = paddle.reshape(x, [b, c // 4, 2, 2, h, w])
    x = paddle.transpose(x, [0, 1, 4, 2, 5, 3])
    x = paddle.reshape(x, [b, c // 4, h * 2, w * 2])

    return x


def concat_elu(x):
    """Concatenated ReLU (http://arxiv.org/abs/1603.05201), but with ELU."""
    return F.elu(paddle.concat((x, -x), axis=1))


def safe_log(x):
    return paddle.log(x.clip(min=1e-22))
