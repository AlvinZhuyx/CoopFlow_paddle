import paddle
import paddle.nn as nn
import math
import paddle.nn.functional as F


def Conv2D(c_in, c_out, k_sz, stride, padding):
    n = float(c_in * k_sz * k_sz)
    stdv = 1. / math.sqrt(n)
    initializer = nn.initializer.Uniform(low= -stdv, high=stdv)
    param_attr =  paddle.ParamAttr(initializer=initializer)
    return nn.Conv2D(c_in, c_out, k_sz, stride, padding, weight_attr=param_attr, bias_attr=param_attr) 


class block(nn.Layer):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.conv1 = Conv2D(n_in, n_in, 3, 1, 1)
        self.conv2 = Conv2D(n_in, n_out, 3, 1, 1)
        self.res = Conv2D(n_in, n_out, 3, 1, 1)
        self.pooling = nn.AvgPool2D(2, stride=2)

    def forward(self, x):
        h1 = F.swish(self.conv1(x))
        h2 = F.swish(self.conv2(h1))
        h1_res = F.swish(self.res(x))
        return self.pooling(h2 + h1_res)

'''
class EBM_res(nn.Layer):
    def __init__(self, n_c, n_f):
        super(EBM_res, self).__init__()
        self.f = nn.Sequential(
            Conv2D(n_c, n_f, 3, 1, 1),
            nn.Swish(),
            block(n_f, n_f * 2),
            block(n_f * 2, n_f * 4),
            block(n_f * 4, n_f * 8),
            Conv2D(n_f * 8, 1, 4, 1, 0))

    def forward(self, x):
        return self.f(x).squeeze().sum(-1)
'''
def _swish(x):
    _x = paddle.cast(x, 'float64')
    return paddle.cast(_x * F.sigmoid(_x), x.dtype)

class EBM_res(nn.Layer):
    def __init__(self, n_c, n_f):
        super(EBM_res, self).__init__()
        self.h0 = Conv2D(n_c, n_f, 3, 1, 1)

        # block1 n_f --> n_f * 2
        n_in, n_out = n_f, 2 * n_f
        self.h1_conv1 = Conv2D(n_in, n_in, 3, 1, 1)
        self.h1_conv2 = Conv2D(n_in, n_out, 3, 1, 1)
        self.h1_res = Conv2D(n_in, n_out, 3, 1, 1)
        self.h1_pooling = nn.AvgPool2D(2, stride=2)

        # block2 n_f * 2 --> n_f * 4 
        n_in, n_out = 2 * n_f, 4 * n_f
        self.h2_conv1 = Conv2D(n_in, n_in, 3, 1, 1)
        self.h2_conv2 = Conv2D(n_in, n_out, 3, 1, 1)
        self.h2_res = Conv2D(n_in, n_out, 3, 1, 1)
        self.h2_pooling = nn.AvgPool2D(2, stride=2)

        # block3 n_f * 4 --> n_f * 8
        n_in, n_out = 4 * n_f, 8 * n_f
        self.h3_conv1 = Conv2D(n_in, n_in, 3, 1, 1)
        self.h3_conv2 = Conv2D(n_in, n_out, 3, 1, 1)
        self.h3_res = Conv2D(n_in, n_out, 3, 1, 1)
        self.h3_pooling = nn.AvgPool2D(2, stride=2)

        self.h4 = Conv2D(8 * n_f, 100, 4, 1, 0)


    def forward(self, x):
        h0 = _swish(self.h0(x))

        h1_1 = _swish(self.h1_conv1(h0))
        h1_2 = _swish(self.h1_conv2(h1_1))
        h1_res = _swish(self.h1_res(h0))
        h1 = self.h1_pooling(h1_2 + h1_res)

        h2_1 = _swish(self.h2_conv1(h1))
        h2_2 = _swish(self.h2_conv2(h2_1))
        h2_res = _swish(self.h2_res(h1))
        h2 = self.h2_pooling(h2_2 + h2_res)
        h3_1 = _swish(self.h3_conv1(h2))
        h3_2 = _swish(self.h3_conv2(h3_1))
        h3_res = _swish(self.h3_res(h2))
        h3 = self.h3_pooling(h3_2 + h3_res)

        h4 = self.h4(h3)

        return h4.squeeze().sum(-1)



class EBM(nn.Layer):
    def __init__(self, n_c, n_f, l=0.2):
        super(EBM, self).__init__()
        self.f = nn.Sequential(
            Conv2D(n_c, n_f, 3, 1, 1),
            nn.Swish(),
            Conv2D(n_f, n_f * 2, 4, 2, 1),
            nn.Swish(),
            Conv2D(n_f * 2, n_f * 4, 4, 2, 1),
            nn.Swish(),
            Conv2D(n_f * 4, n_f * 8, 4, 2, 1),
            nn.Swish(),
            Conv2D(n_f * 8, 100, 4, 1, 0))

    def forward(self, x):
        return self.f(x).squeeze().sum(-1)