import paddle 
import paddle.nn as nn
import paddle.nn.functional as F 

class FlowPlusPlus(nn.Layer):
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
        # first implement a fake flowplusplus model to debug the other parts of the code
        fake_param = paddle.static.create_parameter(shape=[1, 3, 32, 32], dtype='float32', default_initializer=nn.initializer.Normal())
        self.add_parameter("fake_param", fake_param)

    def forward(self, x, reverse=False):
        if not reverse:
            return x + self.fake_param, paddle.to_tensor([0.0])
        else:
            return x - self.fake_param, paddle.to_tensor([0.0])  
