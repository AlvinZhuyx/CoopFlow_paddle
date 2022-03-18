import paddle.nn as nn
import paddle
import math
import copy

class name_register(object):
    def __init__(self, norm_param):
        self.norm_param = copy.deepcopy(norm_param)
    def name_check(self, name):
        return name.endswith('weight_g') or (name in self.norm_param) 


def get_param_groups(net, weight_decay, norm_suffix='weight_g', verbose=False):
    """Get two parameter groups from `net`: One named "normalized" which will
    override the optimizer with `weight_decay`, and one named "unnormalized"
    which will inherit all hyperparameters from the optimizer.
    Args:
        net (torch.nn.Module): Network to get parameters from
        weight_decay (float): Weight decay to apply to normalized weights.
        norm_suffix (str): Suffix to select weights that should be normalized.
            For WeightNorm, using 'weight_g' normalizes the scale variables.
        verbose (bool): Print out number of normalized and unnormalized parameters.
    """
    norm_params = []
    unnorm_params = []
    norm_params_name = []
    unnorm_params_name = []
    for n, p in net.named_parameters():
        if n.endswith(norm_suffix):
            norm_params.append(p)
            norm_params_name.append(p.name)
        else:
            unnorm_params.append(p)
            unnorm_params_name.append(p.name)
    flow_name_register = name_register(norm_params_name)
    '''
    print('normalized params----------------------------------------')
    for n in norm_params_name:
        print(n)
    print('unnormalized params--------------------------------------')
    for n in unnorm_params_name:
        print(n)
    '''
    param_groups = [{'name': 'normalized', 'params': norm_params, 'weight_decay': weight_decay},
                    {'name': 'unnormalized', 'params': unnorm_params}]

    if verbose:
        print('{} normalized parameters'.format(len(norm_params)))
        print('{} unnormalized parameters'.format(len(unnorm_params)))

    return param_groups, flow_name_register

def Conv2D(c_in, c_out, k_sz, stride=1, padding=0):
    n = float(c_in * k_sz * k_sz)
    stdv = 1. / math.sqrt(n)
    initializer = nn.initializer.Uniform(low= -stdv, high=stdv)
    param_attr =  paddle.ParamAttr(initializer=initializer)
    return nn.Conv2D(c_in, c_out, k_sz, stride, padding, weight_attr=param_attr, bias_attr=param_attr) 

class WNConv2d(nn.Layer):
    """Weight-normalized 2d convolution.
    Args:
        in_channels (int): Number of channels in the input.
        out_channels (int): Number of channels in the output.
        kernel_size (int): Side length of each convolutional kernel.
        padding (int): Padding to add on edges of input.
        bias (bool): Use bias in the convolution operation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, padding, bias=True):
        super(WNConv2d, self).__init__()
        if not bias:
            self.conv = nn.utils.weight_norm(
                Conv2D(in_channels, out_channels, kernel_size, padding=padding, bias_attr=False))
        else:
            self.conv = nn.utils.weight_norm(
                Conv2D(in_channels, out_channels, kernel_size, padding=padding))

    def forward(self, x):
        x = self.conv(x)

        return x
