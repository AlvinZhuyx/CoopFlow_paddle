import os
import six
import pickle
import paddle
import pathlib
import warnings
import math
import numpy as np
from PIL import Image
from typing import Union, Optional, List, Tuple, Text, BinaryIO

def save(state_dicts, file_name):
    def convert(state_dict):
        model_dict = {}

        for k, v in state_dict.items():
            if isinstance(
                    v,
                (paddle.fluid.framework.Variable, paddle.fluid.core.VarBase)):
                model_dict[k] = v.numpy()
            else:
                model_dict[k] = v

        return model_dict

    final_dict = {}
    for k, v in state_dicts.items():
        if isinstance(
                v,
            (paddle.fluid.framework.Variable, paddle.fluid.core.VarBase)):
            final_dict = convert(state_dicts)
            break
        elif isinstance(v, dict):
            final_dict[k] = convert(v)
        else:
            final_dict[k] = v

    with open(file_name, 'wb') as f:
        pickle.dump(final_dict, f, protocol=2)


def load(file_name):
    with open(file_name, 'rb') as f:
        state_dicts = pickle.load(f) if six.PY2 else pickle.load(
            f, encoding='latin1')
    return state_dicts


@paddle.no_grad()
def make_grid(tensor: Union[paddle.Tensor, List[paddle.Tensor]],
              nrow: int=8,
              padding: int=2,
              normalize: bool=False,
              value_range: Optional[Tuple[int, int]]=None,
              scale_each: bool=False,
              pad_value: int=0,
              **kwargs) -> paddle.Tensor:
    if not (isinstance(tensor, paddle.Tensor) or
            (isinstance(tensor, list) and all(
                isinstance(t, paddle.Tensor) for t in tensor))):
        raise TypeError(
            f'tensor or list of tensors expected, got {type(tensor)}')

    if "range" in kwargs.keys():
        warning = "range will be deprecated, please use value_range instead."
        warnings.warn(warning)
        value_range = kwargs["range"]

    # if list of tensors, convert to a 4D mini-batch Tensor
    if isinstance(tensor, list):
        tensor = paddle.stack(tensor, axis=0)

    if tensor.dim() == 2:  # single image H x W
        tensor = tensor.unsqueeze(0)
    if tensor.dim() == 3:  # single image
        if tensor.shape[0] == 1:  # if single-channel, convert to 3-channel
            tensor = paddle.concat((tensor, tensor, tensor), 0)
        tensor = tensor.unsqueeze(0)

    if tensor.dim() == 4 and tensor.shape[1] == 1:  # single-channel images
        tensor = paddle.concat((tensor, tensor, tensor), 1)

    if normalize is True:
        if value_range is not None:
            assert isinstance(value_range, tuple), \
                "value_range has to be a tuple (min, max) if specified. min and max are numbers"

        def norm_ip(img, low, high):
            img.clip(min=low, max=high)
            img = img - low
            img = img / max(high - low, 1e-5)

        def norm_range(t, value_range):
            if value_range is not None:
                norm_ip(t, value_range[0], value_range[1])
            else:
                norm_ip(t, float(t.min()), float(t.max()))

        if scale_each is True:
            for t in tensor:  # loop over mini-batch dimension
                norm_range(t, value_range)
        else:
            norm_range(tensor, value_range)

    if tensor.shape[0] == 1:
        return tensor.squeeze(0)

    # make the mini-batch of images into a grid
    nmaps = tensor.shape[0]
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.shape[2] + padding), int(tensor.shape[3] +
                                                        padding)
    num_channels = tensor.shape[1]
    grid = paddle.full((num_channels, height * ymaps + padding,
                        width * xmaps + padding), pad_value)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            grid[:, y * height + padding:(y + 1) * height, x * width + padding:(
                x + 1) * width] = tensor[k]
            k = k + 1
    return grid


@paddle.no_grad()
def save_image(tensor: Union[paddle.Tensor, List[paddle.Tensor]],
               fp: Union[Text, pathlib.Path, BinaryIO],
               format: Optional[str]=None,
               **kwargs) -> None:
    grid = make_grid(tensor, **kwargs)
    ndarr = paddle.clip(grid * 255 + 0.5, 0, 255).transpose(
        [1, 2, 0]).cast("uint8").numpy()
    im = Image.fromarray(ndarr)
    im.save(fp, format=format)