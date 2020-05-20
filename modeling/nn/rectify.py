##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Hang Zhang
## Email: zhanghang0704@gmail.com
## Copyright (c) 2020
##
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

"""Rectify Module"""
import warnings

import torch
from torch.nn import Conv2d
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

from torch.autograd import Function
from .. import lib

__all__ = ['rectify', 'RFConv2d']

class _rectify(Function):
    @staticmethod
    def forward(ctx, y, x, kernel_size, stride, padding, dilation, average):
        ctx.save_for_backward(x)
        # assuming kernel_size is 3
        kernel_size = [k + 2 * (d - 1) for k,d in zip(kernel_size, dilation)]
        ctx.kernel_size = kernel_size
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.average = average
        if x.is_cuda:
            lib.gpu.conv_rectify(y, x, kernel_size, stride, padding, dilation, average)
        else:
            lib.cpu.conv_rectify(y, x, kernel_size, stride, padding, dilation, average)
        ctx.mark_dirty(y)
        return y

    @staticmethod
    def backward(ctx, grad_y):
        x, = ctx.saved_variables
        if x.is_cuda:
            lib.gpu.conv_rectify(grad_y, x, ctx.kernel_size, ctx.stride,
                                 ctx.padding, ctx.dilation, ctx.average)
        else:
            lib.cpu.conv_rectify(grad_y, x, ctx.kernel_size, ctx.stride,
                                 ctx.padding, ctx.dilation, ctx.average)
        ctx.mark_dirty(grad_y)
        return grad_y, None, None, None, None, None, None

rectify = _rectify.apply


class RFConv2d(Conv2d):
    """Rectified Convolution
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros',
                 average_mode=False):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        self.rectify = average_mode or (padding[0] > 0 or padding[1] > 0)
        self.average = average_mode

        super(RFConv2d, self).__init__(
                 in_channels, out_channels, kernel_size, stride=stride,
                 padding=padding, dilation=dilation, groups=groups,
                 bias=bias, padding_mode=padding_mode)

    def _conv_forward(self, input, weight):
        if self.padding_mode != 'zeros':
            return F.conv2d(F.pad(input, self._padding_repeated_twice, mode=self.padding_mode),
                            weight, self.bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        return F.conv2d(input, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def forward(self, input):
        output = self._conv_forward(input, self.weight)
        if self.rectify:
            output = rectify(output, input, self.kernel_size, self.stride,
                             self.padding, self.dilation, self.average)
        return output

    def extra_repr(self):
        return super().extra_repr() + ', rectify={}, average_mode={}'. \
            format(self.rectify, self.average)
