# Copyright (c) 2021-present, Zhuang AI Group.
# All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import logging

if 'mesa' not in __name__:
    import custom_quant
    import native
else:
    from . import custom_quant
    from . import native

# Uniform Quantization based Convolution
class conv2d_uniform(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, stride, padding, dilation, groups, clip_val, level, iteration, ema_decay, quant_groups, shift):

        custom_quant.Quant.forward(ctx, x, clip_val, level, iteration, ema_decay, quant_groups, shift)
        x = F.conv2d(x, weight, bias, stride, padding, dilation, groups)

        ctx.conv_weight = (weight, bias)
        ctx.hyperparameters_conv = (stride, padding, dilation, groups)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        grad_input, grad_weight, grad_bias = None, None, None

        weight, bias = ctx.conv_weight
        stride, padding, dilation, groups = ctx.hyperparameters_conv

        x = custom_quant.Quant.restore(ctx)
        # conv
        benchmark = True
        deterministic = True
        allow_tf32 = True
        output_mask = [True, True] #ctx.needs_input_grad[:2]
        grad_output = grad_output.to(dtype=weight.dtype)
        x = x.to(dtype=weight.dtype)
        if torch.__version__ >= "1.7":
            grad_input, grad_weight = native.conv2d_backward(x, grad_output, weight, padding, stride, dilation, groups,
                    benchmark, deterministic, allow_tf32, output_mask)
        else:
            grad_input, grad_weight = native.conv2d_backward(x, grad_output, weight, padding, stride, dilation, groups,
                    benchmark, deterministic, output_mask)
        x = None

        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum((0,2,3)).squeeze(0)

        ctx.conv_weight = None
        ctx.hyperparameters_conv = None
        return grad_input, grad_weight, grad_bias, None, None, None, None, None, None, None, None, None, None, None


class Conv2d(nn.Conv2d, custom_quant.Quant):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, args=None, logger=None, quant_groups=1):
        super(Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        custom_quant.Quant.__init__(self, args=args, logger=logger, quant_groups=quant_groups)
        self.tag = 'conv'

    def __repr__(self):
        return self.__str__()

    def forward(self, x):
        if self.enable and self.training:
            y = conv2d_uniform.apply(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups,
                                     self.clip_val, self.level, self.iteration, self.ema_decay, self.quant_groups, self.shift)
        else:
            y = F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return y
