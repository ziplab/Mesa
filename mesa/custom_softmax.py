# Copyright (c) 2021-present, Zhuang AI Group.
# All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F

if 'mesa' not in __name__:
    import custom_quant
    import native
else:
    from . import custom_quant
    from . import native

class softmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, dim,
                clip_val1=None, level1=256, iteration1=None, ema_decay1=None, quant_groups1=None, shift1=None,
                clip_val2=None, level2=256, iteration2=None, ema_decay2=None, quant_groups2=None, shift2=None):
        custom_quant.Quant.forward(ctx, x, clip_val1, level1, iteration1, ema_decay1, quant_groups1, shift1, '_1')
        y = F.softmax(x, dim)
        custom_quant.Quant.forward(ctx, y, clip_val2, level2, iteration2, ema_decay2, quant_groups2, shift2, '_2')
        ctx.dim = dim
        return y

    @staticmethod
    def backward(ctx, grad_output):
        x = custom_quant.Quant.restore(ctx, '_1')
        y = custom_quant.Quant.restore(ctx, '_2')

        if x.is_cuda:
            grad_input = native.softmax_backward_cuda(grad_output, y, ctx.dim, x)
        else:
            grad_input = native.softmax_backward_cpu(grad_output, y, ctx.dim, x)

        return grad_input, None, None, None, None, None, None, None, None, None, None, None, None, None

class Softmax(nn.Softmax):
    def __init__(self, dim=None, args=None, logger=None, quant_groups=1):
        super(Softmax, self).__init__(dim=dim)
        self.quant1 = custom_quant.quantization(tag='softmax-1', quant_groups=quant_groups)
        self.quant2 = custom_quant.quantization(tag='softmax-2', quant_groups=quant_groups)
        self.tag = 'softmax'

    def update_quantization_parameter(self, **parameters):
        self.quant1.update_quantization_parameter(**parameters)
        self.quant2.update_quantization_parameter(**parameters)

    def forward(self, x):
        if self.quant1.enable and self.quant2.enable and self.training:
            y = softmax.apply(x, self.dim,
                              self.quant1.clip_val, self.quant1.level, self.quant1.iteration, self.quant1.ema_decay, self.quant1.quant_groups, self.quant1.shift,
                              self.quant2.clip_val, self.quant2.level, self.quant2.iteration, self.quant2.ema_decay, self.quant2.quant_groups, self.quant2.shift)
        else:
            y = F.softmax(x, self.dim)
        return y

if __name__ == "__main__":
    model = Softmax()
    print(model)
    model.enable = True
    print(model)

    import mesa as  ms
    model = ms.Softmax()
    print(model)
    model.enable = True
    print(model)
