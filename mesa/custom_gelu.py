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

class gelu(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, clip_val=None, level=256, iteration=None, ema_decay=None, quant_groups=None, shift=None):
        custom_quant.Quant.forward(ctx, x, clip_val, level, iteration, ema_decay, quant_groups, shift)
        y = F.gelu(x)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        x = custom_quant.Quant.restore(ctx)
        if x.is_cuda:
            grad_input = native.gelu_backward_cuda(grad_output, x)
        else:
            grad_input = native.gelu_backward_cpu(grad_output, x)

        return grad_input, None, None, None, None, None, None

class GELU(nn.GELU, custom_quant.Quant):
    def __init__(self, args=None, logger=None, quant_groups=1):
        super(GELU, self).__init__()
        custom_quant.Quant.__init__(self, args=args, logger=logger, quant_groups=quant_groups)
        self.tag = 'gelu'

    def __repr__(self):
        return self.__str__()

    def forward(self, x):
        if self.enable and self.training:
            y = gelu.apply(x, self.clip_val, self.level,
                           self.iteration, self.ema_decay, self.quant_groups, self.shift)
        else:
            y = F.gelu(x)
        return y

if __name__ == "__main__":
    model = GELU()
    print(model)
    model.enable = True
    print(model)

    import mesa as  ms
    model = ms.GELU()
    print(model)
    model.enable = True
    print(model)
