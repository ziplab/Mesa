# Copyright (c) 2021-present, Zhuang AI Group.
# All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F

if 'mesa' not in __name__:
    import packbit
    import custom_quant
else:
    from . import packbit
    from . import custom_quant

class relu(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, inplace=False, dim=1, keep_tensor=True):
        if inplace:
            output = x.clamp_(min=0)
        else:
            output = x.clamp(min=0)

        if keep_tensor:
            y = output
        else:
            y = x <= 0
            y = packbit.packbits_padded(y, dim=dim)
            ctx.relu_dim = dim

        ctx.relu_output = y
        ctx.relu_keep_tensor = keep_tensor
        return output

    @staticmethod
    def backward(ctx, grad_output):
        y = ctx.relu_output
        if ctx.relu_keep_tensor:
            y = y <= 0
        else:
            y = packbit.unpackbits_padded(y, dim=ctx.relu_dim).to(dtype=torch.bool)
            ctx.relu_dim = None
        grad_input = grad_output.masked_fill(y, 0)
        ctx.relu_output= None
        ctx.relu_keep_tensor = None
        return grad_input, None, None, None, None

class ReLU(nn.ReLU, custom_quant.Quant):
    def __init__(self, inplace=False, dim=1, args=None, logger=None):
        super(ReLU, self).__init__(inplace)
        self.repr = super(ReLU, self).__repr__()
        custom_quant.Quant.__init__(self, args=args, logger=logger)
        self.dim = dim
        self.keep_tensor = False
        self.tag = 'relu'

    def __repr__(self):
        return self.__str__()

    def forward(self, x):
        if self.enable and self.training:
            y = relu.apply(x, self.inplace, self.dim, self.keep_tensor)
        else:
            y = F.relu(x, inplace=self.inplace)
        return y

if __name__ == "__main__":
    model = ReLU(True, args=None)
    print(model)

    from test import test
    test(model)



