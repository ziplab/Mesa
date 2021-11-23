# Copyright (c) 2021-present, Zhuang AI Group.
# All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F

if 'mesa' not in __name__:
    import custom_quant
    import packbit
    import native
else:
    from . import custom_quant
    from . import native
    from . import packbit


def SyncBatchNorm_forward(self, input, weight, bias, running_mean, running_var, eps, momentum, process_group, world_size):
    if not input.is_contiguous(memory_format=torch.channels_last):
        input = input.contiguous()
    if weight is not None:
        weight = weight.contiguous()

    size = int(input.numel() // input.size(1))
    if size == 1 and world_size < 2:
        raise ValueError('Expected more than 1 value per channel when training, got input size {}'.format(size))

    # calculate mean/invstd for input.
    mean, invstd = torch.batch_norm_stats(input, eps)
    
    count = torch.full((1,), input.numel() // input.size(1), dtype=mean.dtype, device=mean.device)
    
    num_channels = input.shape[1]
    # C, C, 1 -> (2C + 1)
    combined = torch.cat([mean, invstd, count], dim=0)
    # world_size * (2C + 1)
    combined_list = [ torch.empty_like(combined) for k in range(world_size) ]
    # Use allgather instead of allreduce since I don't trust in-place operations ..
    dist.all_gather(combined_list, combined, process_group, async_op=False)
    combined = torch.stack(combined_list, dim=0)
    # world_size * (2C + 1) -> world_size * C, world_size * C, world_size * 1
    mean_all, invstd_all, count_all = torch.split(combined, num_channels, dim=1)
    
    # calculate global mean & invstd
    mean, invstd = torch.batch_norm_gather_stats_with_counts(
        input,
        mean_all,
        invstd_all,
        running_mean,
        running_var,
        momentum,
        eps,
        count_all.view(-1)
    )

    self.process_group = process_group

    # apply element-wise normalization
    out = torch.batch_norm_elemt(input, weight, bias, mean, invstd, eps)
    return out

def SyncBatchNorm_backward(saved_input, weight, mean, invstd, count_tensor, process_group, needs_input_grad, grad_output):
    if not grad_output.is_contiguous(memory_format=torch.channels_last):
        grad_output = grad_output.contiguous()
    #saved_input, weight, mean, invstd, count_tensor = self.saved_tensors
    #process_group = self.process_group
    grad_input = grad_weight = grad_bias = None

    # calculate local stats as well as grad_weight / grad_bias
    sum_dy, sum_dy_xmu, grad_weight, grad_bias = torch.batch_norm_backward_reduce(
        grad_output,
        saved_input,
        mean,
        invstd,
        weight,
        True,
        needs_input_grad[0],
        needs_input_grad[1]
    )

    if True:
        # synchronizing stats used to calculate input gradient.
        num_channels = sum_dy.shape[0]
        combined = torch.cat([sum_dy, sum_dy_xmu], dim=0)
        torch.distributed.all_reduce(
            combined, torch.distributed.ReduceOp.SUM, process_group, async_op=False)
        sum_dy, sum_dy_xmu = torch.split(combined, num_channels)

        # backward pass for gradient calculation
        grad_input = torch.batch_norm_backward_elemt(
            grad_output,
            saved_input,
            mean,
            invstd,
            weight,
            sum_dy,
            sum_dy_xmu,
            count_tensor
        )

    return grad_input, grad_weight, grad_bias #, None, None, None, None, None, None

def bn_pre_forward(self, input):
    self._check_input_dim(input)

    if self.momentum is None:
        exponential_average_factor = 0.0
    else:
        exponential_average_factor = self.momentum
        
    if self.training and self.track_running_stats:
        # TODO: if statement only here to tell the jit to skip emitting this when it is None
        if self.num_batches_tracked is not None:  # type: ignore
            self.num_batches_tracked = self.num_batches_tracked + 1  # type: ignore
            if self.momentum is None:  # use cumulative moving average
                exponential_average_factor = 1.0 / float(self.num_batches_tracked)
            else:  # use exponential moving average
                exponential_average_factor = self.momentum

    if self.training:
        bn_training = True
    else:
        bn_training = (self.running_mean is None) and (self.running_var is None)
        
    assert self.running_mean is None or isinstance(self.running_mean, torch.Tensor)
    assert self.running_var is None or isinstance(self.running_var, torch.Tensor)
    running_mean = self.running_mean if not self.training or self.track_running_stats else None
    running_var = self.running_var if not self.training or self.track_running_stats else None

    need_sync = bn_training and input.is_cuda and hasattr(self, 'process_group')
    process_group = None
    world_size = 1
    if need_sync:
        process_group = torch.distributed.group.WORLD
        if self.process_group:
            process_group = self.process_group
        try:
            world_size = torch.distributed.get_world_size(process_group)
        except AssertionError:
            world_size = 1
        need_sync = world_size > 1

    # fallback to framework BN when synchronization is not necessary
    if need_sync:
        if not self.ddp_gpu_size:
            raise AttributeError('SyncBatchNorm is only supported within torch.nn.parallel.DistributedDataParallel')
        
    return exponential_average_factor, bn_training, running_mean, running_var, need_sync, process_group, world_size

class batchnorm2d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, mean, var, average_factor, training, need_sync, process_group, world_size, eps,
                clip_val, level, iteration, ema_decay, quant_groups, shift):
        if need_sync:
            # currently not support
            output = SyncBatchNorm_forward(ctx, input, bn_weight, bn_bias, bn_mean, bn_var, bn_eps, average_factor, process_group, world_size)
        else:
            output, save_mean, save_var, reverse = native.batch_norm_forward(input, weight, bias, mean, var, training, average_factor, eps)
            if training:
                ctx.bn_parameter = (weight, bias, mean, var, save_mean, save_var, reverse, eps)
                custom_quant.Quant.forward(ctx, input, clip_val, level, iteration, ema_decay, quant_groups, shift)
        if training:
            ctx.need_sync = need_sync
        return output

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.need_sync:
            grad_output, grad_bn_weight, grad_bn_bias = SyncBatchNorm_backward(input, bn_weight, bn_mean, bn_invstd, bn_count_all, \
                    bn_process_group, ctx.needs_input_grad[7:9], grad_output)
        else:
            weight, bias, running_mean, running_var, save_mean, save_var, reverse, eps = ctx.bn_parameter
            # input = ctx.bn_input
            input = custom_quant.Quant.restore(ctx)
            grad_input, grad_weight, grad_bias = native.batch_norm_backward(input, grad_output, weight, running_mean, running_var, \
                    save_mean, save_var, 0, reverse)
            ctx.bn_input = None
            ctx.bn_parameter = None
        ctx.need_sync = None

        return grad_input, grad_weight, grad_bias, None, None, None, None, None, None, None, None, None, None, None, None, None, None

class BatchNorm2d(nn.BatchNorm2d, custom_quant.Quant):
    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, args=None, logger=None, quant_groups=1):
        super(BatchNorm2d, self).__init__(num_features, eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats)
        self.repr = super(BatchNorm2d, self).__repr__()
        custom_quant.Quant.__init__(self, args=args, logger=logger, quant_groups=quant_groups)
        self.tag = 'bn'

    def __repr__(self):
        return self.__str__()

    def forward(self, x):
        if self.enable and self.training:
            assert x.is_cuda, "Not supprot cpu mode yet"
            average_factor, training, mean, var, need_sync, process_group, world_size = bn_pre_forward(self, x)
            y = batchnorm2d.apply(x, self.weight, self.bias, mean, var, average_factor, training, need_sync, process_group, world_size, self.eps,
                                  self.clip_val, self.level, self.iteration, self.ema_decay, self.quant_groups, self.shift)
        else:
            y = super().forward(x)
        return y

if __name__ == "__main__":
    model = BatchNorm2d(64, args=None)
    input = torch.randn(4, 100, 35, 45)

    from test import test
    test(model)

