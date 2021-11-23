# Copyright (c) 2021-present, Zhuang AI Group.
# All rights reserved.

import torch

def packbits_padded(tensor, dim = -1, mask = 0b1, out = None, dtype = torch.uint8):
    dim = dim if dim >= 0 else dim + tensor.dim()
    nbits_element, nbits = 8, (1 if mask == 0b00000001 else 2 if mask == 0b00000011 else 4 if mask == 0b00001111 else 8 if mask == 0b11111111 else None)
    nibbles = nbits_element // nbits
    assert tensor.shape[dim] % nibbles == 0, "shape: {}, dim: {}, nibbles: {}".format(tensor.shape, dim, nibbles)
    
    out = out if out is not None else torch.empty(*tensor.shape[:dim], tensor.shape[dim] // nibbles, *tensor.shape[1 + dim:], dtype = dtype, device = tensor.device)
    shift = torch.arange(nbits_element - nbits, -1, -nbits, dtype = torch.uint8, device = tensor.device)
    shift = shift.view(nibbles, *((1,) * (tensor.dim() - dim - 1)))
    torch.sum(tensor.view(*tensor.shape[:dim], -1, nibbles, *tensor.shape[1 + dim:]) << shift , dim = 1 + dim, out = out)
    return out

def unpackbits_padded(tensor, dim = -1, mask = 0b1, out = None):
    dim = dim if dim >= 0 else dim + tensor.dim()
    nbits_element, nbits = 8, (1 if mask == 0b00000001 else 2 if mask == 0b00000011 else 4 if mask == 0b00001111 else 8 if mask == 0b11111111 else None)
    nibbles = nbits_element // nbits
    
    out = out if out is not None else \
            torch.empty(*tensor.shape[:dim], tensor.shape[dim] * nibbles, *tensor.shape[1 + dim:], dtype = torch.uint8, device = tensor.device)
    shift = torch.arange(nbits_element - nbits, -1, -nbits, dtype = torch.uint8, device = tensor.device)
    shift = shift.view(nibbles, *((1,) * (tensor.dim() - dim - 1)))
    torch.bitwise_and((tensor.unsqueeze(1 + dim) >> shift).view_as(out), mask, out = out)
    return out

