import torch
import numpy as np
import qtorch.quant as quant

def count_parameters(weights):
    total_params = 0
    for name, parameter in weights.items():
        total_params += torch.count_nonzero(parameter, dim=None)
    return total_params


def quantize_weights(weights, bits):
    total_params = count_parameters(weights)
    print(f"Total Trainable Params: {total_params}")
    print(f"Total Size: {round(total_params.cpu().numpy() * 32 / 8 / 1024 / 1024 , 1)}MB")
    q_weights = {}
    total_size_new = 0
    for k, v in weights.items():
        if 'weight' in k or 'bias' in k or 'running_mean' in k:
            v_quantize = (torch.round(v.type(torch.float32) * 2 ** bits) / (2 ** bits)).type(torch.float32)
            total_size_new += torch.count_nonzero(v_quantize, dim=None).item() * bits / 8 / 1024
            # v_quantize = quant.fixed_point_quantize(v, 5, 4, clamp=True, symmetric=False, rounding='stochastic')
        else:
            v_quantize = v
            total_size_new += torch.count_nonzero(v_quantize, dim=None).item() * 32 / 8 / 1024
        q_weights[k] = v_quantize
    total_params_quantize = count_parameters(q_weights)
    print(f"Total Trainable Params after quantize: {total_params_quantize}")
    print(f"Total Size after quantize: {round(total_size_new)}KB")

    return q_weights, total_size_new

def pruning(weights, th, total_size_new):
    total_params = count_parameters(weights)
    p_weights = {}
    for k, v in weights.items():
        v_prun = torch.multiply(v, v.abs() > th)
        p_weights[k] = v_prun
    total_params_prun = count_parameters(p_weights)
    print(f"Total Trainable Params after prune: {total_params_prun}")
    print(f"Total Size after prune: {round(total_size_new - (total_params.cpu().numpy()-total_params_prun.cpu().numpy()) * 32 / 8 / 1024)}KB")
    return p_weights