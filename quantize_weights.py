import torch


def count_parameters(weights):
    total_params = 0
    for name, parameter in weights.items():
        total_params += torch.count_nonzero(parameter, dim=None)
    return total_params


def quantize_weights(weights):
    total_params = count_parameters(weights)
    print(f"Total Trainable Params: {total_params}")
    q_weights = {}
    for k, v in weights.items():
        v_quantize = torch.round(v.type(torch.float32) * 2 ** 3) / (2 ** 3)
        q_weights[k] = v_quantize
    total_params_quantize = count_parameters(q_weights)
    print(f"Total Trainable Params after quantize: {total_params_quantize}")
    return q_weights
