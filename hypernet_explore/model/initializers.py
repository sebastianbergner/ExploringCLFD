import torch.nn as nn
import math


def kaiming(weights, additional_parameter):
    nn.init.kaiming_uniform_(weights, a=math.sqrt(5))


def xavier(weights, additional_parameter):
    nn.init.xavier_uniform_(weights, gain=1.0)


def principled_weight_init_uniform(weights, additional_parameter):
    """'
    This function is based on https://openreview.net/pdf?id=H1lma24tPB
    with some support from the hypnettorch library
    https://hypnettorch.readthedocs.io/en/latest/hnets.html#hypnettorch.hnets.mlp_hnet.HMLP.apply_hyperfan_init
    very helpful https://openreview.net/pdf?id=dIX34JWnIAL 
    Args:
        weights: weight vector of the hypernetwork
        additional_param: last layer of the hypernetwork 
    """
    last_layer, init_bias = additional_parameter
    c_relu = 1 if weights is last_layer else 2 # if last layer then 1 as this layer is linear
    h_bias = 2 if init_bias else 1

    input_variance = 1 # is initialized with torch rand with mean=0 and std=1 therefore var=1
    fan_in_current, _ = nn.init._calculate_fan_in_and_fan_out(weights)
    fan_in_last_layer, _ = nn.init._calculate_fan_in_and_fan_out(last_layer)
    var = c_relu / (h_bias * fan_in_current * fan_in_last_layer * input_variance)
    std = math.sqrt(var)
    a = math.sqrt(3.0) * std
    nn.init.uniform_(weights, -a, a)

def principled_weight_init_normal(weights, additional_parameter):
    """'
    This function is based on https://openreview.net/pdf?id=H1lma24tPB
    with some support from the hypnettorch library
    https://hypnettorch.readthedocs.io/en/latest/hnets.html#hypnettorch.hnets.mlp_hnet.HMLP.apply_hyperfan_init
    very helpful https://openreview.net/pdf?id=dIX34JWnIAL 
    Args:
        weights: weight vector of the hypernetwork
        additional_param: last layer of the hypernetwork 
    """
    last_layer, task_emb, init_bias = additional_parameter
    c_relu = 1 if weights is last_layer else 2 # if last layer then 1 as this layer is linear
    h_bias = 2 if init_bias else 1

    input_variance = 1 # is initialized with torch rand with mean=0 and std=1 therefore var=1
    fan_in_current, _ = nn.init._calculate_fan_in_and_fan_out(weights)
    fan_in_last_layer, _ = nn.init._calculate_fan_in_and_fan_out(last_layer)
    var = c_relu / (h_bias * fan_in_current * fan_in_last_layer * input_variance)
    std = math.sqrt(var)
    a = math.sqrt(3.0) * std
    nn.init._no_grad_normal_(weights, 0, var)