import torch
import numpy as np

# https://github.com/peerdavid/layerwise-batch-entropy/blob/be2337bf1a3b5d39a634bd13a987a600c0c0d3a6/experiment_fnn/batch_entropy.py
def batch_entropy(x):
    """ Estimate the differential entropy by assuming a gaussian distribution of
        values for different samples of a mini-batch.
    """
    if(x.shape[0] <= 1):
        raise Exception("The batch entropy can only be calculated for |batch| > 1.")

    x = torch.flatten(x, start_dim=1)
    x_std = torch.std(x, dim=0)
    entropies = 0.5 * torch.log(np.pi * np.e * x_std**2 + 1)
    return torch.mean(entropies).item()