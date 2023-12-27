import numpy as np
import random
from matplotlib import animation, rc
import torch
from torchdiffeq import odeint
import os

def check_cuda():
    """
    Checks if GPU is available.
    """    
    cuda_available = torch.cuda.is_available()
    device = torch.device('cuda' if cuda_available else 'cpu')
    return cuda_available, device

def set_seed(seed=1000):
    """
    Sets the seed for reproducability

    Args:
        seed (int, optional): Input seed. Defaults to 1000.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    cuda_available, _ = check_cuda()
    
    if cuda_available:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

def get_sequence(seq_file):
    """
    Returns a list of containing each line of `seq_file`
    as an element

    Args:
        seq_file (str): File with name of demonstration files
                        in each line

    Returns:
        [str]: List of demonstration files
    """
    seq = None
    with open(seq_file) as x:
        seq = [line.strip() for line in x]
    return seq
    
def get_beta_for_tasks(initial_beta, beta_decay, num_tasks):
    """
    Returns a list of betas for each task where beta is decayed using
    beta *= beta_decay from the second task onward.
    """

    betas = list()
    beta = initial_beta

    # Task 0 does not have regularization
    # So for task 0 and task 1, initial beta is used
    betas.append(beta)

    for i in range(num_tasks-1):
        betas.append(beta)
        beta *= beta_decay

    return betas
    
