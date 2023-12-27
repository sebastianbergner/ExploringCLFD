import logging
import math

import torch
import torch.nn.functional as F
from torch import nn

from hypernet_explore.model.lsddm import configure as configure_base
from hypernet_explore.model.lsddm_t import configure as configure_base_t

logger = logging.getLogger(__file__)

class Dynamics(nn.Module):
    def __init__(self, alpha=0.01, fhat_activation=nn.ReLU(), props=None):
        super().__init__()
        self.alpha = alpha

        self.weights = None
        self.param_names = None

        self.fhat_activation = fhat_activation
        self.props = props
        self.device = props['device']

    def forward(self, x):

        param_names = self.param_names
        weights = self.weights

        # Compose parameters for fhat
        fhat = list()
        for i, name in enumerate(param_names):
            if name.startswith('fhat'):
                fhat.append(weights[i])

        # Compose parameters for ICNN
        count_V_f_W = len([param_name for param_name in param_names if param_name.startswith('V.f.W.')])
        count_V_f_U = len([param_name for param_name in param_names if param_name.startswith('V.f.U.')])
        count_V_f_bias = len([param_name for param_name in param_names if param_name.startswith('V.f.bias.')])

        idx_V_f_W = [param_names.index(f'V.f.W.{i}') for i in range(count_V_f_W)]
        idx_V_f_U = [param_names.index(f'V.f.U.{i}') for i in range(count_V_f_U)]
        idx_V_f_bias = [param_names.index(f'V.f.bias.{i}') for i in range(count_V_f_bias)]

        V_f_W = [weights[i] for i in idx_V_f_W]
        V_f_U = [weights[i] for i in idx_V_f_U]
        V_f_bias = [weights[i] for i in idx_V_f_bias]

        #print(f'count_fhat={len(fhat)}')    
        #print(f'count_V_f_W={count_V_f_W}')
        #print(f'count_V_f_U={count_V_f_U}')
        #print(f'count_V_f_bias={count_V_f_bias}')

        #print('fhat: ', [i.shape for i in fhat])
        #print('V_f_W: ', [i.shape for i in V_f_W])
        #print('V_f_U: ', [i.shape for i in V_f_U])
        #print('V_f_bias: ', [i.shape for i in V_f_bias])

        ############ Forward through fhat ############

        w_weights = []
        b_weights = []
        for i, p in enumerate(fhat):
            if i % 2 == 1:
                b_weights.append(p)
            else:
                w_weights.append(p)

        # If the input does not have a batch dimension, insert it
        if len(list(x.shape)) == 1:
            hidden = x.unsqueeze(-1)
        else:
            hidden = x

        for l in range(len(w_weights)):
            W = w_weights[l]
            b = b_weights[l]

            # Linear layer.
            hidden = F.linear(hidden, W, bias=b)

            # Only for hidden layers.
            if l < len(w_weights) - 1:

                # Non-linearity
                hidden = self.fhat_activation(hidden)

        fhat_out = hidden

        # Insert 1 as the last output
        extra_ones = torch.ones((fhat_out.shape[0], 1)).to(fhat_out.device)
        fhat_out = torch.cat((fhat_out, extra_ones), dim=-1)

        ############ Forward through ICNN ############

        icnn_activation = ReHU(float(self.props["rehu"]))

        def icnn(x_):

            x_ = x_.to(self.device)

            z = F.linear(x_, V_f_W[0], V_f_bias[0])
            z = icnn_activation(z)

            for W,b,U in zip(V_f_W[1:-1], V_f_bias[1:-1], V_f_U[:-1]):
                z = F.linear(x_, W, b) + F.linear(z, F.softplus(U)) / U.shape[0]
                z = icnn_activation(z)

            return F.linear(x_, V_f_W[-1], V_f_bias[-1]) + F.linear(z, F.softplus(V_f_U[-1])) / V_f_U[-1].shape[0]

        icnn_out = icnn(x)

        ############ Forward through MakePSD ############
        lsd = int(self.props["latent_space_dim"])
        psd_zero = torch.nn.Parameter(icnn(torch.zeros(1,lsd+1)), requires_grad=False)
        psd_eps = float(self.props["projfn_eps"])
        psd_d = 1.0
        psd_rehu = ReHU(psd_d)

        psd_smoothed_output = psd_rehu(icnn_out - psd_zero)
        psd_quadratic_under = psd_eps*(x**2).sum(1,keepdim=True)
        psd_out = psd_smoothed_output + psd_quadratic_under


        ############ LSDDM output ############

        fx = fhat_out
        Vx = psd_out

        # autograd.grad computes the gradients and returns them without
        # populating the .grad field of the leaves of the computation graph
        gV = torch.autograd.grad([a for a in Vx], [x], create_graph=True, only_inputs=True, allow_unused=True)[0]

        # Implementation of equation 10 from paper
        rv = fx - gV * (F.relu((gV*fx).sum(dim=1) + self.alpha*Vx[:,0])/(gV**2).sum(dim=1))[:,None]

        # Remove the last dimension
        rv = rv[:, :-1]

        return rv

    def get_param_shapes(self):

        if self.props["explicit_time"]==1:
            dummy_model = configure_base_t(props=self.props)
        elif self.props["explicit_time"]==0:
            dummy_model = configure_base(props=self.props)

        param_names = list()
        param_shapes = list()
        for n, p in dummy_model.named_parameters():
            if p.requires_grad:
                param_names.append(n)
                param_shapes.append(list(p.shape))
        return param_names, param_shapes

    def set_weights(self, weights, param_names):  

        self.weights = weights
        self.param_names = param_names


class ICNN(nn.Module):
    def __init__(self, layer_sizes, activation=F.relu_):
        super().__init__()
        self.W = nn.ParameterList([nn.Parameter(torch.Tensor(l, layer_sizes[0])) 
                                   for l in layer_sizes[1:]])
        self.U = nn.ParameterList([nn.Parameter(torch.Tensor(layer_sizes[i+1], layer_sizes[i]))
                                   for i in range(1,len(layer_sizes)-1)])
        self.bias = nn.ParameterList([nn.Parameter(torch.Tensor(l)) for l in layer_sizes[1:]])
        self.act = activation
        self.reset_parameters()
        # logger.info(f"Initialized ICNN with {self.act} activation")

    def reset_parameters(self):
        # copying from PyTorch Linear
        for W in self.W:
            nn.init.kaiming_uniform_(W, a=5**0.5)
        for U in self.U:
            nn.init.kaiming_uniform_(U, a=5**0.5)
        for i,b in enumerate(self.bias):
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.W[i])
            bound = 1 / (fan_in**0.5)
            nn.init.uniform_(b, -bound, bound)

    def forward(self, x):
        z = F.linear(x, self.W[0], self.bias[0])
        z = self.act(z)

        for W,b,U in zip(self.W[1:-1], self.bias[1:-1], self.U[:-1]):
            z = F.linear(x, W, b) + F.linear(z, F.softplus(U)) / U.shape[0]
            z = self.act(z)

        return F.linear(x, self.W[-1], self.bias[-1]) + F.linear(z, F.softplus(self.U[-1])) / self.U[-1].shape[0]



class ReHU(nn.Module):
    """ Rectified Huber unit"""
    def __init__(self, d):
        super().__init__()
        self.a = 1/d
        self.b = -d/2

    def forward(self, x):
        return torch.max(torch.clamp(torch.sign(x)*self.a/2*x**2,min=0,max=-self.b),x+self.b)

class MakePSD(nn.Module):
    def __init__(self, f, n, eps=0.01, d=1.0):
        super().__init__()
        self.f = f
        self.zero = torch.nn.Parameter(f(torch.zeros(1,n)), requires_grad=False)
        self.eps = eps
        self.d = d
        self.rehu = ReHU(self.d)

    def forward(self, x):
        smoothed_output = self.rehu(self.f(x) - self.zero)
        quadratic_under = self.eps*(x**2).sum(1,keepdim=True)
        return smoothed_output + quadratic_under

def configure(props):
    #logger.info(props)

    alpha = float(props["a"]) if "a" in props else 0.01

    # Do we consider an explicit time input
    explicit_time = int(props["explicit_time"])


    model = Dynamics(alpha=alpha, fhat_activation=nn.ReLU(), props=props)
    return model
