import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
class GumbelSoftmax(nn.Module):
    '''
    Returns differentiable discrete outputs. Applies a Gumbel-Softmax trick on every element of x.
    '''
    def __init__(self, eps=1e-8):
        super(GumbelSoftmax, self).__init__()
        self.eps = eps

    def forward(self, x, gumbel_temp=1.0, gumbel_noise=True):
        if not self.training:  # no Gumbel noise during inference
            return (x >= 0).float()

        logger.add('gumbel_noise', gumbel_noise)
        logger.add('gumbel_temp', gumbel_temp)

        if gumbel_noise:
            eps = self.eps
            U1, U2 = torch.rand_like(x), torch.rand_like(x)
            g1, g2 = -torch.log(-torch.log(U1 + eps)+eps), - \
                torch.log(-torch.log(U2 + eps)+eps)
            x = x + g1 - g2

        soft = torch.sigmoid(x / gumbel_temp)
        hard = ((soft >= 0.5).float() - soft).detach() + soft
        assert not torch.any(torch.isnan(hard))
        return hard
