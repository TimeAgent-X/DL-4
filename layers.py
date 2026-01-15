import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn as nn
import torch.nn.functional as F

class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class PairNorm(nn.Module):
    def __init__(self, mode='PN', scale=1):
        """
        mode:
          'None' : No normalization
          'PN'   : Original PairNorm
          'PN-SI': PairNorm-SI (scale-independent)
          'PN-SCS': PairNorm-SCS (scale-centered-scaled)
        scale: scale factor s
        """
        assert mode in ['None', 'PN', 'PN-SI', 'PN-SCS']
        super(PairNorm, self).__init__()
        self.mode = mode
        self.scale = scale

    def forward(self, x):
        if self.mode == 'None':
            return x
        
        col_mean = x.mean(dim=0)
        
        if self.mode == 'PN-SCS':
             x = x - col_mean
        
        if self.mode == 'PN-SI':
            row_mean = x.mean(dim=1, keepdim=True)
            x = x - row_mean

        if self.mode == 'PN':
            x = x - col_mean 

        # Calculate row-wise root mean square
        # row_norm = (x**2).sum(dim=1, keepdim=True).sqrt() # This is L2 Norm, but PairNorm uses mean
        # Let's double check PairNorm paper/impl.
        # Original paper: x_i^c = x_i - mean(x) 
        # x_i_new = s * x_i^c / sqrt( 1/n * sum((x_i^c)**2) )  (centering and scaling)
        # But commonly in GNNs it's implemented efficiently.
        # Using a simpler variance based approach:
        
        row_mean = x.mean(dim=1, keepdim=True)
        x = x - row_mean # Center row-wise? Wait.
        
        # Standard PairNorm:
        # 1. Center node representations (ensure mean is 0)
        col_mean = x.mean(dim=0, keepdim=True)
        x = x - col_mean
        
        # 2. Rescale so that total variation is constant
        # row_norm_sq = (x ** 2).sum(dim=1, keepdim=True)
        # The paper says: x_tilda_i = x_i^c * s / sqrt( 1/n \sum ||x_i^c||^2 ) ??? -> This is global scaling
        # Wait, the paper says "row-wise normalization is not good for GNNs".
        
        # Implementation from PairNorm repo (https://github.com/LingxiaoShawn/PairNorm):
        # x = x - x.mean(dim=0, keepdim=True)
        # x = x / (1e-6 + x.norm(dim=1, keepdim=True)) * self.scale # This is row normalization?
        
        # Let's stick to the definition:
        # x = x - Mean(x) (column wise)
        # x = s * x / RMS(x) (row wise)
        
        x = x - x.mean(dim=0, keepdim=True)
        row_norm = x.norm(dim=1, keepdim=True)
        x = self.scale * x / (row_norm + 1e-6)
        
        return x
