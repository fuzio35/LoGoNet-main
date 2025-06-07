import torch
import torch.nn as nn

class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx,input,weight,bias,eps):
        ctx.eps = eps
        N,C,X,Y,Z = input.shape
        mu = input.mean(1,keepdim=True)
        