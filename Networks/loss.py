import torch
import torch.nn as nn
import math


def WCEL(fx, label, n_classes=8):
	"""
	Multi-label Classification Loss Layer: Weighted Cross Entropy Loss
	Eq(1)
	"""
    P = label.sum()
    N = label.size(0) * label.size(1) - P
    betaP = (P + N)/(P + 1e-5) # avoid zero in denominator
    betaN = (P + N)/(N + 1e-5) 
    y0 = torch.abs(label - 1)
    loss = (-betaP * torch.log(fx) * label).sum() - (betaN * torch.log(1 - fx) * y0).sum()

    return loss

if __name__=='__main__':
    S = 8
    D = 2048
    n_classes = 8
    sample_input = torch.ones(size=(32, 8), requires_grad=False)
    sample_label = torch.ones(size=(32, 8), requires_grad=False)
    loss = WCEL(sample_input, sample_label)
