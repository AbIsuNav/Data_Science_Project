import torch
import torch.nn as nn


def WCEL(fx, label, att2=False):
    """
    Multi-label Classification Loss Layer: Weighted Cross Entropy Loss, Eq(1)
    :param fx: the output (prediction) of the unified network, tensor of shape (B, 14)
    :param label: the true labels, tensor of shape (B, 14)
    :return:
    """
    # converting the labels batch  to from Long tensor to Float tensor (otherwise won't work on GPU)
    label = label.float()

    P = label.sum()
    N = label.size(0) * label.size(1) - P
    betaP = (P + N) / (P + 1e-5)  # avoid zero in denominator
    betaN = (P + N) / (N + 1e-5)
    y0 = torch.abs(label - 1)
    if att2:
        loss = (-betaP * torch.log(fx+1e-6) * label).sum() - (betaN * torch.log(1 - fx+1e-6) * y0).sum()
    else:
        loss = (-betaP * torch.log(fx) * label).sum() - (betaN * torch.log(1 - fx) * y0).sum()

    batch_size = fx.shape[0]
    loss_avg = loss / batch_size

    # return loss
    return loss_avg


if __name__ == '__main__':
    S = 8
    D = 2048
    n_classes = 8
    sample_input = torch.ones(size=(32, 8), requires_grad=False)
    sample_label = torch.ones(size=(32, 8), requires_grad=False)
    loss = WCEL(sample_input, sample_label)
