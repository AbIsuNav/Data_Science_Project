import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F


class TransitionLayer(nn.Module):
    """
    Insert a transition layer, a global pooling layer, a prediction layer
    (and a loss layer in the end -- written in the loss function WCLE) (after the last convolutional layer in ResNet).
    input: output of last convolutional layer in ResNet (with a size of batch_size x D x S x S)
    output: weighted spatial activation maps for each disease class (with a size of batch_size x S × S × C)
    """
    def __init__(self, pool_mode, input_features, S, D, n_classes, r=0):
        super(TransitionLayer, self).__init__()
        self.S = S
        self.r = r
        self.pool_mode = pool_mode
        # After conv1 dim=(1, D, S, S)
        self.conv1 = nn.Conv2d(input_features, D, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(D)
        # After fc dim=(1, n_classes)
        self.fc = nn.Linear(D, n_classes)

    def forward(self, x, CAM=False, verbose=False):
        """
        The forward pass of the transition layer, for either classification or generation of heat-maps.
        :param x: the input image
        :param CAM: if true, the hat-maps will be produced
        :return: the generated output
        """
        '''
        Currently, the following two lines are commented as we don not know what actually the transition layer does
        with regards to the dimensions.
        x = self.conv1(x)
        x = self.bn(x)
        '''
        if not CAM:  # classification
            # After global pool dim=
            # x = self.global_pool(x, app=True)
            x = self.global_pool(x)
            if verbose:
                print(f'In [forward] of TransitionLayer: output of global_pool of shape: {x.shape}')

            out = torch.exp(self.fc(x))
            if verbose:
                print(f'In [forward] of TransitionLayer: output of the fc layer of shape: {out.shape}')

            out = out / out.sum(-1).view(-1, 1)
            if verbose:
                print(f'In [forward] of TransitionLayer: output of the normalization of shape: {out.shape}')
        else:  # class activation map -- for heatmap
            out = [torch.einsum(("ab, bcd ->acd"), (self.fc.weight.data, x[i])) for i in range(x.size(0))]

        return out

    def global_pool(self, x, app=False):
        """
        The pooling layer, currently could be either determined Log-Sum-Exp(LSE) pooling (mode='lse') or global max
        pooling (mode='max')
        :param x:
        :param mode:
        :param app:
        :return:
        """
        if self.pool_mode == 'lse':  # log-sum-exp pooling
            if app:
            # Eq(3) version, used in Xray paper
                x_star = torch.abs(x).max(dim=-1)[0].max(dim=-1)[0]
                x_p = x_star + 1/self.r*torch.log(1/self.S*torch.exp(self.r*(x-x_star.unsqueeze(-1).unsqueeze(-1))).sum(-1).sum(-1))
            else:
            # Eq(2)
                x_p = 1/self.r*torch.log(1/self.S*torch.exp(x*self.r).sum(-1).sum(-1))
            return x_p

        elif self.pool_mode == 'max':
            height, width = x.shape[2], x.shape[3]  # x of shape (B, C, H, W)
            return F.max_pool2d(x, kernel_size=(height, width)).squeeze(dim=3).squeeze(dim=2)  # (B, C, 1, 1) -> (B, C)


if __name__=='__main__':
    # shape of the output of last conv layer in ResNet
    S = 8
    D = 2048
    n_classes = 8
    input_features = D
    r = 10  # hyperparameter for global pooling layer in the transition layer/model
    batch_size = 3
	# test
    sample_input = torch.ones(size=(batch_size, D, S, S), requires_grad=False)
    transition = TransitionLayer(input_features, S, D, n_classes, r=r)
    out = transition(sample_input, CAM=False)
    for i in range(len(out)):
        for c in range(n_classes):
            plt.pcolormesh(out[i][c].detach().numpy())
            plt.title('class activation map{}{}'.format(i, c))
            plt.show()








