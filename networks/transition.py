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
    def __init__(self, include_1x1_conv, pool_mode, input_features, S, D, n_classes, r=0.1, upsample_conv=False):
        super(TransitionLayer, self).__init__()
        self.include_1x1_conv = include_1x1_conv
        self.pool_mode = pool_mode
        self.S = S
        self.r = r
        self.upsample_conv = upsample_conv

        # this 1x1 does not change either the depth or the spatial dimension of the input
        if include_1x1_conv:
            self.conv1 = nn.Conv2d(input_features, D, kernel_size=1)
            # self.conv1 = nn.Conv2d(input_features, D, kernel_size=3, stride=1, padding=1, bias=False)
            # self.bn = nn.BatchNorm2d(D)
        elif upsample_conv:
            self.relu = nn.ReLU(inplace=True)
            self.conv1 = nn.ConvTranspose2d(input_features, D, 4, 2, 1, bias=False)
            self.bn1 = nn.BatchNorm2d(D)
            self.conv2 = nn.ConvTranspose2d(D, D, 4, 2, 1, bias=False)
            self.bn2 = nn.BatchNorm2d(D)

        # After fc dim=(1, n_classes)
        self.fc = nn.Linear(D, n_classes)
        self.sig = nn.Sigmoid()
        print(f'In [TransitionLayer]: constructor is built with '
              f'pooling mode: "{self.pool_mode}", '
              f'include_1x1_conv: {self.include_1x1_conv}, '
              f'S: {self.S} \n')

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
        if self.include_1x1_conv:
            x = self.conv1(x)
        elif self.upsample_conv:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.conv2(x)
            x = self.bn2(x)
            x = self.relu(x)

        if not CAM:  # classification
            # After global pool dim=
            # x = self.global_pool(x, app=True)

            x = self.global_pool(x)
            if verbose:
                print(f'In [forward] of TransitionLayer: output of global_pool of shape: {x.shape}')

            out = self.fc(x)
            # out = torch.exp(self.fc(x))
            if verbose:
                print(f'In [forward] of TransitionLayer: output of the fc layer of shape: {out.shape}')

            out = self.sig(out)
            # out = out / out.sum(-1).view(-1, 1)
            if verbose:
                print(f'In [forward] of TransitionLayer: output of the normalization of shape: {out.shape}')

        else:  # class activation map -- for heatmap
            out = [torch.einsum(("ab, bcd ->acd"), (self.fc.weight.data, x[i])).unsqueeze(0) for i in range(x.size(0))]
            # make it a tensor
            out = torch.cat(out)
            # need normalization(done in the plot_heatmap)
        return out

    def global_pool(self, x, app=True):
        """
        The pooling layer, currently could be either determined Log-Sum-Exp(LSE) pooling (mode='lse') or global max
        pooling (mode='max')
        :param x: all the feature maps, a tensor of shape (B, C, S, S), where B is the batch size, C is the channel, and
        S is the spatial dimension.
        :param app: if True, the approximated version of LSE is used.
        :return: the pooled tensor of shape (B, C)

        Note: if pool_mode is 'max_avg', both max pooling and avg pooling are performed independently and the average
        of those is returned.
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

        elif self.pool_mode == 'avg':
            height, width = x.shape[2], x.shape[3]  # x of shape (B, C, H, W)
            return F.avg_pool2d(x, kernel_size=(height, width)).squeeze(dim=3).squeeze(dim=2)  # (B, C, 1, 1) -> (B, C)

        elif self.pool_mode == 'max_avg':
            height, width = x.shape[2], x.shape[3]  # x of shape (B, C, H, W)

            # squeeze operation: (B, C, 1, 1) -> (B, C)
            max_val = F.max_pool2d(x, kernel_size=(height, width)).squeeze(dim=3).squeeze(dim=2)
            avg_val = F.avg_pool2d(x, kernel_size=(height, width)).squeeze(dim=3).squeeze(dim=2)
            return (max_val + avg_val) / 2

        else:  # is it the best Exception to be thrown?
            raise ValueError('In [global_pool]: pooling mode not implemented!')


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
    transition = TransitionLayer('lse', input_features, S, D, n_classes, r=r)
    out = transition(sample_input, CAM=True)
    for i in range(len(out)):
        for c in range(n_classes):
            plt.pcolormesh(out[i][c].detach().numpy())
            plt.title('class activation map{}{}'.format(i, c))
            plt.show()








