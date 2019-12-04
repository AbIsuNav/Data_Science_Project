#from . import resnet
import torch
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import cv2 as cv
from torchvision import models
from torch.nn import functional as F



def weights_init_kaiming(m):
    classname = m.__class__.__name__
    #print(classname)
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)



class AttentionGate(nn.Module):
        def __init__(self, in_channels, gating_channels, inter_channels=None, mode='concatenation',
                     sub_sample_factor=(2, 2, 2)):
            super(AttentionGate, self).__init__()

            assert mode in ['concatenation', 'concatenation_debug', 'concatenation_residual']

            # Downsampling rate for the input featuremap
            if isinstance(sub_sample_factor, tuple):
                self.sub_sample_factor = sub_sample_factor
            elif isinstance(sub_sample_factor, list):
                self.sub_sample_factor = tuple(sub_sample_factor)
            else:
                self.sub_sample_factor = tuple([sub_sample_factor]) * 2

            # Default parameter set
            self.mode = mode
            self.sub_sample_kernel_size = self.sub_sample_factor

            # Number of channels (pixel dimensions)
            self.in_channels = in_channels
            self.gating_channels = gating_channels
            self.inter_channels = inter_channels

            if self.inter_channels is None:
                self.inter_channels = in_channels // 2
                if self.inter_channels == 0:
                    self.inter_channels = 1

            # upsample dimension=2
            conv_nd = nn.Conv2d
            bn = nn.BatchNorm2d
            self.upsample_mode = 'bilinear'

            # Output transform
            self.W = nn.Sequential(
                conv_nd(in_channels=self.in_channels, out_channels=self.in_channels, kernel_size=1, stride=1,
                        padding=0),
                bn(self.in_channels),
            )

            # Theta^T * x_ij + Phi^T * gating_signal + bias
            self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                                 kernel_size=self.sub_sample_kernel_size, stride=self.sub_sample_factor, padding=0,
                                 bias=False)
            self.phi = conv_nd(in_channels=self.gating_channels, out_channels=self.inter_channels,
                               kernel_size=1, stride=1, padding=0, bias=True)
            self.psi = conv_nd(in_channels=self.inter_channels, out_channels=1, kernel_size=1, stride=1, padding=0,
                               bias=True)

            # Initialise weights
            for m in self.children():
                m.apply(weights_init_kaiming)

            # Define the operation
            if mode == 'concatenation':
                self.operation_function = self._concatenation
            elif mode == 'concatenation_debug':
                self.operation_function = self._concatenation_debug
            elif mode == 'concatenation_residual':
                self.operation_function = self._concatenation_residual
            else:
                raise NotImplementedError('Unknown operation function.')

        def forward(self, x, g):
            '''
            :param x: (b, c, t, h, w)
            :param g: (b, g_d)
            :return:
            '''

            output = self.operation_function(x, g)
            return output

        def _concatenation(self, x, g):
            input_size = x.size()
            batch_size = input_size[0]
            assert batch_size == g.size(0)

            # theta => (b, c, t, h, w) -> (b, i_c, t, h, w) -> (b, i_c, thw)
            # phi   => (b, g_d) -> (b, i_c)
            theta_x = self.theta(x)
            theta_x_size = theta_x.size()

            # g (b, c, t', h', w') -> phi_g (b, i_c, t', h', w')
            #  Relu(theta_x + phi_g + bias) -> f = (b, i_c, thw) -> (b, i_c, t/s1, h/s2, w/s3)
            phi_g = F.upsample(self.phi(g), size=theta_x_size[2:], mode=self.upsample_mode)
            f = F.relu(theta_x + phi_g, inplace=True)

            #  psi^T * f -> (b, psi_i_c, t/s1, h/s2, w/s3)
            sigm_psi_f = F.sigmoid(self.psi(f))

            # upsample the attentions and multiply
            sigm_psi_f = F.upsample(sigm_psi_f, size=input_size[2:], mode=self.upsample_mode)
            y = sigm_psi_f.expand_as(x) * x
            W_y = self.W(y)

            return W_y, sigm_psi_f

        def _concatenation_debug(self, x, g):
            input_size = x.size()
            batch_size = input_size[0]
            assert batch_size == g.size(0)

            # theta => (b, c, t, h, w) -> (b, i_c, t, h, w) -> (b, i_c, thw)
            # phi   => (b, g_d) -> (b, i_c)
            theta_x = self.theta(x)
            theta_x_size = theta_x.size()

            # g (b, c, t', h', w') -> phi_g (b, i_c, t', h', w')
            #  Relu(theta_x + phi_g + bias) -> f = (b, i_c, thw) -> (b, i_c, t/s1, h/s2, w/s3)
            phi_g = F.upsample(self.phi(g), size=theta_x_size[2:], mode=self.upsample_mode)
            f = F.softplus(theta_x + phi_g)

            #  psi^T * f -> (b, psi_i_c, t/s1, h/s2, w/s3)
            sigm_psi_f = F.sigmoid(self.psi(f))

            # upsample the attentions and multiply
            sigm_psi_f = F.upsample(sigm_psi_f, size=input_size[2:], mode=self.upsample_mode)
            y = sigm_psi_f.expand_as(x) * x
            W_y = self.W(y)

            return W_y, sigm_psi_f

        def _concatenation_residual(self, x, g):
            input_size = x.size()
            batch_size = input_size[0]
            assert batch_size == g.size(0)

            # theta => (b, c, t, h, w) -> (b, i_c, t, h, w) -> (b, i_c, thw)
            # phi   => (b, g_d) -> (b, i_c)
            theta_x = self.theta(x)
            theta_x_size = theta_x.size()

            # g (b, c, t', h', w') -> phi_g (b, i_c, t', h', w')
            #  Relu(theta_x + phi_g + bias) -> f = (b, i_c, thw) -> (b, i_c, t/s1, h/s2, w/s3)
            phi_g = F.upsample(self.phi(g), size=theta_x_size[2:], mode=self.upsample_mode)
            f = F.relu(theta_x + phi_g, inplace=True)

            #  psi^T * f -> (b, psi_i_c, t/s1, h/s2, w/s3)
            f = self.psi(f).view(batch_size, 1, -1)
            sigm_psi_f = F.softmax(f, dim=2).view(batch_size, 1, *theta_x.size()[2:])

            # upsample the attentions and multiply
            sigm_psi_f = F.upsample(sigm_psi_f, size=input_size[2:], mode=self.upsample_mode)
            y = sigm_psi_f.expand_as(x) * x
            W_y = self.W(y)

            return W_y, sigm_psi_f


class ResNet_AG(nn.Module):

    def __init__(self, n_classes=14, freeze=False):
        """
        :param transition_params: a dictionary containing the parameters needed to initialize a transition layer.
        :param
        """
        super().__init__()
        self.resnet = models.resnet34(pretrained=True)
        #if freeze:
            #freeze weight
        self.channels = [64, 128, 256, 512]
        self.ag1 = AttentionGate(self.channels[1], gating_channels=self.channels[3])
        self.ag2 = AttentionGate(self.channels[2], gating_channels=self.channels[3])

        self.classifier = nn.Linear(n_classes * 3, n_classes)
        self.aggregate = self.aggregation_ft

    def aggregation_ft(self, *attended_maps):
        preds = self.aggregation_sep(*attended_maps)
        return self.classifier(torch.cat(preds, dim=1))

    def aggregation_sep(self, *attended_maps):
        return [clf(att) for clf, att in zip(self.classifiers, attended_maps)]

    def forward(self, input, CAM=False, verbose=False):
        """
        The forward pass of the network, plugging in the output of the resnet to the transition layer
        :param verbose:
        :param input: the input image of shape (C, H, W), denoting channel, height, and width (excluding batch size)
        :return: the prediction (or the heat-map) on the image
        """
        # resnet -- feature extraction
        x = self.resnet.conv1(input)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x1 = self.resnet.layer1(x)
        x2 = self.resnet.layer2(x1)
        x3 = self.resnet.layer3(x2)
        x4 = self.resnet.layer4(x3)

        pooled = self.resnet.avgpool(x4)
        pooled = torch.flatten(pooled, 1)

        batch_size = input.shape[0]

        # Attention Mechanism
        g1, att1 = self.ag1(x2, x4)
        g2, att2 = self.ag2(x3, x4)

        # flatten to get single feature vector
        g1 = torch.sum(g1.view(batch_size, self.channels[2], -1), dim=-1)
        g2 = torch.sum(g2.view(batch_size, self.channels[3], -1), dim=-1)

        # aggregation
        out = self.aggregate(g1, g2, pooled)

        if CAM:
            return att1, att2

        '''
        if verbose:
            print('In [forward] of Attention2_Network: resnet output size:', x1.size())
            print('In [forward] of Attention2_Network: prediction output size:', x.size())
            # print('In [forward] of UnifiedNetwork: prediction for the first 5 images:')
            # print(pred[:5])
        '''
        return out


if __name__ == "__main__":
    param_set = {
        "include_1x1_conv": True,
        "input_features": 512,
        "S": 8,
        "D": 512,
        "n_classes": 14,
        "pool_mode": "max",
        "r": 5
    }
    model = ResNet_AG()
    test_batch = torch.rand((4, 3, 256, 256))
    #from torch.autograd import Variable
    output = model(test_batch)

