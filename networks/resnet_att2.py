import torch
import torch.nn as nn
from torchvision import models

from . import resnet

"""
Pre-trained ResNet50
"""


class ResNet_A2(nn.Module):
    """
    Loads the wanted pre-trained ResNet, creates a transition layer, and then connects the loaded ResNet to the
    TransitionLayer which contains the pooling and prediction layers in itself.
    """

    def __init__(self, which_resnet):
        """
        :param transition_params: a dictionary containing the parameters needed to initialize a transition layer.
        :param which_resnet: a string such as 'resnet34' indicating the resnet to be loaded.
        """
        super().__init__()
        self.inplanes = 14
        self.resnet = resnet.load_resnet(train_params=True)
        self.pool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.resize = nn.Upsample(scale_factor=4)
        self.conv1 = nn.Conv2d(1024, self.inplanes, kernel_size=1)
        #self.norm = nn.BatchNorm2d(self.inplanes)
        #self.relu = nn.LeakyReLU()
        self.sig = nn.Sigmoid()

    def forward(self, input_img, get_heatmap=False, verbose=False):
        """
        The forward pass of the network, plugging in the output of the resnet to the transition layer
        :param verbose:
        :param input_img: the input image of shape (C, H, W), denoting channel, height, and width (excluding batch size)
        :return: the prediction (or the heat-map) on the image
        """
        # before = time.time()
        x1 = self.resnet(input_img)
        x2 = self.pool(x1)
        x2 = self.resize(x2)
        x = torch.cat((x2, x1), 1)
        x = self.conv1(x)
        # end attention

        if verbose:
            print('In [forward] of Attention2_Network: input batch size:', input_img.size())
            print('Heatmap size: ', x.size())
        if get_heatmap:
            return x
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.sig(x)
        # print(f'In [UnifiedNetwork].[forward]: the forward of resnet took {time.time() - before}')
        if verbose:
            print('In [forward] of Attention2_Network: resnet output size:', x1.size())
            print('In [forward] of Attention2_Network: prediction output size:', x.size())
            # print('In [forward] of UnifiedNetwork: prediction for the first 5 images:')
            # print(pred[:5])
        return x


class MyResnet2(nn.Module):
    def __init__(self, resnet):
        super(MyResnet2, self).__init__()
        channels = 64
        #self.inplanes = 14
        self.resnet = resnet
        self.se1 = SELayer(channels)
        self.se2 = SELayer(channels*2)
        self.down2 = Downsample(channels, channels*2)
        self.se3 = SELayer(channels*4)
        self.down3 = Downsample(channels*2, channels * 4)
        self.se4 = SELayer(channels*8)
        self.down4 = Downsample(channels*4, channels * 8)
        self.pool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.resize = nn.Upsample(scale_factor=4)
        self.conv_att = nn.Conv2d(1024, 14, kernel_size=1)
        # self.norm = nn.BatchNorm2d(self.inplanes)
        # self.relu = nn.LeakyReLU()
        self.sig = nn.Sigmoid()



    def forward(self, x, get_heatmap=False,verbose=False):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        residual = x
        x = self.resnet.layer1(x)
        x = self.se1(x)
        x += residual
        residual = nn.functional.interpolate(x, scale_factor=0.5)
        residual = self.down2(residual) #64
        x = self.resnet.layer2(x)
        x = self.se2(x)  # 128
        x += residual
        residual = self.down3(x)
        residual = nn.functional.interpolate(residual, scale_factor=0.5)
        x = self.resnet.layer3(x)
        x = self.se3(x)
        x += residual
        residual = self.down4(x)
        residual = nn.functional.interpolate(residual, scale_factor=0.5)
        x = self.resnet.layer4(x)
        x = self.se4(x)
        x += residual
        x2 = self.pool(x)
        x2 = self.resize(x2)
        x = torch.cat((x2, x), 1)
        x = self.conv_att(x)
        if get_heatmap:
            return x
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.sig(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Downsample, self).__init__()
        self.down = nn.Sequential(nn.Conv2d(in_channel, out_channel, 1),
                                  nn.BatchNorm2d(out_channel))
        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.down(x)
        return x


class SELayer(nn.Module):
    def __init__(self, in_channel):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel, in_channel, 1),
            nn.Sigmoid()
        )
        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x)
        y = self.fc(y)
        return x * y


if __name__ == "__main__":
    param_set = {
        "include_1x1_conv": True,
        "input_features": 512,
        "S": 7,
        "D": 512,
        "n_classes": 14,

        "pool_mode": "max",
        "r": 5
    }

    model = MyResnet2(models.resnet34(pretrained=True))
    test_batch = torch.rand((4, 3, 256, 256))
    output = model(test_batch)

    #model = ResNet_A2("resnet34")
    #model(test_batch)
