import resnet
import torch
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import cv2 as cv

"""
Pre-trained ResNet50
"""


class ResNet_A2(nn.Module):
    """
    Loads the wanted pre-trained ResNet, creates a transition layer, and then connects the loaded ResNet to the
    TransitionLayer which contains the pooling and prediction layers in itself.
    """

    def __init__(self, transition_params, which_resnet):
        """
        :param transition_params: a dictionary containing the parameters needed to initialize a transition layer.
        :param which_resnet: a string such as 'resnet34' indicating the resnet to be loaded.
        """
        super().__init__()
        self.inplanes = 14
        self.resnet = resnet.load_resnet(train_params=True)
        self.pool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.resize = nn.Upsample(scale_factor=7)
        self.conv1 = nn.Conv2d(1024, self.inplanes, kernel_size=7, stride=1, padding=3,
                               bias=False)


    def forward(self, input_img, verbose=False):
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
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        # print(f'In [UnifiedNetwork].[forward]: the forward of resnet took {time.time() - before}')
        if verbose:
            print('In [forward] of UnifiedNetwork: input batch size:', input_img.size())
            print('In [forward] of UnifiedNetwork: resnet output size:', x1.size())

        if verbose:
            print('In [forward] of UnifiedNetwork: trans_pool_prediction output size:', x.size())
            # print('In [forward] of UnifiedNetwork: prediction for the first 5 images:')
            # print(pred[:5])
        return x


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
    test_batch = torch.rand((10, 3, 224, 224))

    model = ResNet_A2(param_set,"resnet34")
    model(test_batch)