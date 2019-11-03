import networks
from networks import UnifiedNetwork

import torch
from torchsummary import summary
from torchvision import models


def test_resnet_get_last_layer():
    resnet = networks.load_resnet(verbose=True)
    print(resnet(torch.rand(10, 3, 224, 224)))


def test_unified_net():
    # unified_net = UnifiedNetwork()

    # the output of our resnet is now of shape [512, 7, 7] (excluding batch size)
    d = 8
    input_features = 512
    s = 7
    n_classes = 14


def main():
    test_resnet_get_last_layer()


if __name__ == '__main__':
    main()
