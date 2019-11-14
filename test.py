import networks
from networks import UnifiedNetwork
import helper

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


def test_load_models():
    model_name = "models/unified_net_step_1.pt"
    optimizer_name = 'models/optimizer_step_1.pt'

    transition_params = {
        'input_features': 512,
        'S': 7,  # spatial dimension of the resnet output (for our 224 x 224 images)
        'D': 512,  # the channel dimension of the resnet output
        'n_classes': 14,
        'pool_mode': 'max'
        # 'CAM': False,
        # 'r': 10
    }
    unified_net = helper.load_model(model_name, torch.device('cpu'), transition_params, 'resnet34')
    optimizer = helper.load_model(optimizer_name, torch.device('cpu'), unified_net.parameters())
    # print(unified_net)
    print(optimizer.parameters())


def main():
    test_resnet_get_last_layer()


if __name__ == '__main__':
    main()
