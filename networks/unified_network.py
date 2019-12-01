from . import resnet, transition

import torch
import torch.nn

import time


class UnifiedNetwork(torch.nn.Module):
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
        self.resnet = resnet.load_resnet(train_params=True)  # train the resnet as well
        self.trans_pool_prediction = transition.TransitionLayer(**transition_params)

    def forward(self, input_img, verbose=False, CAM=False):
        """
        The forward pass of the network, plugging in the output of the resnet to the transition layer
        :param verbose:
        :param input_img: the input image of shape (C, H, W), denoting channel, height, and width (excluding batch size)
        :return: the prediction (or the heat-map) on the image
        """
        # before = time.time()
        resnet_out = self.resnet(input_img)
        # print(f'In [UnifiedNetwork].[forward]: the forward of resnet took {time.time() - before}')
        if verbose:
            print('In [forward] of UnifiedNetwork: input batch size:', input_img.size())
            print('In [forward] of UnifiedNetwork: resnet output size:', resnet_out.size())

        pred = self.trans_pool_prediction(resnet_out, CAM=CAM)
        if verbose:
            print('In [forward] of UnifiedNetwork: trans_pool_prediction output size:', pred.size())
            # print('In [forward] of UnifiedNetwork: prediction for the first 5 images:')
            # print(pred[:5])
        return pred


if __name__ == '__main__':
    param_set = {
        'input_features': 512,
        'S': 8,
        'D': 512,
        'n_classes': 14,
        # 'CAM': False,
        'r': 10
    }
    test_batch = torch.rand((10, 3, 224, 224))

    resnet34 = resnet.load_resnet()
    trans_pool_prediction = transition.TransitionLayer(**param_set)

    res_out = resnet34(test_batch)
    print(resnet34)

    print(dict(resnet34.named_children()).keys())

    print('layer 4:')
    layer4 = dict(resnet34.named_children())['layer4']
    print(layer4)
    input()
    for module in resnet34.modules():
        print(module.name)

    # list(dict(features.named_children()).keys()).index('conv_2')
    # print(resnet34.features[-3])

    # print('resnet out:', res_out.size())

    out = trans_pool_prediction(res_out)
