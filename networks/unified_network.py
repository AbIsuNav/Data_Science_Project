from . import resnet, transition
# import resnet
# import transition
import torch
import torch.nn


class UnifiedNetwork(torch.nn.Module):
    def __init__(self, params_set, which_resnet):
        super().__init__()
        self.resnet = resnet.load_resnet(which_resnet)
        self.trans_pool_prediction = transition.TransitionLayer(**params_set)

    def forward(self, input_img):
        resnet_out = self.resnet(input_img)
        print('resnet output:', resnet_out.size())

        pred = self.trans_pool_prediction(resnet_out)
        print(pred.size())
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

    resnet34 = resnet.load_resnet('resnet34')
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
