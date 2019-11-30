import json
import numpy as np
import torch
from torch.utils import data

import data_handler
import helper
import networks
from helper.ploting_fn import plot_ROC


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


def test_load_models_2():
    # model_name = "models/unified_net_step_9.pt"
    model_1 = 'models/max_epochs=30_batch_size=256_pool_mode=max_lr=0.0001_no_crop=False/unified_net_epoch_11.pt'
    model_2 = 'models/max_epochs=30_batch_size=256_pool_mode=max_lr=0.0001_no_crop=True/unified_net_epoch_8.pt'
    model_3 = 'models/max_epochs=30_batch_size=256_pool_mode=max_lr=5e-05_no_crop=False/unified_net_epoch_14.pt'
    model_4 = "models/max_epochs=30_batch_size=256_pool_mode=max_lr=5e-05_no_crop=True/unified_net_epoch_23.pt"
    models_trained = [model_1, model_2, model_3, model_4]
    # reading the other params from the JSON file
    with open('params.json', 'r') as f:
        params = json.load(f)
    for name in models_trained:
        helper.evaluate_model(model_path=name, params=params)


def test_read_data():
    h5_file = 'chest_xray.h5'
    # img_ids, labels, labels_hot = data_handler.read_ids_and_labels(h5_file)
    partition, labels, labels_hot = data_handler.read_already_partitioned(h5_file)

    print(len(partition['train']), len(partition['test']))
    print(partition['train'][:10])
    print(partition['validation'][:10])
    print(partition['test'][:10])

    '''for k, v in labels.items():
        print(k, v)
        print(type(k), type(v))

    for k, v in labels_hot.items():
        print(k, v)
        print(type(k), type(v))'''


def test_download_data():
    data_handler.download_data()


def main():
    # test_load_models_2()
    test_download_data()


if __name__ == '__main__':
    main()


"""
Test notes: if using early stopping (once it is fixed), the following could be used in the json file:
"es_params": {"patience": 5, "min_delta": 1e-3}
"""