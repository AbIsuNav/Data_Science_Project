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


def test_load_models():
    model_name = "models/unified_net_epoch_25.pt"
    # optimizer_name = 'models/optimizer_step_1.pt'
    with open('params.json', 'r') as f:
        params = json.load(f)
    transition_params = {
        'input_features': 512,
        'S': 7,  # spatial dimension of the resnet output (for our 224 x 224 images)
        'D': 512,  # the channel dimension of the resnet output
        'n_classes': 14,
        'pool_mode': 'max'
        # 'CAM': False,
        # 'r': 10
    }
    # data files
    data_folder = params['data_folder']
    h5_file = params['h5_file']
    pathologies = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion', 'Emphysema', 'Fibrosis',
                   'Hernia',
                   'Infiltration', 'Mass', 'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax']
    # training params
    batch_size = params['batch_size']
    shuffle = params['shuffle']
    num_workers = params['num_workers']
    # not sure why such preprocessing is needed (taken from taken from https://pytorch.org/hub/pytorch_vision_resnet/)
    preprocess = helper.preprocess_fn()
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    partition, labels, labels_hot = \
        data_handler.read_and_partition_data(h5_file, val_frac=0.2, test_frac=0.1)
    data_set = data_handler.Dataset(partition['test'], labels, labels_hot, data_folder, preprocess, device)
    data_loader = data.DataLoader(dataset=data_set, batch_size=batch_size,
                                   shuffle=shuffle, num_workers=num_workers)
    # creating the validation data loader
    # val_set = data_handler.Dataset(partition['validation'], labels, labels_hot, data_folder, preprocess, device)
    # val_loader = data.DataLoader(dataset=val_set, batch_size=batch_size,
    #                             shuffle=False, num_workers=num_workers)
    total_predicted = np.zeros((batch_size, 14))
    total_labels = np.zeros((batch_size, 14))
    for i_batch, batch in enumerate(data_loader):
        img_batch = batch['image'].to(device).float()
        label_batch = batch['label'].to(device).float()
        net = helper.load_model(model_name, torch.device('cpu'), transition_params, 'resnet34')
        pred = net(img_batch, verbose=False)
        if i_batch > 0:
            total_predicted = np.append(total_predicted, pred.detach().numpy(), axis=0)
            total_labels = np.append(total_labels, label_batch.detach().numpy(), axis=0)
        else:
            total_predicted = pred.detach().numpy()
            total_labels = label_batch.detach().numpy()
    plot_ROC(total_predicted, total_labels, pathologies, save=True)


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


def main():
    test_load_models()
    #test_read_data()


if __name__ == '__main__':
    main()


"""
Test notes: if using early stopping (once it is fixed), the following could be used in the json file:
"es_params": {"patience": 5, "min_delta": 1e-3}
"""