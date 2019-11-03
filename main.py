import data_handler
# from networks import unified_network as un
import networks

import torch
import json
from torch.backends import cudnn
from torch.utils import data
from torchvision import transforms


def main():
    with open('params.json', 'r') as f:
        params = json.load(f)

    # data files
    data_folder = params['data_folder']
    h5_file = params['h5_file']

    # training params
    batch_size = params['batch_size']
    shuffle = params['shuffle']
    num_workers = params['num_workers']
    max_epochs = params['max_epochs']

    # reading image ids and partition them into train, validation, and test sets
    partition, labels, labels_hot = data_handler.read_and_partition_data(h5_file, val_frac=0.2, test_frac=0.1)

    # image preprocessing functions (not sure why, taken from https://pytorch.org/hub/pytorch_vision_resnet/)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # cuda for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    cudnn.benchmark = True  # this finds the optimum algorithm for the hardware if possible by kind of auto-tuning

    # creating the data loaders
    train_set = data_handler.Dataset(partition['train'], labels, labels_hot, data_folder, preprocess, device)
    train_loader = data.DataLoader(dataset=train_set, batch_size=batch_size,
                                   shuffle=shuffle, num_workers=num_workers)

    validation_set = data_handler.Dataset(partition['validation'], labels, labels_hot, data_folder, preprocess, device)
    validation_loader = data.DataLoader(dataset=validation_set, batch_size=batch_size,
                                        shuffle=shuffle, num_workers=num_workers)

    # training
    which_resnet = params['which_resnet']
    param_set = {
        'input_features': 512,
        'S': 7,
        'D': 512,
        'n_classes': 14
        # 'CAM': False,
        # 'r': 10
    }
    # resnet = networks.load_resnet(which_resnet).to(device)
    unified_net = networks.UnifiedNetwork(param_set, which_resnet).to(device)

    for epoch in range(max_epochs):
        for img_batch, label_batch in train_loader:
            img_batch, label_batch = img_batch.to(device), label_batch.to(device)
            print('batch read:', img_batch.size(), label_batch.size())
            # pred = unified_net(img_batch)
            # print('pred size:', pred.size())
            # print('resnet output without softmax and any other layer:', resnet(img_batch))
            # print(img_batch.size(), label_batch.size())


if __name__ == '__main__':
    main()
