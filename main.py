import data_handler
import networks
import helper

import json
import argparse

import torch
from torch.backends import cudnn
from torch.utils import data
from torchvision import transforms


def main():
    # Since not everybody uses comet, we determine if we should use comet_ml to track the experiment using the args
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_comet', type=bool, default=False)
    args = parser.parse_args()

    # reading the other params from the JSON file
    with open('params.json', 'r') as f:
        params = json.load(f)

    # initialize our comet experiment to track the run, if wanted by the user
    if args.use_comet:
        tracker = helper.init_comet(params)
        print("Comet experiment initialized...")

    # data files
    data_folder = params['data_folder']
    h5_file = params['h5_file']

    # training params
    batch_size = params['batch_size']
    shuffle = params['shuffle']
    num_workers = params['num_workers']
    max_epochs = params['max_epochs']

    # reading image ids and partition them into train, validation, and test sets
    partition, labels, labels_hot = \
        data_handler.read_and_partition_data(h5_file, limited=True, val_frac=0.2, test_frac=0.1)

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

    # creating the train data loader
    train_set = data_handler.Dataset(partition['train'], labels, labels_hot, data_folder, preprocess, device)
    train_loader = data.DataLoader(dataset=train_set, batch_size=batch_size,
                                   shuffle=shuffle, num_workers=num_workers)
    # creating the validation data loader
    val_set = data_handler.Dataset(partition['validation'], labels, labels_hot, data_folder, preprocess, device)
    val_loader = data.DataLoader(dataset=val_set, batch_size=batch_size,
                                 shuffle=False, num_workers=num_workers)

    # training
    which_resnet = params['which_resnet']
    transition_params = {
        'input_features': 512,
        'S': 7,  # spatial dimension of the resnet output (for our 224 x 224 images)
        'D': 512,  # the channel dimension of the resnet output
        'n_classes': 14,
        'pool_mode': params['pool_mode']
        # 'CAM': False,
        # 'r': 10
    }
    # resnet = networks.load_resnet(which_resnet).to(device)
    unified_net = networks.UnifiedNetwork(transition_params, which_resnet).to(device)

    # Adam optimizer with default parameters
    optimizer = torch.optim.Adam(unified_net.parameters())

    # for epoch in range(max_epochs):
    optim_step = 0
    validation_interval = 10  # track validation after such a number of steps
    while optim_step < params['max_optim_steps']:
        for i_batch, batch in enumerate(train_loader):
            print(f'=========== In step: {optim_step}')
            # img_batch, label_batch = img_batch.to(device), label_batch.to(device)
            img_batch, label_batch = batch['image'].to(device), batch['label'].to(device)
            '''print(f'img_batch size: {img_batch.size()}, '
                  f'labels_batch size: {label_batch.size()}')'''

            # making gradients zero in each optimization step
            optimizer.zero_grad()

            # getting thr network prediction and computing the loss
            pred = unified_net(img_batch, verbose=False)
            train_loss = networks.WCEL(pred, label_batch)
            print(f'train loss: {round(train_loss.item(), 3)}')  # round to three floating points

            # computing the validation loss
            val_loss = helper.compute_val_loss(unified_net, val_loader, device)
            if optim_step % validation_interval == 0:
                print(f'validation loss: {val_loss}')

            # tracking the metrics using comet in each iteration
            if args.use_comet:
                tracker.track_metric('train_loss', round(train_loss.item(), 3), optim_step)
                tracker.track_metric('val_loss', val_loss, optim_step)

            # backward and optimization step
            train_loss.backward()
            optimizer.step()
            optim_step += 1


if __name__ == '__main__':
    main()
