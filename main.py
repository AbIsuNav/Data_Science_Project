from comet_ml import Experiment  # do not touch this, comet_ml should be imported before other modules such as torch

import data_handler
import networks
import helper

import json
import argparse
import os

import torch
from torch.backends import cudnn
from torch.utils import data


def main():
    # Since not everybody uses comet, we determine if we should use comet_ml to track the experiment using the args
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_comet', action='store_true')
    parser.add_argument('--save_checkpoints', action='store_true')
    parser.add_argument('--data_limited', action='store_true')
    args = parser.parse_args()

    print(f'Running the program with arguments use_comet: {args.use_comet}, '
          f'save_checkpoints: {args.save_checkpoints}, '
          f'data_limited: {args.data_limited}')

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

    which_resnet = params['which_resnet']
    transition_params = params['transition_params']  # if the pool mode is 'max' or 'avg', the r value is imply ignored
    print('Transition params:', transition_params)
    print('Note: "r" will simply be ignored if the pool mode is "max" or "avg"', '\n')

    # reading image ids and partition them into train, validation, and test sets
    partition, labels, labels_hot = \
        data_handler.read_and_partition_data(h5_file, limited=args.data_limited, val_frac=0.2, test_frac=0.1)

    # not sure why such preprocessing is needed (taken from taken from https://pytorch.org/hub/pytorch_vision_resnet/)
    preprocess = helper.preprocess_fn()

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

    unified_net = networks.UnifiedNetwork(transition_params, which_resnet).to(device)

    # Adam optimizer with default parameters
    optimizer = torch.optim.Adam(unified_net.parameters())

    # for epoch in range(max_epochs):
    epoch = 0
    save_model_interval = 1  # save model checkpoints at such a number of epochs
    # validation_interval = 10  # print validation loss after such a number of epochs

    while epoch < max_epochs:
        print(f'=========== In epoch: {epoch}')

        for i_batch, batch in enumerate(train_loader):
            print(f'Performing on batch: {i_batch}')
            # img_batch, label_batch = img_batch.to(device), label_batch.to(device)
            img_batch = batch['image'].to(device).float()
            label_batch = batch['label'].to(device).float()
            # converted the labels batch  to from Long tensor to Float tensor (otherwise won't work on GPU)

            # making gradients zero in each optimization step
            optimizer.zero_grad()

            # getting thr network prediction and computing the loss
            pred = unified_net(img_batch, verbose=False)

            train_loss = networks.WCEL(pred, label_batch)
            print(f'train loss: {round(train_loss.item(), 3)}')

            # computing the validation loss
            # val_loss = helper.compute_val_loss(unified_net, val_loader, device)
            # print(f'train loss: {round(train_loss.item(), 3)}, validation loss: {val_loss}')

            # if optim_step % validation_interval == 0:
            #    print(f'validation loss: {val_loss}')

            # tracking the metrics using comet in each iteration
            if args.use_comet:
                # tracker.track_metric('train_loss', round(train_loss.item(), 3), epoch)
                tracker.track_metric('train_loss', round(train_loss.item(), 3))
                # tracker.track_metric('val_loss', val_loss, optim_step)

            # backward and optimization step
            train_loss.backward()
            optimizer.step()

        # save the model every several steps if wanted by the user
        if epoch % save_model_interval == 0 and args.save_checkpoints:
            pool_mode = transition_params['pool_mode']  # extracted for saving the models

            # also use the value of r for saving the model in case the pool mode is 'lse'
            if pool_mode == 'lse':
                r = transition_params['r']
                pool_mode += f'_r={r}'

            models_folder = f'models/max_epochs={max_epochs}_batch_size={batch_size}_pool_mode={pool_mode}'
            helper.save_model(unified_net, optimizer, models_folder, epoch)
        epoch += 1


if __name__ == '__main__':
    main()
