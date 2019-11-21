from comet_ml import Experiment  # do not touch this, comet_ml should be imported before other modules such as torch

import data_handler
import networks
import helper

import json
import argparse
import os
import math

import torch
from torch.backends import cudnn
from torch.utils import data


def read_params_and_args():
    # Since not everybody uses comet, we determine if we should use comet_ml to track the experiment using the args
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_comet', action='store_true')  # if used, the experiment would be tracked by comet
    parser.add_argument('--save_checkpoints', action='store_true')  # if used, saves the model checkpoints if wanted
    parser.add_argument('--lr', type=float, default=0.001)  # setting lr, may be removed after grid search
    parser.add_argument('--max_epochs', type=int, default=30)  # setting the max epoch, may be removed after grid search
    parser.add_argument('--no_crop', action='store_true')

    # we probably do not use this anymore
    parser.add_argument('--data_limited', action='store_true')  # if used, only 100 images are chosen for training
    args = parser.parse_args()

    print(f'In [read_params_and_args]: running the program with arguments '
          f'use_comet: {args.use_comet}, '
          f'save_checkpoints: {args.save_checkpoints}, '
          f'lr: {args.lr}, '
          f'max_epochs: {args.max_epochs}, '
          f'no_crop: {args.no_crop}, '
          f'data_limited: {args.data_limited} \n')

    # reading the other params from the JSON file
    with open('params.json', 'r') as f:
        params = json.load(f)

    return args, params


def train(model, optimizer, model_params, train_params, args, es_params, tracker=None):
    max_epochs = train_params['max_epochs']
    batch_size = train_params['batch_size']
    save_model_interval = train_params['save_model_interval']

    train_loader, val_loader = train_params['train_loader'], train_params['val_loader']
    device = train_params['device']
    transition_params = model_params['transition_params']

    # count trainable params
    num_learnable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'In [train]: number of learnable params of the model: {num_learnable_params}, max_epochs = {max_epochs} \n')

    # setting variables for early stopping, if wanted by the user
    if es_params is not None:
        prev_val_loss = math.inf  # set to inf so that the first validation loss is less than this
        no_improvements = 0  # the number consecutive epochs through which validation loss has not improved

    epoch = 0
    while epoch < max_epochs:
        print(f'{"=" * 40} In epoch: {epoch} {"=" * 40}')
        print(f'Training on {len(train_loader)} batches...')

        # check if resnet output is saved
        '''
        if save_resnet_out:
        if not directory exists
        create the dir and save the whole (train) tensor
        
        else
        resnet_out = load_tensor  (B, D, S, S) => e.g, (296, 256, 7, 7)
        
        i_batch = ?
        batch = resnet_out[i_batch]  # (D, S, S) => e.g, (256, 7, 7)
        
        if the first time:
            concat tensor
        '''
        for i_batch, batch in enumerate(train_loader):
            # print(f'Performing on batch: {i_batch}')
            # img_batch, label_batch = img_batch.to(device), label_batch.to(device)
            img_batch = batch['image'].to(device).float()
            label_batch = batch['label'].to(device).float()
            # converted the labels batch  to from Long tensor to Float tensor (otherwise won't work on GPU)

            # making gradients zero in each optimization step
            optimizer.zero_grad()

            # getting the network prediction and computing the loss
            pred = model(img_batch, verbose=False)

            train_loss = networks.WCEL(pred, label_batch)
            if i_batch % 50 == 0:
                print(f'Batch: {i_batch}, train loss: {round(train_loss.item(), 3)}')

            # tracking the metrics using comet in each iteration
            if args.use_comet:
                tracker.track_metric('train_loss', round(train_loss.item(), 3))

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

            # determining this part of the models folder based on whether we are using early stopping
            if es_params is None:
                max_epochs_name = max_epochs
            else:
                max_epochs_name = f'{max_epochs}_es_patience={es_params["patience"]}_min_delta={es_params["min_delta"]}'

            models_folder = f'models/max_epochs={max_epochs_name}_' \
                            f'batch_size={batch_size}_' \
                            f'pool_mode={pool_mode}_' \
                            f'lr={args.lr}_' \
                            f'no_crop={args.no_crop}'
            helper.save_model(model, optimizer, models_folder, epoch)

        # compute the validation loss at the end of each epoch
        val_loss = helper.compute_val_loss(model, val_loader, device)

        # track the validation loss using comet, if wanted by the user
        if args.use_comet:
            tracker.track_metric('val_loss', val_loss)

        # check validation loss for early stopping
        if es_params is not None:
            print(f'\nIn [train]: prev_val_loss: {prev_val_loss}, current_val_loss: {val_loss}')

            # check if the validation loss is improved compared to the previous epochs
            if val_loss > prev_val_loss or prev_val_loss - val_loss < es_params['min_delta']:
                no_improvements += 1
                print(f'In [train]: no_improvements incremented to {no_improvements} \n\n')

            else:  # if it is improved, reset no_improvements
                no_improvements = 0
                print(f'In [train]: no_improvements set to 0 \n\n')

            # update the validation loss for the next epoch
            prev_val_loss = val_loss

            # terminate training after several epochs without validation improvement
            if no_improvements > es_params['patience']:
                print(f'In [train]: no_improvements = {no_improvements}, training terminated...')
                break
        epoch += 1


def main():
    args, params = read_params_and_args()

    # data files
    data_folder = params['data_folder']
    h5_file = params['h5_file']

    # training params
    batch_size = params['batch_size']
    shuffle = params['shuffle']
    num_workers = params['num_workers']
    # max_epochs = params['max_epochs']
    max_epochs = args.max_epochs
    save_model_interval = params['save_model_interval']

    # resnet and transition params
    which_resnet = params['which_resnet']
    transition_params = params['transition_params']  # if the pool mode is 'max' or 'avg', the r value is imply ignored

    # adjusting the S value, if no_crop is used, the 256x256 images will result in 512x8x8 feature maps
    transition_params['S'] = 8 if args.no_crop else 7

    print('In [main]: transition params:', transition_params)
    print('In [main]: es_params:', params['es_params'])
    # print('Note: "r" will simply be ignored if the pool mode is "max" or "avg"', '\n')

    '''# reading image ids and partition them into train, validation, and test sets
    partition, labels, labels_hot = \
        data_handler.read_and_partition_data(h5_file, limited=args.data_limited, val_frac=0.2, test_frac=0.1)'''

    # read the data and the labels
    partition, labels, labels_hot = \
        data_handler.read_already_partitioned(h5_file)

    # not sure why such preprocessing is needed (taken from taken from https://pytorch.org/hub/pytorch_vision_resnet/)
    preprocess = helper.preprocess_fn(no_crop=args.no_crop)

    # cuda for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    cudnn.benchmark = True  # this finds the optimum algorithm for the hardware if possible by kind of auto-tuning

    # creating the train and validation data loaders
    loader_params = {'batch_size': batch_size, 'shuffle': shuffle, 'num_workers': num_workers}
    train_loader, val_loader = \
        data_handler.create_data_loaders(partition, labels, labels_hot, data_folder, preprocess, device, loader_params)

    # the model
    unified_net = networks.UnifiedNetwork(transition_params, which_resnet).to(device)

    # Adam optimizer with default parameters
    lr = args.lr
    optimizer = torch.optim.Adam(params=unified_net.parameters(), lr=lr)
    print(f'In [main]: created the Adam optimizer with learning rate: {lr}')

    # setting the training params and model params used during training
    model_params = {'transition_params': transition_params}
    train_params = {'max_epochs': max_epochs,
                    'batch_size': batch_size,
                    'save_model_interval': save_model_interval,
                    'train_loader': train_loader,
                    'val_loader': val_loader,
                    'device': device}

    # params for early stopping, set to None if not interested in early stopping
    es_params = params['es_params']

    # initialize our comet experiment to track the run, if wanted by the user
    if args.use_comet:
        tracker = helper.init_comet(params)
        print("In [main]: comet experiment initialized...")
        train(unified_net, optimizer, model_params, train_params, args, es_params, tracker)

    else:
        train(unified_net, optimizer, model_params, train_params, args, es_params)


if __name__ == '__main__':
    main()
