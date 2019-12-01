from comet_ml import Experiment  # do not touch this, comet_ml should be imported before other modules such as torch

import data_handler
import networks
import helper

import json
import argparse
import os
import math
import time

from networks import resnet_att2
import torch
from torch.backends import cudnn
from torch.optim.lr_scheduler import MultiStepLR


def read_params_and_args():
    # Since not everybody uses comet, we determine if we should use comet_ml to track the experiment using the args
    parser = argparse.ArgumentParser()
    parser.add_argument('--evaluate', action='store_true')  # if true, ROC plots will be drawn
    parser.add_argument('--model_path', type=str, default='')  # the model to be evaluated

    parser.add_argument('--use_comet', action='store_true')  # if used, the experiment would be tracked by comet
    parser.add_argument('--save_checkpoints', action='store_true')  # if used, saves the model checkpoints if wanted

    parser.add_argument('--lr', type=float, default=0.001)  # setting lr, may be removed after grid search
    parser.add_argument('--wdecay', type=float, default=0.0)
    parser.add_argument('--max_epochs', type=int, default=30)  # setting the max epoch, may be removed after grid search
    parser.add_argument('--simple_lr_decay', action='store_true')

    parser.add_argument('--net_type', type=str, default='unified_net')
    args = parser.parse_args()

    print(f'In [read_params_and_args]: running the program with arguments '
          f'use_comet: {args.use_comet}, '
          f'save_checkpoints: {args.save_checkpoints}, '
          f'lr: {args.lr}, '
          f'max_epochs: {args.max_epochs} \n')

    # reading the other params from the JSON file
    with open('params.json', 'r') as f:
        params = json.load(f)

    return args, params


def train(model, optimizer, model_params, train_params, args, early_stopping=True, tracker=None, scheduler=None, att2=False):
    max_epochs = train_params['max_epochs']
    batch_size = train_params['batch_size']
    save_model_interval = train_params['save_model_interval']

    train_loader, val_loader = train_params['train_loader'], train_params['val_loader']
    device = train_params['device']
    transition_params = model_params['transition_params']

    # count trainable params
    num_learnable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'In [train]: number of learnable params of the model: {num_learnable_params}, max_epochs = {max_epochs} \n')

    # setting up the name of the folder in which the model is going to be saved
    pool_mode = transition_params['pool_mode']  # extracted for saving the models

    # also use the value of r for saving the model in case the pool mode is 'lse'
    if pool_mode == 'lse':
        r = transition_params['r']
        pool_mode += f'_r={r}'

    # determining this part of the models folder based on whether we are using early stopping
    mins_since_epoch = int(time.time() / 60)  # used in naming the model folder to be unique from other runs
    models_folder = f'models/max_epochs={max_epochs}_' \
                    f'batch_size={batch_size}_' \
                    f'pool_mode={pool_mode}_' \
                    f'lr={args.lr}_' \
                    f'no_crop={True}_' \
                    f'es={early_stopping}_{mins_since_epoch}'

    # set up early stopping
    if early_stopping:
        best_val_loss = math.inf  # set to inf so that the first validation loss is less than this
        no_improvements = 0   # the number consecutive epochs through which validation loss has not improved
        patience = 3
        min_delta = .001

    # training
    epoch = 0
    while epoch < max_epochs:
        print(f'{"=" * 40} In epoch: {epoch} {"=" * 40}')
        print(f'Training on {len(train_loader)} batches...')

        for i_batch, batch in enumerate(train_loader):
            # converting the labels batch  to from Long tensor to Float tensor (otherwise won't work on GPU)
            img_batch = batch['image'].to(device).float()
            label_batch = batch['label'].to(device).float()

            # making gradients zero in each optimization step
            optimizer.zero_grad()

            # getting the network prediction and computing the loss
            pred = model(img_batch, verbose=False)
            train_loss = networks.WCEL(pred, label_batch, att2)

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
            helper.save_model(model, optimizer, models_folder, epoch)

        # compute the validation loss at the end of each epoch
        val_loss = helper.compute_val_loss(model, val_loader, device, att2)

        # track the validation loss using comet, if wanted by the user
        if args.use_comet:
            tracker.track_metric('val_loss', val_loss)

        # check validation loss for early stopping
        if early_stopping:
            print(f'\nIn [train]: prev_val_loss: {best_val_loss}, current_val_loss: {val_loss}')

            # check if the validation loss is improved compared to the previous epochs
            if val_loss > best_val_loss or best_val_loss - val_loss < min_delta:
                no_improvements += 1
                print(f'In [train]: no_improvements incremented to {no_improvements} \n\n')

            else:  # it is improved, reset no_improvements to 0
                no_improvements = 0
                # update the validation loss for the next epoch
                best_val_loss = val_loss
                print(f'In [train]: no_improvements set to 0 \n')

            # terminate training after several epochs without validation improvement
            if no_improvements >= patience:
                print(f'In [train]: no_improvements = {no_improvements}, training terminated...')
                break

        # learning rate decay, if wanted
        if scheduler is not None:
            scheduler.step()
            print('In [train]: learning rate scheduling step() done \n\n')
        epoch += 1


def main():
    args, params = read_params_and_args()

    # model evaluation
    if args.evaluate:
        helper.evaluate_model(args.model_path, params)

    # training
    else:
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
        low_lr = params['lower_lr']
        # network_type = params["network"]
        network_type = args.net_type
        # resnet and transition params
        which_resnet = params['which_resnet']
        transition_params = params['transition_params']  # if the pool mode is 'max' or 'avg', the r value is imply ignored
        print('In [main]: transition params:', transition_params)
        attention2 = False
        # read the data and the labels
        partition, labels, labels_hot = \
            data_handler.read_already_partitioned(h5_file)

        # not sure why such preprocessing is needed (taken from taken from https://pytorch.org/hub/pytorch_vision_resnet/)
        preprocess = helper.preprocess_fn(no_crop=True)  # does not crop the images

        # cuda for PyTorch
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda:0" if use_cuda else "cpu")
        cudnn.benchmark = True  # this finds the optimum algorithm for the hardware if possible by kind of auto-tuning

        # creating the train and validation data loaders
        loader_params = {'batch_size': batch_size, 'shuffle': shuffle, 'num_workers': num_workers}
        train_loader, val_loader, _ = \
            data_handler.create_data_loaders(partition, labels, labels_hot, data_folder, preprocess, device, loader_params)

        # the model
        if network_type == "attention2":
            unified_net = resnet_att2.ResNet_A2(which_resnet).to(device)
            attention2=True
        else:
            unified_net = networks.UnifiedNetwork(transition_params, which_resnet).to(device)

        # Adam optimizer with default parameters
        lr = args.lr
        decay = args.wdecay

        optimizer = torch.optim.Adam(params=unified_net.parameters(), lr=lr, weight_decay=decay)
        print(f'In [main]: created the Adam optimizer with learning rate: {lr}')

        if low_lr:
            milestones = list(range(0,max_epochs,20))
            scheduler = MultiStepLR(optimizer, milestones=milestones[1:])

        # simple learning rate decay
        lr_scheduler = None
        if args.simple_lr_decay:
            # make the learning rate half after the first 3 epochs: e.g., 1e-4 -> 5e-5 -> 2.5e-5 -> 1.25e-5
            lr_scheduler = MultiStepLR(optimizer, milestones=[1, 2, 3], gamma=0.5)

        # setting the training params and model params used during training
        model_params = {'transition_params': transition_params}
        train_params = {'max_epochs': max_epochs,
                        'batch_size': batch_size,
                        'save_model_interval': save_model_interval,
                        'train_loader': train_loader,
                        'val_loader': val_loader,
                        'device': device}

        # initialize our comet experiment to track the run, if wanted by the user
        if args.use_comet:
            tracker = helper.init_comet(params)
            print("In [main]: comet experiment initialized...")
            train(unified_net, optimizer, model_params, train_params, args,
                  early_stopping=True, tracker=tracker, scheduler=lr_scheduler, att2=attention2)

        else:
            train(unified_net, optimizer, model_params, train_params, args,
                  early_stopping=True, scheduler=lr_scheduler,att2=attention2)


def prepare_data_if_not_available():
    data_path = 'data/data_big_1024x1024'
    download_path = 'data/data_big_original/archives'
    extracted_path = 'data/data_big_original/extracted'

    # 'images' would automatically be appended to 'extracted_path' when extracting the tar files, so we need to add it
    # images_path = f'{extracted_path}/images'

    # if not os.path.isdir(data_path):
    # if not os.path.exists(download_path):  # download data if not already downloaded (naive checking)
    print(f'In [prepare_data_if_not_available]: downloading data at: "{download_path}"...')
    data_handler.download_data(download_path)

    # if not os.path.exists(extracted_path):  # extract archive path if not already extracted (naive checking)
    # print(f'In [prepare_data_if_not_available]: extracting the data to: "{extracted_path}..."\n')
    # data_handler.extract_data(archive_path=download_path, extract_path=extracted_path)

    print(f'In [prepare_data_if_not_available]: extracting the data to: "{data_path}..."\n')
    data_handler.extract_data(archive_path=download_path, extract_path=data_path)

    '''# down-sampling using the 'images_path'
    print(f'In [prepare_data_if_not_available]: down-sampling the data to: "{data_path}"...\n')
    data_handler.down_sample(images_path, data_path)'''
    # else:
    #    print(f'In [prepare_data_if_not_available]: data path "{data_path}" already exists.')


if __name__ == '__main__':
    # prepare_data_if_not_available()
    main()
