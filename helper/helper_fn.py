import networks

import os

import torch
from torchvision import transforms


def compute_val_loss(unified_net, val_loader, device):
    """
    Computes the validation loss and return its float number.
    :param unified_net: the trained net.
    :param val_loader: validation data loader
    :param device: -
    :return: the float number representing the validation loss
    """
    print(f'\nIn [compute_val_loss]: computing validation loss for {len(val_loader)} batches...')

    val_loss = 0
    for i_batch, batch in enumerate(val_loader):
        img_batch, label_batch = batch['image'].to(device), batch['label'].to(device)
        val_pred = unified_net(img_batch)

        batch_loss = networks.WCEL(val_pred, label_batch)
        val_loss += batch_loss.item()

        # print(f'batch: {i_batch}, val_loss: {batch_loss.item()}')

    val_avg = val_loss / len(val_loader)  # taking the average over all the batches
    val_avg = round(val_avg, 3)  # round to three floating points

    print(f'In [compute_val_loss]: validation loss: {val_avg} \n')
    return val_avg


def save_model(model, optimizer, models_folder, epoch):
    """
    This function saves the state of the model and optimizer at the given iteration. I save the optimizer because it may
    be useful later for continuing training.
    :param model: the model to be saved
    :param optimizer: the optimizer to be saved
    :param models_folder: the models folder where the model and optimizer will be saved at
    :param epoch: the optimization step, used for naming the saved model and optimizer
    :return: -
    """
    # create the models directory if not exists
    if not os.path.isdir(models_folder):
        os.mkdir(models_folder)
        print(f'Models folder created at: {models_folder}/')

    model_name = f'{models_folder}/unified_net_epoch_{epoch}.pt'
    optimizer_name = f'{models_folder}/optimizer_epoch_{epoch}.pt'

    torch.save(model.state_dict(), model_name)
    torch.save(optimizer.state_dict(), optimizer_name)
    print(f'Saved the model and optimizer at: "{model_name}" and "{optimizer_name}"')


def load_model(model_name, device, transition_params=None, which_resnet=None, unified_net_params=None):
    """
    This function loads a model or an optimizer, based on the model name and the parameters given. It infers from the
    model name if it should load a model or an optimizer, so the model name is super important.
    :param model_name:
    :param device:
    :param transition_params:
    :param which_resnet:
    :param unified_net_params:
    :return:
    """
    # loading a model
    if 'unified_net' in model_name:
        unified_net = networks.UnifiedNetwork(transition_params, which_resnet).to(device)
        if device == "cpu":
            unified_net.load_state_dict(torch.load(model_name, map_location=torch.device('cpu')))
        else:
            unified_net.load_state_dict(torch.load(model_name))
        unified_net.eval()  # putting in evaluation mode (for batch normalization, dropout and so on, if it has any)
        # do I need to set all param requires_grad to False?
        return unified_net
    # loading an optimizer
    elif 'optimizer' in model_name:
        optimizer = torch.optim.Adam(unified_net_params)
        optimizer.load_state_dict(torch.load(model_name))
        return optimizer
    else:  # is it the best kind of Exception to be used?
        raise ValueError('In [load_model]: Model name not supported for loading!')


def preprocess_fn(no_crop=False):
    # image preprocessing functions (not sure why, taken from https://pytorch.org/hub/pytorch_vision_resnet/)
    if no_crop:  # does not crop the image into 224x224 size
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:  # crops the image in the center and changes the size to 224x224
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    return preprocess
