"""
Python package for loading the data and creating batches.
Code adapted from: https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel
"""

import torch
from torch.utils import data
from PIL import Image
from torchvision import transforms

import os
import h5py
import numpy as np
import random


class Dataset(data.Dataset):
    """
    This characterizes a custom PyTorch dataset.
    """
    def __init__(self, img_ids, labels, labels_hot, data_folder, preprocess, device, scale='rgb'):
        """
        Initialization of the custom Dataset.
        :param img_ids: list of the id of the images in the dataset_path
        :param labels: list of all the corresponding labels
        :param labels_hot: labels in the one-hot format (for our case multiple hots)
        :param data_folder: the folder containing the data. NOTE: this function expects a single directory name
        :param scale: determines the type of the input images. If scale is 'gray', it will be converted to RGB in the
        __getitem__ function.
        and automatically looks for that directory in the 'data/' folder.
        """
        self.labels = labels
        self.labels_hot = labels_hot
        self.img_ids = img_ids
        self.data_folder = data_folder
        self.preprocess = preprocess
        self.device = device
        self.scale = scale

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, index):
        """
        :param index:
        :return: a dictionary accessible with sample['image'] and sample['label'] to obtain the image and the label
        respectively.
        """
        img_id = self.img_ids[index]

        # Load data and get label
        img = Image.open('data/{}/{}'.format(self.data_folder, img_id))

        # convert to RGB if needed
        if self.scale == 'gray':
            img = to_rgb(img)

        # preprocessing the image (crop etc.) and converting to the available device
        input_tensor = self.preprocess(img)

        # convert 1d list to np array so that Pytorch can easily convert to tensor
        labels = np.array(self.labels_hot[img_id])

        sample = {'image': input_tensor, 'label': labels}
        return sample
        # return input_tensor, labels


def to_rgb(gray_image):
    """
    Converts the gray-scale image to RGB.
    :param gray_image:
    :return:
    """
    rgb = Image.new('RGB', gray_image.size)
    rgb.paste(gray_image)
    return rgb


def read_already_partitioned(h5_file):
    """
    This function reads all the 112120 images of the dataset, and partitions them according to the 'train_val_list.txt'
    and 'test_list.txt' files obtained from the dataset website. It also extracts the labels from the .h5 file already
    created to simplify this label extraction process. It then stores the extracted labels as a Python dictionary
    for future references.

    :param h5_file: -
    :return: the dictionaries partition, labels, labels_hot. Read the 'read_ids_and_labels' function for more info.
    """
    if not os.path.exists('data/formatted_data.npy'):
        print('In [read_already_partitioned]: Reading the h5 file and saving the formatted data...')
        img_ids, labels, labels_hot = read_ids_and_labels(h5_file)
        save_formatted_data(img_ids, labels, labels_hot)
    else:
        print('In [read_already_partitioned]: "data/formatted_data.npy" already exists. Reading the formatted data...')
        # refer to: https://stackoverflow.com/questions/19201290/how-to-save-a-dictionary-to-a-file
        formatted_data = np.load('data/formatted_data.npy', allow_pickle=True).item()
        img_ids = formatted_data['image_ids']  # not actually used since the image ids are already in the .txt files
        labels = formatted_data['labels']
        labels_hot = formatted_data['labels_hot']

    # read the .txt file containing the image ids used for training, validation, and test
    with open('data/train_val_list.txt', 'r') as f:
        train_val_list = f.read().splitlines()

    with open('data/test_list.txt', 'r') as f:
        test_list = f.read().splitlines()

    # extract the train_val list to train and validation lists, based on the percentages in the paper
    train_size = int(.875 * len(train_val_list))  # 70% out of 80% for training, 10% out of 80% for validation
    train_list = train_val_list[:train_size]
    val_list = train_val_list[train_size:]

    # train_list = train_val_list[:300]

    # the value for each key is the list of ids for the corresponding set
    partition = {
        'train': train_list,
        'validation': val_list,
        'test': test_list
    }

    data_info = (len(partition['train']), len(partition['validation']),
                 len(partition['test']))
    print('In [read_already_partitioned]: returning with size train: {}, validation: {}, '
          'test: {}'.format(*data_info))

    return partition, labels, labels_hot


def read_and_partition_data(h5_file, limited=False, val_frac=0.2, test_frac=0.1):
    """
    This function reads the image ids from the h5 file and partitions the data into train, validation, and test.
    Note: This function was written to read 18000 images from the chest_xray_18000.h5 file, extract the labels, and
    then partition it manually using the given fractions. If one wants to read ALL the 112120 images, he/she should
    use the 'read_already_partitioned' function which uses the partition given in the dataset website (using the
    'train_val_list.txt' and 'test_list.txt' files)

    :param limited: if True, reads only a limited portion of data (used for testing)
    :param h5_file: -
    :param val_frac: -
    :param test_frac: -
    :return: three dictionaries: partition, labels, and labels_hot. partition is accessed through 'train',
    'validation', or 'test' and returns the list of image ids corresponding to that set. labels is accesses
    through labels['img_id'] and returns the list of the labels corresponding to that image. The same holds for
    labels_hot except that it returns the (multiple) hot encoded version of the labels corresponding to the image.

    NOTE: even when setting limited=True, the labels and labels_hot dictionaries contain the key, values for ALL the
    images, but this is not a problem as the limited labels are in img_ids and only image indexes from there are given
    to labels and labels_hot to obtain the labels.
    """
    img_ids, labels, labels_hot = read_ids_and_labels(h5_file)

    # limiting the data to a few images if wanted
    if limited:
        lim = 100
        img_ids = img_ids[:lim]  # only this is important, as labels and labels_hot are simply two dictionaries
        # labels = labels[:lim]
        # labels_hot = labels_hot[:lim]
        print(f'In [read_and_partition_data]: limited the data to {lim} images')

    data_size = len(img_ids)

    val_size = int(val_frac * data_size)
    test_size = int(test_frac * data_size)

    # note: these are indexes (for instance between 1 and 18000) rather than the actual img_id
    val_indexes = random.sample(population=range(data_size), k=val_size)
    test_indexes = random.sample(population=range(data_size), k=test_size)

    # the value for each key is the list of ids for the corresponding set
    partition = {
        'train': [],
        'validation': [],
        'test': []
    }

    # for each index in for example (0, 18000), gt the set this index belongs to (train, validation, or test)
    # and add the corresponding image id to the according set
    for index in range(data_size):
        belongs_to = 'validation' if index in val_indexes else 'test' if index in test_indexes else 'train'
        partition[belongs_to].append(img_ids[index])

    data_info = (len(partition['train']), len(partition['validation']),
                 len(partition['test']))  # , len(labels.keys()), len(labels_hot.keys()))
    print('In [read_and_partition_data]: returning with size train: {}, validation: {}, '
          'test: {}'.format(*data_info))  # , labels: {}, labels_hot:{}'.format(*data_info))

    return partition, labels, labels_hot


def read_ids_and_labels(h5_file, verbose=False):
    """
    This function reads the ids and labels of the images in an h5 file.
    :param h5_file: the file to read from. It should be the name of a file that exists in the 'data/' folder.
    :param verbose: if True, the function prints complete information while reading the data
    :return: img_ids, labels, labels_hot: img_ids is the list containing all the image names, labels and labels hot
    are two dictionaries. For usage please see the documentation of the 'read_and_partition_data' function.
    """
    # to be moved to a separate JSON file (maybe)
    pathologies = {'Atelectasis': 0,
                   'Cardiomegaly': 1,
                   'Consolidation': 2,
                   'Edema': 3,
                   'Effusion': 4,
                   'Emphysema': 5,
                   'Fibrosis': 6,
                   'Hernia': 7,
                   'Infiltration': 8,
                   'Mass': 9,
                   'Nodule': 10,
                   'Pleural_Thickening': 11,
                   'Pneumonia': 12,
                   'Pneumothorax': 13}

    # read the h5 file containing information about the images of the dataset
    with h5py.File('data/{}'.format(h5_file), 'r') as h5_data:
        image_ids = h5_data['Image Index']  # list of shape (18000,) for the limited data
        finding_labels = h5_data['Finding Labels']

        data_size = len(image_ids)
        print(f'In [read_indices_and_partition]: found {data_size} images in the h5 file. Extracting the labels...')

        img_ids, labels, labels_hot = [], {}, {}

        # for each image in the (limited) data
        for i in range(data_size):
            img_id = image_ids[i].decode('utf-8')  # file name is stored as byte-formatted in the h5 file
            diseases = finding_labels[i].decode("utf-8") .split('|')  # diseases split by '|'

            # vector containing multiple hot elements based on what pathologies are present
            diseases_hot_enc = [0] * len(pathologies.keys())

            if verbose:
                print('In [read_indices_and_labels]: diseases found:', diseases)

            # check if the patient has any diseases
            if diseases != ['']:
                pathology_ids = [pathologies[disease] for disease in diseases]  # get disease index
                if verbose:
                    print('In [read_indices_and_labels]: pathology indexes:', pathology_ids)

                for pathology_id in pathology_ids:
                    diseases_hot_enc[pathology_id] = 1  # change values from 0 to 1

            if verbose:
                print('In [read_indices_and_labels]: pathology indexes: one_hot: ', diseases_hot_enc)

            # append to the lists
            img_ids.append(img_id)
            # labels.append(pathology_ids)
            labels.update({img_id: pathology_ids})
            # labels_hot.append(diseases_hot_enc)
            labels_hot.update({img_id: diseases_hot_enc})

    print('In [read_indices_and_labels]: reading imag ids and extracting the labels done.')
    return img_ids, labels, labels_hot


def save_formatted_data(image_ids, labels, labels_hot):
    """
    This function saves the image ids, labels, and labels_hot into an .npy file for further reference. This is useful
    because reading the .h5 file and extracting the labels and labels_hot every time is time-consuming for all the
    112120 images.
    :param image_ids: -
    :param labels: -
    :param labels_hot: -
    :return: -
    """
    # packing all the data into a dictionary
    formatted_data = {'image_ids': image_ids,
                      'labels': labels,
                      'labels_hot': labels_hot}

    np.save('data/formatted_data.npy', formatted_data)
    print('In [save_formatted_data]: save the formatted data.')


def create_data_loaders(partition, labels, labels_hot, data_folder, preprocess, device, loader_params, scale='rgb'):
    batch_size = loader_params['batch_size']
    shuffle = loader_params['shuffle']
    num_workers = loader_params['num_workers']

    # creating the train data loader
    train_set = Dataset(partition['train'], labels, labels_hot, data_folder, preprocess, device, scale)
    train_loader = data.DataLoader(dataset=train_set, batch_size=batch_size,
                                   shuffle=shuffle, num_workers=num_workers)
    # creating the validation data loader
    val_set = Dataset(partition['validation'], labels, labels_hot, data_folder, preprocess, device, scale)
    val_loader = data.DataLoader(dataset=val_set, batch_size=batch_size,
                                 shuffle=False, num_workers=num_workers)

    # creating the validation data loader
    test_set = Dataset(partition['test'], labels, labels_hot, data_folder, preprocess, device, scale)
    test_loader = data.DataLoader(dataset=test_set, batch_size=batch_size,
                                  shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader
