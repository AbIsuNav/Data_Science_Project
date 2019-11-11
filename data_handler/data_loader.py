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
    def __init__(self, img_ids, labels, labels_hot, data_folder, preprocess, device):
        """
        Initialization of the custom Dataset.
        :param img_ids: list of the id of the images in the dataset_path
        :param labels: list of all the corresponding labels
        :param labels_hot: labels in the one-hot format (for our case multiple hots)
        :param data_folder: the folder containing the data. NOTE: this function expects a single directory name
        and automatically looks for that directory in the 'data/' folder.
        """
        self.labels = labels
        self.labels_hot = labels_hot
        self.img_ids = img_ids
        self.data_folder = data_folder
        self.preprocess = preprocess
        self.device = device

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
        # preprocessing the image (crop etc.) and converting to the available device
        input_tensor = self.preprocess(img)
        # unsqueeze avoided because the image has 3 channels itself
        # input_tensor = input_tensor.unsqueeze(0)

        # convert 1d list to np array so that Pytorch can easily convert to tensor
        labels = np.array(self.labels_hot[img_id])

        sample = {'image': input_tensor, 'label': labels}
        return sample
        # return input_tensor, labels


def read_and_partition_data(h5_file, limited=False, val_frac=0.2, test_frac=0.1):
    """
    This function reads the image ids from the h5 file and partitions the data into train, validation, and test.
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
        print('In [read_indices_and_partition]: found {} images in the h5 file'.format(data_size))

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

    return img_ids, labels, labels_hot
