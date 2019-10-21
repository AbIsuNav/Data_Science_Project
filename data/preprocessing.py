import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import cv2
from cv2 import imread, createCLAHE # read and equalize images
from glob import glob
#matplotlib inline
import matplotlib.pyplot as plt
from itertools import chain
import h5py
from tqdm import tqdm
from skimage.transform import resize
#https://pytorch.org/hub/pytorch_vision_resnet/
#https://github.com/pytorch/examples/blob/42e5b996718797e45c46a25c55b031e6768f8440/imagenet/main.py#L89-L101
#https://www.kaggle.com/kmader/create-a-mini-xray-dataset-equalized/data


def write_df_as_hdf(out_path,
                    out_df,
                    compression='gzip'):
    with h5py.File(out_path, 'w') as h:
        for k, arr_dict in tqdm(out_df.to_dict().items()):
            try:
                s_data = np.stack(arr_dict.values(), 0)
                try:
                    h.create_dataset(k, data=s_data, compression=
                    compression)
                except TypeError as e:
                    try:
                        h.create_dataset(k, data=s_data.astype(np.string_),
                                         compression=compression)
                    except TypeError as e2:
                        print('%s could not be added to hdf5, %s' % (
                            k, repr(e), repr(e2)))
            except ValueError as e:
                print('%s could not be created, %s' % (k, repr(e)))
                all_shape = [np.shape(x) for x in arr_dict.values()]
                warn('Input shapes: {}'.format(all_shape))


def imread_and_normalize(im_path):
    clahe_tool = createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_data = np.mean(imread(im_path), 2).astype(np.uint8)
    img_data = clahe_tool.apply(img_data)
    n_img = (255*resize(img_data, OUT_DIM, mode = 'constant')).clip(0,255).astype(np.uint8)
    return np.expand_dims(n_img, -1)


all_xray_df = pd.read_csv('./Data_Entry_2017.csv')
all_image_paths = {os.path.basename(x): x for x in glob(os.path.join('.', 'images*', '*.png'))}
print('Scans found:', len(all_image_paths), ', Total Headers', all_xray_df.shape[0])
all_xray_df['path'] = all_xray_df['Image Index'].map(all_image_paths.get)
all_xray_df['Patient Age'] = all_xray_df['Patient Age'].map(lambda x: int(x))
# Here we take the labels and make them into a more clear format.
# The primary step is to see the distribution of findings and then to convert them to simple binary labels.
label_counts = all_xray_df['Finding Labels'].value_counts()[:15]
all_xray_df['Finding Labels'] = all_xray_df['Finding Labels'].map(lambda x: x.replace('No Finding', ''))
all_labels = np.unique(list(chain(*all_xray_df['Finding Labels'].map(lambda x: x.split('|')).tolist())))
print('All Labels', all_labels)
for c_label in all_labels:
    if len(c_label)>1: # leave out empty labels
        all_xray_df[c_label] = all_xray_df['Finding Labels'].map(lambda finding: 1.0 if c_label in finding else 0)
# all_xray_df.sample(3)
# since we can't have everything make a nice subset
# weight is 0.1 + number of findings
sample_weights = all_xray_df['Finding Labels'].map(lambda x: len(x.split('|')) if len(x)>0 else 0).values + 1e-1
sample_weights /= sample_weights.sum()
all_xray_df = all_xray_df.sample(18000, weights=sample_weights)

label_counts = all_xray_df['Finding Labels'].value_counts()[:15]

write_df_as_hdf('chest_xray.h5', all_xray_df)
with h5py.File('chest_xray.h5', 'r') as h5_data:
    for c_key in h5_data.keys():
        print(c_key, h5_data[c_key].shape, h5_data[c_key].dtype)

for i in range(len(all_xray_df['path'])):
    impath = all_xray_df['path'].values[i]
    oriimg = imread(impath, 1)
    height, width, depth = oriimg.shape
    imgScale = 256/width
    newX, newY = oriimg.shape[1] * imgScale, oriimg.shape[0] * imgScale
    newimg = cv2.resize(oriimg, (int(newX), int(newY)))
    #cv2.imshow("Show by CV2", newimg)
    #cv2.waitKey(0)
    cv2.imwrite("./new_images/"+all_xray_df['Image Index'].values[i], newimg)