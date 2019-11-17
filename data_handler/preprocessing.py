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
from data_loader import read_ids_and_labels
#https://pytorch.org/hub/pytorch_vision_resnet/
#https://github.com/pytorch/examples/blob/42e5b996718797e45c46a25c55b031e6768f8440/imagenet/main.py#L89-L101
#https://www.kaggle.com/kmader/create-a-mini-xray-dataset-equalized/data


img_ids, _, _ = read_ids_and_labels("chest_xray.h5")
for i in range(len(img_ids)):
    impath = "data/images/"+img_ids[i]
    oriimg = imread(impath, 1)
    height, width, depth = oriimg.shape
    imgScale = 256/width
    newX, newY = oriimg.shape[1] * imgScale, oriimg.shape[0] * imgScale
    newimg = cv2.resize(oriimg, (int(newX), int(newY)))
    #cv2.imshow("Show by CV2", newimg)
    #cv2.waitKey(0)
    print(img_ids[i])
    cv2.imwrite("./data/new_images/"+img_ids[i], newimg)
print("Finished!")