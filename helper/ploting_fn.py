import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import cv2

def plot_ROC(prediction, target, class_names, save=False):
    """
    This function plots the ROC graph and prints the AUC values for each class
    :param prediction: a one-hot array [n_points, n_classes] with the predicted classification
    :param target: a one-hot array [n_points, n_classes] with the correct classification
    :param class_names: list of strings with names of the classes on the same order as the last arrays [n_classes]
    :param save: boolean, True to save the plot as roc_curve.png
    :return:
    """
    n_classes = prediction.shape[1]
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(prediction[:, i], target[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot ROC curve
    plt.figure()
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label=' {} '.format(class_names[i]))
        print("class {0} with AUC = {1:0.2f}".format(class_names[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('1 - Specificity')
    plt.ylabel('Sensitivity')
    plt.title('ResNet')
    plt.legend(loc="lower right")
    if save:
        plt.savefig("roc_curve.png")
    plt.show()

def plot_heatmaps(image_batch, model, save_path='./figures/', resize_dim=(224, 224)):
    """
    Plots class activation maps for a batch of images.
    :param image_batch: Batch of images, dimensions (Batch size, 3, H, W)
    :param model: Trained model having option to compute CAM
    :param save_path: Path to store figures of heatmaps
    :param resize_dim: Dimension for upsampling of images
    :return:
    """
    out = model(image_batch, CAM=True)
    for i in range(len(out)):
        for c in range(out[0].size(0)):
            img = out[i][c].detach().numpy()

            #plt.pcolormesh(cv2.resize(img, resize_dim))

            plt.imshow(cv2.resize(img, resize_dim))
            
            plt.title('Activation map, sample {}, class {}'.format(i, c))
            plt.savefig(f'{save_path}CAM_sample_{i}_class_{c}.png')
            #plt.show()