import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


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