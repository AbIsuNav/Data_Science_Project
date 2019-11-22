import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from sklearn.metrics import roc_curve, auc
import cv2
import torch

from . import helper_fn
import data_handler


def evaluate_model(model_path, params):
    no_crop = True if 'no_crop' in model_path else False

    # adjust S, for models with no_crop in their names, S is 8 because the training and test images were of size 256x256
    transition_params = params['transition_params']
    transition_params['S'] = 8 if no_crop else 7

    # for the very first saved models, include_1x1_conv was set to False and needs to be adjusted
    # if model_path == 'models/unified_net_step_9.pt' or model_path == 'models/unified_net_epoch_25.pt':
    #    transition_params['include_1x1_conv'] = False

    print(f'In [evaluate_model]: \n'
          f'model: {model_path} \n'
          f'transition_params: {transition_params} \n')

    # data files
    data_folder = params['data_folder']
    h5_file = params['h5_file']
    pathologies = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion', 'Emphysema', 'Fibrosis',
                   'Hernia',
                   'Infiltration', 'Mass', 'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax']

    # training params
    batch_size = params['batch_size']
    shuffle = params['shuffle']
    num_workers = params['num_workers']

    preprocess = helper_fn.preprocess_fn(no_crop)  # the function needs to know if it should crop the images
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    # read the data and the labels
    partition, labels, labels_hot = \
        data_handler.read_already_partitioned(h5_file)

    loader_params = {'batch_size': batch_size, 'shuffle': shuffle, 'num_workers': num_workers}
    train_loader, _, test_loader = \
        data_handler.create_data_loaders(partition, labels, labels_hot, data_folder, preprocess, device, loader_params)

    print(f'In [evaluate_model]: loaded train and test loaders '
          f'with number of batches {len(train_loader)} and {len(test_loader)}')

    # copied exactly from Abgeiba's code
    total_predicted = np.zeros((batch_size, 14))
    total_labels = np.zeros((batch_size, 14))
    for i_batch, batch in enumerate(train_loader):
        img_batch = batch['image'].to(device).float()
        label_batch = batch['label'].to(device).float()
        net = helper_fn.load_model(model_path, torch.device('cpu'), transition_params, 'resnet34')
        pred = net(img_batch, verbose=False)
        if i_batch > 0:
            total_predicted = np.append(total_predicted, pred.detach().numpy(), axis=0)
            total_labels = np.append(total_labels, label_batch.detach().numpy(), axis=0)
        else:
            total_predicted = pred.detach().numpy()
            total_labels = label_batch.detach().numpy()
    plot_ROC(total_predicted, total_labels, pathologies, save=True)
    folder_path = model_path.split("/")
    plot_ROC(total_predicted, total_labels, pathologies, save=True, folder="results/"+folder_path[1])


def plot_ROC(prediction, target, class_names, save=False, folder=""):
    """
    This function plots the ROC graph and prints the AUC values for each class
    :param predicted: a one-hot array [n_points, n_classes] with the predicted classification
    :param target: a one-hot array [n_points, n_classes] with the correct classification
    :param class_names: list of strings with names of the classes on the same order as the last arrays [n_classes]
    :param save: boolean, True to save the plot as roc_curve.png
    :return:
    """
    n_classes = prediction.shape[1]
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    target = target.astype(int)
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(target[:, i], prediction[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot ROC curve
    plt.figure()
    auc_list = list()
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label=' {} '.format(class_names[i]))
        auc_list.append("class {0}, AUC = {1:0.2f}".format(class_names[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('1 - Specificity')
    plt.ylabel('Sensitivity')
    plt.title('ResNet')
    plt.legend(loc="lower right")
    if save:
        plt.savefig(folder+"/roc_curve.png")
    with open(folder+'/auc.txt', 'w') as f:
        for item in auc_list:
            f.write("%s\n" % item)
    #plt.show()


def plot_heatmaps(image_batch, model, resize_dim=(224, 224), save_path='./figures/', compare=False):
    """
    Plots class activation maps for a batch of images.
    :param image_batch: Batch of images, dimensions (Batch size, 3, H, W)
    :param model: Trained model having option to compute CAM
    :param save_path: Path to store figures of heatmaps, False
    :param resize_dim: Dimension for upsampling of images
    :param compare: set True to show figure with heatmap and original image overlapping
    :return:
    """
    model.eval()
    out = model(image_batch, CAM=True)
    # normalization
    out = out - torch.min(out, -1)[0].min(-1)[0].unsqueeze(-1).unsqueeze(-1)
    out = out / torch.max(out, -1)[0].max(-1)[0].unsqueeze(-1).unsqueeze(-1)
    for i in range(len(out)):
        for c in range(out[0].size(0)):
            heatmap = out[i][c].detach().numpy()
            result = cv2.resize(heatmap, resize_dim)
            # merge heatmap and image(make them transparent) on a same image
            if compare:
                result = result * 0.3 + image_batch[i] * 0.5
            #plt.pcolormesh(cv2.resize(img, resize_dim))
            plt.imshow(result)
            plt.title('Activation map, sample {}, class {}'.format(i, c))
            plt.savefig(f'{save_path}CAM_sample_{i}_class_{c}.png')
            #plt.show()



def generate_bbox(image_batch, model, resize_dim=(224, 224), save_path='./figures/', threshold=[60, 180], merge=False):
    """
    Generationg bounding box on the original image:
    Threshold the image and rectangular the contours, pick the largest one.
    further method include merge/add boxes from different classes given the prediction.
    :return: cooridnator of the bounding box of each class
    """
    # assume it comes in batch, generate the heatmap first
    model.eval()
    out = model(image_batch, CAM=True)
    # normalization
    out = out - torch.min(out, -1)[0].min(-1)[0].unsqueeze(-1).unsqueeze(-1)
    out = out / torch.max(out, -1)[0].max(-1)[0].unsqueeze(-1).unsqueeze(-1)

    H, W = resize_dim
    classes = out[0].size(0)
    bbox_index = {}# with (sample, class, threshold) as keys
    for i in range(len(out)):
        image = image_batch[i]
        for c in range(classes):
            heatmap = cv2.resize(out[i][c].detach().numpy(), resize_dim)
            # 'two-level thresholding'
            maxValue = heatmap.max(axis=-1).max()
            for thre in threshold:
                thresh = thre/255 * maxValue
                th, dst = cv2.threshold(heatmap, thresh, maxValue, cv2.THRESH_BINARY)
                # detect all contours and rectangular them
                data = 255 * dst
                img = data.astype(np.uint8)
                _, contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                bbox = []
                bbox_area = []
                for cnt in contours:
                    (x, y, w, h) = cv2.boundingRect(cnt)
                    bbox_x, bbox_y, bbox_w, bbox_h = limit(x, y, w, h, W, H)
                    bbox.append([bbox_x, bbox_y, bbox_w, bbox_h])
                    bbox_area.append(bbox_w * bbox_h)
                # choose the largest one
                largest = bbox_area.index(max(bbox_area))
                (x, y, w, h) = bbox[largest][0], bbox[largest][1], bbox[largest][2], bbox[largest][3]
                if merge:
                    # compute overlap, rank and merge, unfinished
                    pass
                bbox_index[i, c, thre] = (x, y, w, h)
            fig, ax = plt.subplots(1)
            (x, y, w, h) = bbox_index[i, c, threshold[0]]
            (x2, y2, w2, h2) = bbox_index[i, c, threshold[1]]
            ax.imshow(image)
            ax.add_patch(patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='r', facecolor='none'))
            ax.add_patch(patches.Rectangle((x2, y2), w2, h2, linewidth=2, edgecolor='g', facecolor='none'))
            #plt.savefig(f'{save_path}Bbox_sample_{i}_class_{c}.png')
            plt.show()

    return bbox_index


def limit(x, y, w, h, W, H):
    bbox_x = min(max(x, 0), W - 5)
    bbox_y = min(max(y, 0), H - 5)
    bbox_w = min(max(w, 0), W - x - 5)
    bbox_h = min(max(h, 0), H - y - 5)

    return bbox_x, bbox_y, bbox_w, bbox_h

