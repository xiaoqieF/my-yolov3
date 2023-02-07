from networks.yolov3 import YOLOBody
from util.loss import compute_loss
from torch import nn, optim
import torch
from torch.utils.data import DataLoader
from util.hyp import HYP
from util.transform import DEFAULT_TRANSFORMS, VAL_TRANSFORMS
from util.dataset import YoloDataset
from util.util import non_max_suppression
from util.boxes import BoxDecoder
import tqdm
from util.util import load_classes, ap_per_class, get_batch_statistics, non_max_suppression, xywh2xyxy 
import numpy as np
from terminaltables import AsciiTable
from torch.autograd import Variable


def evaluate_model_file(model, dataloader):
    """Evaluate model on validation dataset.
    :param model_path: Path to model definition file (.cfg)
    :type model_path: str
    :param weights_path: Path to weights or checkpoint file (.weights or .pth)
    :type weights_path: str
    :param img_path: Path to file containing all paths to validation images.
    :type img_path: str
    :param class_names: List of class names
    :type class_names: [str]
    :param batch_size: Size of each image batch, defaults to 8
    :type batch_size: int, optional
    :param img_size: Size of each image dimension for yolo, defaults to 416
    :type img_size: int, optional
    :param n_cpu: Number of cpu threads to use during batch generation, defaults to 8
    :type n_cpu: int, optional
    :param iou_thres: IOU threshold required to qualify as detected, defaults to 0.5
    :type iou_thres: float, optional
    :param conf_thres: Object confidence threshold, defaults to 0.5
    :type conf_thres: float, optional
    :param nms_thres: IOU threshold for non-maximum suppression, defaults to 0.5
    :type nms_thres: float, optional
    :param verbose: If True, prints stats of model, defaults to True
    :type verbose: bool, optional
    :return: Returns precision, recall, AP, f1, ap_class
    """
    class_names = load_classes('./my_yolo_dataset/my_data_label.names')

    metrics_output = _evaluate(
        model,
        dataloader,
        class_names,
        416,
        0.5,
        0.1,
        0.5,
        True)
    return metrics_output

def print_eval_stats(metrics_output, class_names, verbose):
    if metrics_output is not None:
        precision, recall, AP, f1, ap_class = metrics_output
        if verbose:
            # Prints class AP and mean AP
            ap_table = [["Index", "Class", "AP"]]
            for i, c in enumerate(ap_class):
                ap_table += [[c, class_names[c], "%.5f" % AP[i]]]
            print(AsciiTable(ap_table).table)
        print(f"---- mAP {AP.mean():.5f} ----")
    else:
        print("---- mAP not measured (no detections found by model) ----")

def _evaluate(model, dataloader, class_names, img_size, iou_thres, conf_thres, nms_thres, verbose):
    """Evaluate model on validation dataset.
    :param model: Model to evaluate
    :type model: models.Darknet
    :param dataloader: Dataloader provides the batches of images with targets
    :type dataloader: DataLoader
    :param class_names: List of class names
    :type class_names: [str]
    :param img_size: Size of each image dimension for yolo
    :type img_size: int
    :param iou_thres: IOU threshold required to qualify as detected
    :type iou_thres: float
    :param conf_thres: Object confidence threshold
    :type conf_thres: float
    :param nms_thres: IOU threshold for non-maximum suppression
    :type nms_thres: float
    :param verbose: If True, prints stats of model
    :type verbose: bool
    :return: Returns precision, recall, AP, f1, ap_class
    """
    model.eval()  # Set model to evaluation mode
    decoder = BoxDecoder()

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    labels = []
    sample_metrics = []  # List of tuples (TP, confs, pred)
    for imgs, targets in tqdm.tqdm(dataloader, desc="Validating"):
        imgs = imgs.to(torch.device("cuda:0"))
        targets = targets.to(torch.device("cuda:0"))
        # Extract labels
        labels += targets[:, 1].tolist()
        # Rescale target
        targets[:, 2:] = xywh2xyxy(targets[:, 2:])
        targets[:, 2:] *= img_size

        with torch.no_grad():
            outputs = model(imgs)
            outputs = decoder.decode(outputs)
            outputs =non_max_suppression(outputs, conf_thres=conf_thres, iou_thres=nms_thres)

        sample_metrics += get_batch_statistics(outputs, targets, iou_threshold=iou_thres)

    if len(sample_metrics) == 0:  # No detections over whole validation set.
        print("---- No detections over whole validation set ----")
        return None

    # Concatenate sample statistics
    true_positives, pred_scores, pred_labels = [
        np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
    metrics_output = ap_per_class(
        true_positives, pred_scores, pred_labels, labels)

    print_eval_stats(metrics_output, class_names, verbose)

    return metrics_output


if __name__ == '__main__':
    model = YOLOBody(HYP.anchorIndex, 20, pretrained=True)
    model.load_state_dict(torch.load("yolo_119.pth"))
    model.to(torch.device("cuda:0"))

    data = YoloDataset('./my_yolo_dataset', isTrain=False, transform=VAL_TRANSFORMS)
    dataloader = DataLoader(data, 1, False, num_workers=4, collate_fn=data.collate_fn)

    precision, recall, AP, f1, ap_class = evaluate_model_file(model, dataloader)