from util.metrics import *
from networks.yolov3 import YOLOBody
import torch
from util.dataset import YoloDataset
from util.hyp import HYP
from torch.utils.data import DataLoader
from util.transform import DEFAULT_TRANSFORMS, VAL_TRANSFORMS
import tqdm
from util.boxes import BoxDecoder
from util.util import non_max_suppression
from util.util import bbox_iou, box_iou
from terminaltables import AsciiTable

def xywh2xyxy(x):
    y = x.new(x.shape)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y

def get_batch_statistics(outputs, targets, iou_threshold):
    batch_metrics = []
    for sample_i in range(len(outputs)):
        if outputs[sample_i] is None:
            continue

        output = outputs[sample_i]
        pred_boxes = output[:, :4]
        pred_scores = output[:, 4]
        pred_labels = output[:, -1]

        true_positives = np.zeros(pred_boxes.shape[0])

        # 取出当前 batch 的第 sample_i 个 targets
        annotations = targets[targets[:, 0] == sample_i][:, 1:]
        targets_labels = annotations[:, 0] if len(annotations) else []
        if len(annotations):
            detected_boxes = []
            target_boxes = annotations[:, 1:]

            for pred_i, (pred_box, pred_label) in enumerate(zip(pred_boxes, pred_labels)):
                if len(detected_boxes) == len(annotations):
                    break

                if pred_label not in targets_labels:
                    continue
                
                iou, box_index = bbox_iou(pred_box.unsqueeze(0), target_boxes).max(0)
                # Check if the iou is above the min treshold and i
                if iou >= iou_threshold and box_index not in detected_boxes:
                    true_positives[pred_i] = 1
                    detected_boxes += [box_index]
        batch_metrics.append([true_positives, pred_scores, pred_labels])
    return batch_metrics

def print_eval_stats(metrics_output, class_names):
    if metrics_output is not None:
        precision, recall, AP, f1, ap_class = metrics_output
        # Prints class AP and mean AP
        ap_table = [["Index", "Class", "AP"]]
        for i, c in enumerate(ap_class):
            ap_table += [[c, class_names[c], "%.5f" % AP[i]]]
        print(AsciiTable(ap_table).table)
        print(f"---- mAP {AP.mean():.5f} ----")
    else:
        print("---- mAP not measured (no detections found by model) ----")

if __name__ == '__main__':
    device = torch.device("cuda:0")

    model = YOLOBody(HYP.anchorIndex, 20, pretrained=True)
    model.load_state_dict(torch.load("yolo_119.pth"))
    model.to(device)
    model.eval()

    data = YoloDataset('./my_yolo_dataset', isTrain=False, transform=VAL_TRANSFORMS)
    dataloader = DataLoader(data, batch_size=1, shuffle=False, num_workers=1, collate_fn=data.collate_fn)

    decoder = BoxDecoder()

    class_names = None
    with open("./my_yolo_dataset/my_data_label.names", "r") as fp:
        class_names = fp.read().splitlines()
    class_names = {k: v for k, v in enumerate(class_names)}

    img_size = HYP.input_shape
    stats = []
    iouv = torch.linspace(0.5, 0.95, 10)
    niou = iouv.numel()

    for imgs, targets in tqdm.tqdm(dataloader, desc="Validating"):
        imgs = imgs.to(device)

        with torch.no_grad():
            outputs = model(imgs)
            outputs = [o.cpu() for o in outputs]
            outputs = decoder.decode(outputs)
            outputs = non_max_suppression(outputs, conf_thres=0.1, iou_thres=0.5)

        for si, pred in enumerate(outputs):
            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []

            if len(pred) == 0:
                if nl:
                    stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                continue
            correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool)
            if nl:
                detected = []
                tcls_tensor = labels[:, 0]

                tbox = xywh2xyxy(labels[:, 1:5])
                tbox *= img_size

                for cls in torch.unique(tcls_tensor):
                    ti = (cls == tcls_tensor).nonzero(as_tuple=False).view(-1)
                    pi = (cls == pred[:, 5]).nonzero(as_tuple=False).view(-1)

                    # Search for detections
                    if pi.shape[0]:
                        # Prediction to target ious
                        # predn[pi, :4]: 属于该类的预测框[144, 4]  tbox[ti]: 属于该类的gt框[13, 4]
                        # box_iou: [144, 4] + [13, 4] => [144, 13]  计算属于该类的预测框与属于该类的gt框的iou
                        # .max(1): [144] 选出每个预测框与所有gt box中最大的iou值, i为最大iou值时对应的gt索引
                        ious, i = box_iou(pred[pi, :4], tbox[ti]).max(1)  # best ious, indices

                        # Append detections
                        detected_set = set()  # 这个参数好像没什么用
                        for j in (ious > iouv[0]).nonzero(as_tuple=False):  # j: ious中>0.5的索引 只有iou>=0.5才是TP
                            # 获得检测到的目标
                            d = ti[i[j]]  # detected target
                            if d.item() not in detected_set:
                                detected_set.add(d.item())  # 没什么用
                                detected.append(d) # 将当前检测到的gt框d添加到detected()
                                # iouv为以0.05为步长  0.5-0.95的序列
                                # 从所有TP中获取不同iou阈值下的TP true positive  并在correct中记录下哪个预测框是哪个iou阈值下的TP
                                # correct: [pred_num, 10] = [300, 10]  记录着哪个预测框在哪个iou阈值下是TP
                                correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
                                if len(detected) == nl:  # 如果检测到的目标值等于gt框的个数 就结束
                                    break
            stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy   
    
    tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats, names=class_names)
    ap50, ap = ap[:, 0], ap.mean(1)
        # mp: [1] 所有类别的平均precision(最大f1时)
        # mr: [1] 所有类别的平均recall(最大f1时)
        # map50: [1] 所有类别的平均mAP@0.5
        # map: [1] 所有类别的平均mAP@0.5:0.95
    mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()