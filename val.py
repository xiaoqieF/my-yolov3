from util.metrics import *
from networks.yolov3 import YOLOBody
import torch
from util.dataset import YoloDataset
from util.hyp import HYP
from torch.utils.data import DataLoader
from util.transform import VAL_TRANSFORMS
import tqdm
from util.boxes import BoxDecoder
from util.util import non_max_suppression, xywh2xyxy, load_class_names

def evaluate(model, dataloader, device, plot=False, save_dir='./run'):
    model.eval()
    decoder = BoxDecoder()

    class_names = load_class_names("./my_yolo_dataset/my_data_label.names")

    img_size = HYP.input_shape
    stats = []

    for imgs, targets in tqdm.tqdm(dataloader, desc="Validating"):
        imgs = imgs.to(device)
        targets[:, 2:] = xywh2xyxy(targets[:, 2:])
        targets[:, 2:] *= img_size

        with torch.no_grad():
            outputs = model(imgs)
            outputs = [o.cpu() for o in outputs]
            outputs = decoder.decode(outputs)
            outputs = non_max_suppression(outputs, conf_thres=0.1, iou_thres=0.5)
        stats += get_batch_statistics(outputs, targets)
        
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy   
    
    tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats, names=class_names, plot=plot, save_dir=save_dir)
    print_eval_stats(ap, ap_class, class_names)
    return tp, fp, p, r, f1, ap, ap_class

if __name__ == '__main__':
    device = torch.device("cuda:0")

    model = YOLOBody(HYP.anchorIndex, 20, pretrained=True)
    model.load_state_dict(torch.load("yolo_119.pth"))
    model.to(device)
    model.eval()

    data = YoloDataset('./my_yolo_dataset', isTrain=False, transform=VAL_TRANSFORMS)
    dataloader = DataLoader(data, batch_size=1, shuffle=False, num_workers=1, collate_fn=data.collate_fn)
    
    tp, fp, p, r, f1, ap, ap_class = evaluate(model, dataloader, device, plot=True)
    ap50, ap = ap[:, 0], ap.mean(1)
    # mp: [1] 所有类别的平均precision(最大f1时)
    # mr: [1] 所有类别的平均recall(最大f1时)
    # map50: [1] 所有类别的平均mAP@0.5
    # map: [1] 所有类别的平均mAP@0.5:0.95
    mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
