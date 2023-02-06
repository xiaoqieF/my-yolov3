from util.dataset import YoloDataset
from util.util import non_max_suppression
from util.boxes import BoxDecoder
from networks.yolov3 import YOLOBody
from util.hyp import HYP
import torch
from torch.utils.data import DataLoader
from util.transform import DEFAULT_TRANSFORMS
from util.draw_boxes_utils import draw_objs
import torchvision
import matplotlib.pyplot as plt


if __name__ == '__main__':
    device = torch.device("cuda:0")
    model = YOLOBody(HYP.anchorIndex, 20, pretrained=True)
    model.load_state_dict(torch.load("best_epoch_weights.pth"))
    model.to(device)
    model.eval()

    data = YoloDataset('./my_yolo_dataset', isTrain=False, transform=DEFAULT_TRANSFORMS)
    dataloader = DataLoader(data, 1, True, num_workers=4, collate_fn=data.collate_fn)
    decoder = BoxDecoder()
    with torch.no_grad():
        for imgs, targets in dataloader:
            imgs = imgs.to(device)
            targets = targets.to(device)
            
            outputs = model(imgs)
            outputs = decoder.decode(outputs)
            predictions = non_max_suppression(outputs)[0]
            predictions = predictions.cpu()
            print(f"predictions: {predictions}")
            print(f"targets: {targets}")
            img = draw_objs(torchvision.transforms.ToPILImage()(imgs.squeeze(0)), predictions[:, :4], predictions[:, 4])
            plt.imshow(img)
            plt.show()