from networks.yolov3 import YOLOBody
from util.loss import compute_loss
from torch import nn, optim
import torch
from torch.utils.data import DataLoader
from util.hyp import HYP
from util.transform import DEFAULT_TRANSFORMS
from util.dataset import YoloDataset
from util.util import non_max_suppression
from util.boxes import BoxDecoder
from val import evaluate_model_file


def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):
    def f(x):
        if x >= warmup_iters:
            return 1
        alpha = float(x) / warmup_iters
        return warmup_factor * (1 - alpha) + alpha
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)

def train_one_epoch(model, epoch, train_loader, optimizer, device, warm_up=False):
    decoder = BoxDecoder()
    model.to(device)
    model.train()

    lr_scheduler = None
    if epoch == 0 and warm_up is True:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(train_loader) - 1)
        lr_scheduler = warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    for i, (imgs, targets) in enumerate(train_loader):
        imgs = imgs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        total_loss = compute_loss(outputs, targets)
        
        total_loss.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        if (i + 1) % 50 == 0:
            print(f"Train: [{i}/{len(train_loader)}], total loss: {total_loss} lr:{optimizer.state_dict()['param_groups'][0]['lr']:.5}")
    torch.save(model.state_dict(), f"yolo_{str(epoch)}.pth")

if __name__ == '__main__':
    model = YOLOBody(HYP.anchorIndex, 20, pretrained=True)

    train_data = YoloDataset('./my_yolo_dataset', isTrain=True, transform=DEFAULT_TRANSFORMS)
    train_dataloader = DataLoader(train_data, 4, True, num_workers=4, collate_fn=train_data.collate_fn)

    val_data = YoloDataset('./my_yolo_dataset', isTrain=False, transform=DEFAULT_TRANSFORMS)
    val_dataloader = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=4, collate_fn=val_data.collate_fn)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.001,
                                momentum=0.9, weight_decay=0.0005)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=5,
                                                   gamma=0.5)
    for epoch in range(60):
        train_one_epoch(model, epoch, train_dataloader, optimizer, device=torch.device("cuda:0"), warm_up=True)
        lr_scheduler.step()

        evaluate_model_file(model, val_dataloader)