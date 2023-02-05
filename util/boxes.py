import torch
from .hyp import HYP


class BoxDecoder():
    def __init__(self):
        self.grids = None

    def decode(self, predictions):
        """
        将网络预测参数转换为输出
        Args:
            predictions(List[Tensor]): 三个预测层, shape 为 [bs, (class_num + 5) * 3, h, w]
        """
        if self.grids is None:
            self.grids = self._make_grid(predictions)

        outputs = []
        for i, prediction in enumerate(predictions):
            b, c, h, w = prediction.shape
            stride = HYP.input_shape / h

            # [bs, 3, h, w, class_num + 5]
            prediction = prediction.view(b, len(HYP.anchor[i]), -1, h, w)\
                .permute(0, 1, 3, 4, 2).contiguous()
            # tx, ty, conf 以及每个类别的参数都需要做 sigmoid 操作
            prediction[..., [0, 1, 4]] = prediction[..., [0, 1, 4]].sigmoid()
            prediction[..., 5:] = prediction[..., 5:].sigmoid()
            # 乘以 stride 之后就是原图像的尺度
            prediction[..., :2] = (prediction[..., :2] + self.grids[i][..., :2]) * stride
            prediction[..., 2:4] = (torch.exp(prediction[..., 2:4]) * self.grids[i][..., 2:4]) * stride
            prediction = prediction.view(b, -1, 25)
            outputs.append(prediction)
        return torch.cat(outputs, 1)
    
    def _make_grid(self, predictions):
        """
        构建每个预测层的 anchor grid, 
        Return:
            grids(List[Tensor[bs, 3, w, h, 4]]), 每层每个像素有 3 个 anchor, 每个 anchor 的
            4 个参数分别为 grid_x, grid_y, anchor_w, anchor_h, 配合网络预测的调整参数 tx, ty, tw, th
            得到最终的预测框， 计算方法就是论文的公式：
                                bx = sigmoid(tx) + grid_x
                                by = sigmoid(ty) + grid_y
                                bw = anchor_w * exp(tw)
                                bh = anchor_h * exp(th)
            注意， 计算得到的 bx, by, bw, bh 是在当前特征图尺度上的坐标，还需要乘以下采样倍率 stride 得到最终坐标
        """
        grids = []
        for i, prediction in enumerate(predictions):
            device = prediction.device
            b, c, h, w = prediction.shape
            stride = HYP.input_shape / h
            anchor_scaled = torch.tensor(HYP.anchor[i], device=device) / stride
            grid = torch.ones((b, len(HYP.anchor[i]), h, w, 4), device=device)
            gridY, gridX = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
            grid[..., 0] *= gridX.to(device).unsqueeze(0)   # x
            grid[..., 1] *= gridY.to(device).unsqueeze(0)   # y
            grid[..., 2] *= anchor_scaled[:, 0].view(1, len(HYP.anchor[i]), 1, 1)  # w
            grid[..., 3] *= anchor_scaled[:, 1].view(1, len(HYP.anchor[i]), 1, 1)  # h
            grids.append(grid)
        return grids