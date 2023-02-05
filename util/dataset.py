import torch
from PIL import Image
from torch.utils.data.dataset import Dataset
import os
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader

class YoloDataset(Dataset):
    def __init__(self, data_root, img_size=(416, 416), isTrain=True, transform=None):
        self.image_size = img_size
        if isTrain:
            self.images_root = os.path.join(data_root, "train", "images")
        else:
            self.images_root = os.path.join(data_root, "val", "images")
        self.image_paths = [os.path.join(self.images_root, image_name) for image_name in os.listdir(self.images_root)] 
        self.annotations_paths = []
        for image_path in self.image_paths:
            image_name = os.path.split(image_path)[-1]
            label_name = os.path.splitext(image_name)[0] + '.txt'
            self.annotations_paths.append(os.path.join(os.path.dirname(self.images_root), 'labels', label_name))
        self.transform = transform        

    def __getitem__(self, index):
        """
        Return:
            image_data: Tensor[3, image_size, image_size]
            boxes: Tensor[N, 5]: [label, x, y, w, h]
        """
        image = np.array(Image.open(self.image_paths[index]), dtype=np.uint8)
        targets = np.loadtxt(self.annotations_paths[index]).reshape(-1, 5)
        if self.transform:
            image, targets = self.transform((image, targets))
        return image, targets

    def __len__(self):
        return len(self.image_paths)

    def collate_fn(self, batch):
        """
        Returns:
            imgs(Tensor): shape is [batch_size, 3, img_size, img_size]
            targets(Tensor[num_boxes, 6]): each boxes is [index, label, x, y, w, h]
        """
        imgs, targets = list(zip(*batch))
        imgs = torch.stack([resize(img, self.image_size) for img in imgs])

        # add sample index to targets
        # to identify which image the boxes belong to
        for i, boxes in enumerate(targets):
            boxes[:, 0] = i
        targets = torch.cat(targets, 0)

        return imgs, targets

def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image

if __name__ == '__main__':
    from transform import DEFAULT_TRANSFORMS
    data = YoloDataset('./my_yolo_dataset', isTrain=False, transform=DEFAULT_TRANSFORMS)

    dataloader = DataLoader(data, 4, shuffle=False, num_workers=4, collate_fn=data.collate_fn)
    for img, target in dataloader:
        print(img.shape)
        print(target)
        break