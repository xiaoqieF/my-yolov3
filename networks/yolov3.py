from .darknet import darknet53

from collections import OrderedDict
import torch
from torch import nn


def conv_bn_relu(in_channels, out_channels, kernel_size):
    pad = (kernel_size - 1) // 2 if kernel_size else 0
    return nn.Sequential(OrderedDict([
        ("conv", nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=pad, bias=False)),
        ("bn", nn.BatchNorm2d(out_channels)),
        ("relu", nn.LeakyReLU(0.1)),
    ]))

def make_last_layers(channels, in_channel, out_channel):
    return nn.Sequential(
        conv_bn_relu(in_channel, channels[0], 1),
        conv_bn_relu(channels[0], channels[1], 3),
        conv_bn_relu(channels[1], channels[0], 1),
        conv_bn_relu(channels[0], channels[1], 3),
        conv_bn_relu(channels[1], channels[0], 1),
        conv_bn_relu(channels[0], channels[1], 3),
        nn.Conv2d(channels[1], out_channel, kernel_size=1, stride=1, padding=0, bias=True)
    )

class YOLOBody(nn.Module):
    def __init__(self, anchors_mask, num_classes, pretrained=False):
        super().__init__()
        
        self.backbone = darknet53()
        if pretrained:
            self.backbone.load_state_dict(torch.load("darknet53.pth"))

        # [64, 128, 256, 512, 1024]
        out_channels = self.backbone.layers_out_filters

        self.last_layer0 = make_last_layers([512, 1024], out_channels[-1], len(anchors_mask[0]) * (num_classes + 5))

        self.last_layer1_conv = conv_bn_relu(512, 256, 1)
        self.last_layer1_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.last_layer1 = make_last_layers([256, 512], out_channels[-2] + 256, len(anchors_mask[1]) * (num_classes + 5))

        self.last_layer2_conv = conv_bn_relu(256, 128, 1)
        self.last_layer2_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.last_layer2 = make_last_layers([128, 256], out_channels[-3] + 128, len(anchors_mask[2]) * (num_classes + 5))

    def forward(self, x):
        x2, x1, x0 = self.backbone(x)
        out0_conv5l = self.last_layer0[:5](x0)
        out0 = self.last_layer0[5:](out0_conv5l)

        x1_in = self.last_layer1_conv(out0_conv5l)
        x1_in = self.last_layer1_upsample(x1_in)
        x1_in = torch.cat([x1_in, x1], 1)
        out1_conv5l = self.last_layer1[:5](x1_in)
        out1 = self.last_layer1[5:](out1_conv5l)

        
        x2_in = self.last_layer2_conv(out1_conv5l)
        x2_in = self.last_layer2_upsample(x2_in)
        x2_in = torch.cat([x2_in, x2], 1)
        out2 = self.last_layer2(x2_in)

        return out0, out1, out2