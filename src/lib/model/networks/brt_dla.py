from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
import logging
import numpy as np
from os.path import join

import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

from .base_model import BaseModel
from .brtresnetpyramid_lite12 import BrtResnetPyramidLite12

try:
    # from .DCNv2.dcn_v2 import DCN
    import mmcv
    # from mmcv.ops.deform_conv import DeformConv2d as DCN
    from mmcv.ops.modulated_deform_conv import ModulatedDeformConv2dPack as DCN
except:
    print('import DCN failed')
    DCN = None


BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)


def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


def fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = \
                (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :]


class Conv(nn.Module):
    def __init__(self, chi, cho):
        super(Conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(chi, cho, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(cho, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True))
    
    def forward(self, x):
        return self.conv(x)


class GlobalConv(nn.Module):
    def __init__(self, chi, cho, k=7, d=1):
        super(GlobalConv, self).__init__()
        gcl = nn.Sequential(
            nn.Conv2d(chi, cho, kernel_size=(k, 1), stride=1, bias=False, 
                                dilation=d, padding=(d * (k // 2), 0)),
            nn.Conv2d(cho, cho, kernel_size=(1, k), stride=1, bias=False, 
                                dilation=d, padding=(0, d * (k // 2))))
        gcr = nn.Sequential(
            nn.Conv2d(chi, cho, kernel_size=(1, k), stride=1, bias=False, 
                                dilation=d, padding=(0, d * (k // 2))),
            nn.Conv2d(cho, cho, kernel_size=(k, 1), stride=1, bias=False, 
                                dilation=d, padding=(d * (k // 2), 0)))
        fill_fc_weights(gcl)
        fill_fc_weights(gcr)
        self.gcl = gcl
        self.gcr = gcr
        self.act = nn.Sequential(
            nn.BatchNorm2d(cho, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.gcl(x) + self.gcr(x)
        x = self.act(x)
        return x


class DeformConv(nn.Module):
    def __init__(self, chi, cho):
        super(DeformConv, self).__init__()
        self.actf = nn.Sequential(
            nn.BatchNorm2d(cho, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True)
        )
        self.conv = DCN(chi, cho, kernel_size=(3,3), stride=1, padding=1, dilation=1, deformable_groups=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.actf(x)
        return x

class IDAUp(nn.Module):
    def __init__(self, o, channels, up_f, node_type=(DeformConv, DeformConv)):
        super(IDAUp, self).__init__()
        for i in range(1, len(channels)):
            c = channels[i]
            f = int(up_f[i])  
            proj = node_type[0](c, o)
            node = node_type[1](o, o)
     
            up = nn.ConvTranspose2d(o, o, f * 2, stride=f, 
                                    padding=f // 2, output_padding=0,
                                    groups=o, bias=False)
            fill_up_weights(up)

            setattr(self, 'proj_' + str(i), proj)
            setattr(self, 'up_' + str(i), up)
            setattr(self, 'node_' + str(i), node)
                 
        
    def forward(self, layers, startp, endp):
        for i in range(startp + 1, endp):
            upsample = getattr(self, 'up_' + str(i - startp))
            project = getattr(self, 'proj_' + str(i - startp))
            layers[i] = upsample(project(layers[i]))  # upsample by 2
            node = getattr(self, 'node_' + str(i - startp))
            layers[i] = node(layers[i] + layers[i - 1])



class DLAUp(nn.Module):
    def __init__(self, startp, channels, scales, in_channels=None, 
                 node_type=DeformConv):
        super(DLAUp, self).__init__()
        self.startp = startp
        if in_channels is None:
            in_channels = channels
        self.channels = channels
        channels = list(channels)
        scales = np.array(scales, dtype=int)
        for i in range(len(channels) - 1):
            j = -i - 2
            setattr(self, 'ida_{}'.format(i),
                    IDAUp(channels[j], in_channels[j:],
                          scales[j:] // scales[j],
                          node_type=node_type))
            scales[j + 1:] = scales[j]
            in_channels[j + 1:] = [channels[j] for _ in channels[j + 1:]]

    def forward(self, layers):  # [N, 64, 128, 256], [N, 64, 128, 256], [N, 128, 64, 128], [N, 256, 32, 64], [N, 256, 16, 32]
        out = [layers[-1]] # start with 32
        for i in range(len(layers) - self.startp - 1):
            ida = getattr(self, 'ida_{}'.format(i))
            ida(layers, len(layers) -i - 2, len(layers))
            out.insert(0, layers[-1])
        return out


DLA_NODE = {
    'dcn': (DeformConv, DeformConv),
    'gcn': (Conv, GlobalConv),
    'conv': (Conv, Conv),
}

class BRTDLASeg(BaseModel):
    def __init__(self, num_layers, heads, head_convs, opt):
        super(BRTDLASeg, self).__init__(
            heads, head_convs, 1, 64 if num_layers == 34 else 128, opt=opt)
        down_ratio=4
        self.opt = opt
        self.node_type = DLA_NODE[opt.dla_node]  # opt.dla_node: dcn
        print('Using node type:', self.node_type)
        # self.first_level = int(np.log2(down_ratio))  # 2
        self.first_level = 1
        self.last_level = 4  # self.base.channels - 1 ?
        # self.base = globals()['dla{}'.format(num_layers)](
        #     pretrained=(opt.load_model == ''), opt=opt)
        params = {
            "num_classes": 7,
            "input_dims": 3,
            "model_params": {
                "num_block_layers": 2,
                "widening_factor": 2,
                "upsample_mode": "nearest",
                "bias": True, 
                "activation": "relu",
            },
            "pre_img": True,
            "pre_hm": True,
        }
        self.base = BrtResnetPyramidLite12(params)

        channels = self.base.channels
        scales = [2 ** i for i in range(len(channels[self.first_level:]))]
        self.dla_up = DLAUp(
            self.first_level, channels[self.first_level:], scales,
            node_type=self.node_type)
        out_channel = channels[self.first_level]

        self.ida_up = IDAUp(
            out_channel, channels[self.first_level:self.last_level], 
            [2 ** i for i in range(self.last_level - self.first_level)],
            node_type=self.node_type)
        

    def img2feats(self, x):
        x = self.base(x)
        x = self.dla_up(x)

        y = []
        for i in range(self.last_level - self.first_level):
            y.append(x[i].clone())
        self.ida_up(y, 0, len(y))

        return [y[-1]]

    def imgpre2feats(self, x, pre_img=None, pre_hm=None):  # x: Nx3x512x1024, pre_img: Nx3x512x1024, pre_hm: Nx1x512x1024
        x = self.base(x, pre_img, pre_hm)  # [N, 64, 128, 256], [N, 64, 128, 256], [N, 128, 64, 128], [N, 256, 32, 64], [N, 256, 16, 32]
        x = self.dla_up(list(x))  # [N, 64, 128, 256], [N, 128, 64, 128], [N, 256, 32, 64], [N, 256, 16, 32]

        y = []  # after append: [N, 64, 128, 256], [N, 128, 64, 128], [N, 256, 32, 64]
        for i in range(self.last_level - self.first_level):
            y.append(x[i].clone())
        self.ida_up(y, 0, len(y))  # y: [N, 64, 128, 256], [N, 64, 128, 256], [N, 64, 128, 256]

        return [y[-1]]  # [N, 64, 128, 256]
