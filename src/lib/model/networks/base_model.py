from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch import nn

def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

class BaseModel(nn.Module):
    def __init__(self, heads, head_convs, num_stacks, last_channel, opt=None):
        super(BaseModel, self).__init__()
        if opt is not None and opt.head_kernel != 3:
          print('Using head kernel:', opt.head_kernel)
          head_kernel = opt.head_kernel
        else:
          head_kernel = 3
        self.num_stacks = num_stacks
        self.heads = heads  # {'hm': 80, 'reg': 2, 'wh': 2, 'tracking': 2, 'ltrb_amodal': 4}
        for head in self.heads:
            classes = self.heads[head]
            head_conv = head_convs[head]  # head_convs: {'hm': [256], 'reg': [256], 'wh': [256], 'tracking': [256]}
            if len(head_conv) > 0:
              out = nn.Conv2d(head_conv[-1], classes, 
                    kernel_size=1, stride=1, padding=0, bias=True)
              conv = nn.Conv2d(last_channel, head_conv[0],
                               kernel_size=head_kernel, 
                               padding=head_kernel // 2, bias=True)
              convs = [conv]
              for k in range(1, len(head_conv)):
                  convs.append(nn.Conv2d(head_conv[k - 1], head_conv[k], 
                               kernel_size=1, bias=True))
              if len(convs) == 1:
                fc = nn.Sequential(conv, nn.ReLU(inplace=True), out)
              elif len(convs) == 2:
                fc = nn.Sequential(
                  convs[0], nn.ReLU(inplace=True), 
                  convs[1], nn.ReLU(inplace=True), out)
              elif len(convs) == 3:
                fc = nn.Sequential(
                    convs[0], nn.ReLU(inplace=True), 
                    convs[1], nn.ReLU(inplace=True), 
                    convs[2], nn.ReLU(inplace=True), out)
              elif len(convs) == 4:
                fc = nn.Sequential(
                    convs[0], nn.ReLU(inplace=True), 
                    convs[1], nn.ReLU(inplace=True), 
                    convs[2], nn.ReLU(inplace=True), 
                    convs[3], nn.ReLU(inplace=True), out)
              if 'hm' in head:
                fc[-1].bias.data.fill_(opt.prior_bias)
              else:
                fill_fc_weights(fc)
            else:
              fc = nn.Conv2d(last_channel, classes, 
                  kernel_size=1, stride=1, padding=0, bias=True)
              if 'hm' in head:
                fc.bias.data.fill_(opt.prior_bias)
              else:
                fill_fc_weights(fc)
            self.__setattr__(head, fc)

    def img2feats(self, x):
      raise NotImplementedError
    
    def imgpre2feats(self, x, pre_img=None, pre_hm=None):
      raise NotImplementedError

    def forward(self, x, pre_img=None, pre_hm=None):  # x: Nx3x512x1024, pre_img: Nx3x512x1024, pre_hm: Nx1x512x1024
      if (pre_hm is not None) or (pre_img is not None): # entered
        feats = self.imgpre2feats(x, pre_img, pre_hm)  # [Nx64x128x256]
      else:
        feats = self.img2feats(x)
      out = []
      if self.opt.model_output_list:  # no enter
        for s in range(self.num_stacks):
          z = []
          for head in sorted(self.heads):
              z.append(self.__getattr__(head)(feats[s]))
          out.append(z)
      else:  # entered
        for s in range(self.num_stacks):  # self.num_stacks: 1
          z = {}
          for head in self.heads:  # self.heads: {'hm': 80, 'reg': 2, 'wh': 2, 'tracking': 2, 'ltrb_amodal': 4}
              z[head] = self.__getattr__(head)(feats[s])
          out.append(z)
      return out  # [{'hm': Nx80x128x256, 'reg': Nx2x128x256, 'wh': Nx2x128x256, 'tracking': Nx2x128x256, 'ltrb_amodal': Nx4x128x256}]
