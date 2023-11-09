from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torchvision.models as models
import torch
import torch.nn as nn
import os
import pandas as pd

from .networks.brt_dla import BRTDLASeg
from .networks.dla import DLASeg
from .networks.resdcn import PoseResDCN
from .networks.resnet import PoseResNet
from .networks.dlav0 import DLASegv0
from .networks.generic_network import GenericNetwork

_network_factory = {
  'resdcn': PoseResDCN,
  'dla': DLASeg,
  'brtdla': BRTDLASeg,
  'res': PoseResNet,
  'dlav0': DLASegv0,
  'generic': GenericNetwork
}

def create_model(arch, head, head_convs, opt=None):  # head {'hm': 80, 'reg': 2, 'wh': 2, 'tracking': 2}, head_convs {'hm': [256], 'reg': [256], 'wh': [256], 'tracking': [256]}
  num_layers = int(arch[arch.find('_') + 1:]) if '_' in arch else 0  # 34
  arch = arch[:arch.find('_')] if '_' in arch else arch  # dla
  model_class = _network_factory[arch]  # DLASeg
  model = model_class(num_layers, heads=head, head_convs=head_convs, opt=opt)  # DLASeg
  return model

def load_model(model, model_path, opt, optimizer=None):
  if not opt.test and opt.warm_up >= 0:
    # load tracking weights
    model_path_tracking = os.path.join(opt.root_dir, 'models/coco_tracking.pth')
    checkpoint = torch.load(model_path_tracking, map_location=lambda storage, loc: storage)
    print('loaded {}, epoch {}'.format(model_path_tracking, checkpoint['epoch']))
    state_dict_ = checkpoint['state_dict']
    # load BRT weights
    if 'brt' in opt.arch:
      checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
      print('loaded {}'.format(model_path))
      state_dict2 = checkpoint['state_dict']
      state_dict2 = {'base.'+k:v for k,v in state_dict2.items()}
      state_dict_.update(state_dict2)
      if opt.freeze_encoder:
        seg_model_weights_file = os.path.join(opt.root_dir, 'models/brt_lite12_weights.csv')
        seg_model_weights = list(state_dict2.keys())
        df = pd.DataFrame(data={'weight_name': seg_model_weights})
        df.to_csv(seg_model_weights_file, index=False)
        print(f'saved seg model weight names to {seg_model_weights_file}')
  else:
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    print('loaded {}, epoch {}'.format(model_path, checkpoint['epoch']))
    state_dict_ = checkpoint['state_dict']
    if opt.freeze_encoder:
      seg_model_weights_file = os.path.join(opt.root_dir, 'models/brt_lite12_weights.csv')
      assert os.path.isfile(seg_model_weights_file), f"{seg_model_weights_file} doesn't exist"
      seg_model_weights = pd.read_csv(seg_model_weights_file).weight_name.to_list()
      print(f'read seg model weight names from {seg_model_weights_file}')
  
  start_epoch = 0
  state_dict = {}
   
  # convert data_parallal to model
  for k in state_dict_:
    if k.startswith('module') and not k.startswith('module_list'):
      state_dict[k[7:]] = state_dict_[k]
    # adapt for DLA from MMCV
    elif k in [
      'dla_up.ida_0.proj_1.conv.conv_offset_mask.weight',
      'dla_up.ida_0.proj_1.conv.conv_offset_mask.bias',
      'dla_up.ida_0.node_1.conv.conv_offset_mask.weight',
      'dla_up.ida_0.node_1.conv.conv_offset_mask.bias',
      'dla_up.ida_1.proj_1.conv.conv_offset_mask.weight',
      'dla_up.ida_1.proj_1.conv.conv_offset_mask.bias',
      'dla_up.ida_1.node_1.conv.conv_offset_mask.weight',
      'dla_up.ida_1.node_1.conv.conv_offset_mask.bias',
      'dla_up.ida_1.proj_2.conv.conv_offset_mask.weight',
      'dla_up.ida_1.proj_2.conv.conv_offset_mask.bias',
      'dla_up.ida_1.node_2.conv.conv_offset_mask.weight',
      'dla_up.ida_1.node_2.conv.conv_offset_mask.bias',
      'dla_up.ida_2.proj_1.conv.conv_offset_mask.weight',
      'dla_up.ida_2.proj_1.conv.conv_offset_mask.bias',
      'dla_up.ida_2.node_1.conv.conv_offset_mask.weight',
      'dla_up.ida_2.node_1.conv.conv_offset_mask.bias',
      'dla_up.ida_2.proj_2.conv.conv_offset_mask.weight',
      'dla_up.ida_2.proj_2.conv.conv_offset_mask.bias',
      'dla_up.ida_2.node_2.conv.conv_offset_mask.weight',
      'dla_up.ida_2.node_2.conv.conv_offset_mask.bias',
      'dla_up.ida_2.proj_3.conv.conv_offset_mask.weight',
      'dla_up.ida_2.proj_3.conv.conv_offset_mask.bias',
      'dla_up.ida_2.node_3.conv.conv_offset_mask.weight',
      'dla_up.ida_2.node_3.conv.conv_offset_mask.bias',
      'ida_up.proj_1.conv.conv_offset_mask.weight',
      'ida_up.proj_1.conv.conv_offset_mask.bias',
      'ida_up.node_1.conv.conv_offset_mask.weight',
      'ida_up.node_1.conv.conv_offset_mask.bias',
      'ida_up.proj_2.conv.conv_offset_mask.weight',
      'ida_up.proj_2.conv.conv_offset_mask.bias',
      'ida_up.node_2.conv.conv_offset_mask.weight',
      'ida_up.node_2.conv.conv_offset_mask.bias',
    ]:
      splits = k.rsplit('.', 1)
      k_ = splits[0][:-5] + '.' + splits[1]
      state_dict[k_] = state_dict_[k]
    else:
      state_dict[k] = state_dict_[k]
  model_state_dict = model.state_dict()

  # check loaded parameters and created model parameters
  unmatched_weights = []
  for k in state_dict:
    if k in model_state_dict:
      if (state_dict[k].shape != model_state_dict[k].shape) or \
        (opt.reset_hm and k.startswith('hm') and (state_dict[k].shape[0] in [80, 1])):
        if opt.reuse_hm:
          print('Reusing parameter {}, required shape{}, '\
                'loaded shape{}.'.format(
            k, model_state_dict[k].shape, state_dict[k].shape))
          if state_dict[k].shape[0] < state_dict[k].shape[0]:
            model_state_dict[k][:state_dict[k].shape[0]] = state_dict[k]
          else:
            model_state_dict[k] = state_dict[k][:model_state_dict[k].shape[0]]
          state_dict[k] = model_state_dict[k]
        else:
          print('Skip loading parameter {}, required shape{}, '\
                'loaded shape{}.'.format(
            k, model_state_dict[k].shape, state_dict[k].shape))
          state_dict[k] = model_state_dict[k]
        unmatched_weights.append(k)
    else:
      print('Drop parameter {}.'.format(k))
  for k in model_state_dict:
    if not (k in state_dict):
      print('No param {}.'.format(k))
      state_dict[k] = model_state_dict[k]
      unmatched_weights.append(k)
  missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
  if len(missing_keys) > 0 or len(unexpected_keys) > 0:
    print('Num missing keys: {} and Num unexpected keys: {} while loading model'
                    .format(len(missing_keys), len(unexpected_keys)))

  # warmup: disable gradient calculation for matched weights
  if opt.warm_up > 0:
    for name, param in model.named_parameters():
      if not name in unmatched_weights:
        param.requires_grad = False
  # freeze layers in brt seg model
  if opt.freeze_encoder:
    for name, param in model.named_parameters():
      if name in seg_model_weights:
        param.requires_grad = False

  # # resume optimizer parameters
  # if optimizer is not None and opt.resume:
  #   if 'optimizer' in checkpoint:
  #     # optimizer.load_state_dict(checkpoint['optimizer'])
  #     start_epoch = checkpoint['epoch']
  #     start_lr = opt.lr
  #     for step in opt.lr_step:
  #       if start_epoch >= step:
  #         start_lr *= 0.1
  #     for param_group in optimizer.param_groups:
  #       param_group['lr'] = start_lr
  #     print('Resumed optimizer with start lr', start_lr)
  #   else:
  #     print('No optimizer parameters in checkpoint.')
  # if optimizer is not None:
  #   return model, optimizer, start_epoch
  # else:
  return model

def save_model(path, epoch, model, optimizer=None):
  if isinstance(model, torch.nn.DataParallel):
    state_dict = model.module.state_dict()
  else:
    state_dict = model.state_dict()
  data = {'epoch': epoch,
          'state_dict': state_dict}
  if not (optimizer is None):
    data['optimizer'] = optimizer.state_dict()
  torch.save(data, path)

