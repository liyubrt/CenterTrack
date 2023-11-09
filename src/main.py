from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import torch
import torch.utils.data

import _init_paths
from opts import Opts
from model.model import create_model, load_model, save_model
from dataset.dataset_factory import get_dataset
from trainer import Trainer


def get_optimizer(opt, model):
  if opt.optim == 'adam':
    optimizer = torch.optim.Adam([{'params': filter(lambda p: p.requires_grad, model.parameters()),'lr': opt.lr}],
                                  lr=opt.lr,
                                  weight_decay=opt.weight_decay)
  elif opt.optim == 'adamw':
    optimizer = torch.optim.AdamW([{'params': filter(lambda p: p.requires_grad, model.parameters()),'lr': opt.lr}],
                                  lr=opt.lr,
                                  weight_decay=opt.weight_decay)
  elif opt.optim == 'sgd':
    optimizer = torch.optim.SGD([{'params': filter(lambda p: p.requires_grad, model.parameters()), 'lr': opt.lr}], 
                                lr=opt.lr, momentum=0.9, weight_decay=0.0001)
  else:
    assert 0, opt.optim
  return optimizer

def get_scheduler(opt, optimizer):
  assert opt.lr_scheduler["steplr"] + opt.lr_scheduler["cosinelr"] == 1, "only one lr scheduler is allowed"
  if opt.lr_scheduler["steplr"]:
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
                                                step_size=opt.lr_scheduler["steplr_step_size"], 
                                                gamma=opt.lr_scheduler["steplr_gamma"])
  if opt.lr_scheduler["cosinelr"]:
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                                          T_max=opt.lr_scheduler["cosinelr_T_max"], 
                                                          eta_min=opt.lr_scheduler["cosinelr_eta_min"], 
                                                          last_epoch=-1)
  return scheduler

def main(opt):
  torch.manual_seed(opt.seed)
  torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark and not opt.test
  Dataset = get_dataset(opt.dataset)
  opt = Opts().update_dataset_info_and_set_heads(opt, Dataset)
  print(opt)
  if not opt.not_set_cuda_env:
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
  opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')

  opt.logger.write('Creating model...')
  model = create_model(opt.arch, opt.heads, opt.head_convs, opt=opt)
  if opt.load_model != '':
    model = load_model(model, opt.load_model, opt)
  optimizer = get_optimizer(opt, model)
  opt.logger.write(f'Tunable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')
  scheduler = get_scheduler(opt, optimizer)

  trainer = Trainer(opt, model, optimizer)
  trainer.set_device(opt.gpus, opt.chunk_sizes, opt.device)
  
  if opt.val_intervals < opt.num_epochs or opt.test:
    opt.logger.write('Setting up validation data...')
    val_loader = torch.utils.data.DataLoader(
      Dataset(opt, 'val'), batch_size=1, shuffle=False, num_workers=1,
      pin_memory=True)

    if opt.test:
      _, preds = trainer.val(0, val_loader)
      val_loader.dataset.run_eval(preds, opt.save_dir)
      return

  opt.logger.write('Setting up train data...')
  train_loader = torch.utils.data.DataLoader(
      Dataset(opt, 'train'), batch_size=opt.batch_size, shuffle=True,
      num_workers=opt.num_workers, pin_memory=True, drop_last=True
  )

  opt.logger.write('Starting training...')
  start_epoch = 0
  for epoch in range(start_epoch + 1, opt.num_epochs + 1):
    mark = epoch if opt.save_all else 'last'
    log_dict_train, _ = trainer.train(epoch, train_loader)
    log_str = 'epoch: {} | '.format(str(epoch).zfill(3))
    for k, v in log_dict_train.items():
      opt.logger.scalar_summary('train_{}'.format(k), v, epoch)
      log_str += '{} {:8f} | '.format(k, v)
    if opt.val_intervals > 0 and epoch % opt.val_intervals == 0:
      save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(mark)), 
                 epoch, model, optimizer)
      with torch.no_grad():
        log_dict_val, preds = trainer.val(epoch, val_loader)
        if opt.eval_val:
          val_loader.dataset.run_eval(preds, opt.save_dir)
      for k, v in log_dict_val.items():
        opt.logger.scalar_summary('val_{}'.format(k), v, epoch)
        log_str += '{} {:8f} | '.format(k, v)
    if scheduler:
      log_str += f'lr {scheduler.get_last_lr()[0]:.6f} | '
      scheduler.step()
    opt.logger.write(log_str, prefix_time=True)
    if epoch % opt.save_every == 0:
      save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)), 
                 epoch, model, optimizer)
    save_model(os.path.join(opt.save_dir, 'model_last.pth'), 
                epoch, model, optimizer)


if __name__ == '__main__':
  opt = Opts().parse()
  if opt.warm_up > 0:
    # warm up epochs
    full_num_epochs = opt.num_epochs
    opt.num_epochs = opt.warm_up
    main(opt)
    # full tuning epochs
    opt.logger.write()
    opt.load_model = os.path.join(opt.save_dir, 'model_last.pth')
    opt.warm_up = -1
    opt.num_epochs = full_num_epochs
    main(opt)
  else:
    main(opt)
  opt.logger.close()
