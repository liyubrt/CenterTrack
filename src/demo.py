from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os
import sys
import cv2
import json
import copy
import numpy as np
from opts import Opts
from detector import Detector


image_ext = ['jpg', 'jpeg', 'png', 'webp']
video_ext = ['mp4', 'mov', 'avi', 'mkv']
time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge', 'display']

def demo(opt):
  os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
  opt.debug = max(opt.debug, 1)
  detector = Detector(opt)

  if opt.demo == 'webcam' or \
    opt.demo.rsplit('.', 1)[1].lower() in video_ext:
    is_video = True
    # demo on video stream
    opt.logger.write(f'testing on {opt.demo}')
    cam = cv2.VideoCapture(0 if opt.demo == 'webcam' else opt.demo)
  else:
    is_video = False
    # Demo on images sequences
    if os.path.isdir(opt.demo):
      image_names = []
      ls = os.listdir(opt.demo)
      for file_name in sorted(ls):
          ext = file_name[file_name.rfind('.') + 1:].lower()
          if ext in image_ext:
              image_names.append(os.path.join(opt.demo, file_name))
      image_result_dir = os.path.join(f'../results/{opt.exp_id}/{os.path.basename(opt.demo)}')
      os.makedirs(image_result_dir, exist_ok=True)
    else:
      image_names = [opt.demo]

  # Initialize output video
  out = None
  out_name = os.path.basename(opt.demo).rsplit('.', 1)[0] if os.path.isfile(opt.demo) else os.path.basename(opt.demo)
  if opt.save_video:
    save_video_name = '../results/{}/{}_{}.mp4'.format(opt.exp_id, opt.run_id, out_name)
    os.makedirs(os.path.dirname(save_video_name), exist_ok=True)
    opt.logger.write(f'saving video to {save_video_name}')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(save_video_name, fourcc, opt.save_framerate, (opt.video_w, opt.video_h))
  
  if opt.debug < 5:
    detector.pause = False
  cnt = 0
  results = {}

  while True:
    if is_video:
      _, img = cam.read()
      if img is None:
        save_and_exit(opt, out, results, out_name)
    else:
      if cnt < len(image_names):
        img = cv2.imread(image_names[cnt])
      else:
        save_and_exit(opt, out, results, out_name)
    cnt += 1

    # resize the original video for saving video results
    if opt.resize_video:
      img = cv2.resize(img, (opt.video_w, opt.video_h))

    # skip the first X frames of the video
    if cnt < opt.skip_first:
      continue
    
    # cv2.imshow('input', img)

    # track or detect the image.
    ret = detector.run(img)

    # log run time
    time_str = 'frame {} |'.format(cnt)
    for stat in time_stats:
      time_str = time_str + '{} {:.3f}s |'.format(stat, ret[stat])
    opt.logger.write(time_str)

    # results[cnt] is a list of dicts:
    #  [{'bbox': [x1, y1, x2, y2], 'tracking_id': id, 'category_id': c, ...}]
    results[cnt] = ret['results']

    # save debug image to video
    if opt.save_video:
      out.write(ret['generic'])
      if not is_video:
        image_result_path = os.path.join(image_result_dir, os.path.basename(image_names[cnt-1]))
        cv2.imwrite(image_result_path, ret['generic'])
    
    # # esc to quit and finish saving video
    # if cv2.waitKey(1) == 27:
    #   save_and_exit(opt, out, results, out_name)
    #   return 
  
  # save_and_exit(opt, out, results, out_name)
  
  opt.logger.write()
  opt.logger.close()


def save_and_exit(opt, out=None, results=None, out_name=''):
  if opt.save_results and (results is not None):
    save_results_name =  '../results/{}/{}_{}.json'.format(opt.exp_id, opt.run_id, out_name)
    opt.logger.write(f'saving results to {save_results_name}')
    json.dump(_to_list(copy.deepcopy(results)), 
              open(save_results_name, 'w'))
  cv2.destroyAllWindows()
  if opt.save_video and out is not None:
    out.release()
  sys.exit(0)

def _to_list(results):
  for img_id in results:
    for t in range(len(results[img_id])):
      for k in results[img_id][t]:
        if isinstance(results[img_id][t][k], (np.ndarray, np.float32)):
          results[img_id][t][k] = results[img_id][t][k].tolist()
  return results

if __name__ == '__main__':
  opt = Opts().init()
  demo(opt)
