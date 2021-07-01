from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import json
import cv2
import numpy as np
import time
from progress.bar import Bar
import torch
import copy

from utils.utils import AverageMeter
from detector import Detector
from dataloaders.dataset_factory import get_dataset


class PrefetchDataset(torch.utils.data.Dataset):
  def __init__(self, opt, dataset, pre_process_func):
    self.images = dataset.images
    self.load_image_func = dataset.coco.loadImgs
    self.img_dir = dataset.img_dir
    self.pre_process_func = pre_process_func
    self.get_default_calib = dataset.get_default_calib
    self.opt = opt
  
  def __getitem__(self, index):
    img_id = self.images[index]
    img_info = self.load_image_func(ids=[img_id])[0]
    img_path = os.path.join(self.img_dir, img_info['file_name'])
    image = cv2.imread(img_path)
    images, meta = {}, {}
    for scale in [1.0]:
      input_meta = {}
      calib = img_info['calib'] if 'calib' in img_info \
        else self.get_default_calib(image.shape[1], image.shape[0])
      input_meta['calib'] = calib
      images[scale], meta[scale] = self.pre_process_func(
        image, scale, input_meta)
    ret = {'images': images, 'image': image, 'meta': meta}
    if 'frame_id' in img_info and img_info['frame_id'] == 1:
      ret['is_first_frame'] = 1
      ret['video_id'] = img_info['video_id']
    return img_id, ret, img_path

  def __len__(self):
    return len(self.images)

def save_img(img, img_path, out_path, results):
  img = cv2.imread(img_path)
  for rs in results:
    bbox = rs['bbox']
    ct_x = int(bbox[0] + (bbox[2] - bbox[0])/2)
    ct_y = int(bbox[1] + (bbox[3] - bbox[1])/2)
    img = cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
    img = cv2.putText(img, str(rs['dep'][0]), (ct_x, ct_y), 0, 0.5, (255, 0, 0))
    img = cv2.circle(img, (ct_x, ct_y), radius=1, color=(0, 0, 255), thickness=-1)
  
  cv2.imwrite(out_path, img)

def test(opt):
  Dataset = get_dataset('custom')
  split = 'test'
  dataset = Dataset(opt.test_meta_filenames, opt, split)
  detector = Detector(opt)
  
  load_results = {}

  data_loader = torch.utils.data.DataLoader(
    PrefetchDataset(opt, dataset, detector.pre_process), 
    batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

  results = {}
  num_iters = len(data_loader)
  bar = Bar('{}'.format(opt.exp_id), max=num_iters)
  time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge', 'track']
  avg_time_stats = {t: AverageMeter() for t in time_stats}

  for ind, (img_id, pre_processed_images, img_paths) in enumerate(data_loader):
    if ind >= num_iters:
      break
    if ('is_first_frame' in pre_processed_images):
      if '{}'.format(int(img_id.numpy().astype(np.int32)[0])) in load_results:
        pre_processed_images['meta']['pre_dets'] = \
          load_results['{}'.format(int(img_id.numpy().astype(np.int32)[0]))]
      else:
        print()
        print('No pre_dets for', int(img_id.numpy().astype(np.int32)[0]), 
          '. Use empty initialization.')
        pre_processed_images['meta']['pre_dets'] = []
      detector.reset_tracking()
      print('Start tracking video', int(pre_processed_images['video_id']))
    
    ret = detector.run(pre_processed_images)
    results[int(img_id.numpy().astype(np.int32)[0])] = ret['results']

    out_path = img_paths[0].replace("data", "algorithms/" + opt.algorithm + "/results/test/")
    save_img(pre_processed_images["image"], img_paths[0], out_path, ret['results'])
    
    Bar.suffix = '[{0}/{1}]|Tot: {total:} |ETA: {eta:} '.format(
                   ind, num_iters, total=bar.elapsed_td, eta=bar.eta_td)
    for t in avg_time_stats:
      avg_time_stats[t].update(ret[t])
      Bar.suffix = Bar.suffix + '|{} {tm.val:.3f}s ({tm.avg:.3f}s) '.format(
        t, tm = avg_time_stats[t])
    bar.next()

  bar.finish()
  json.dump(_to_list(copy.deepcopy(results)), 
            open("algorithms/" + opt.algorithm + "/results/logs/test_" + opt.exp_name + "_" + opt.exp_id + '.json', 'w'))
  dataset.run_eval(results, opt.save_dir)

def _to_list(results):
  for img_id in results:
    for t in range(len(results[img_id])):
      for k in results[img_id][t]:
        if isinstance(results[img_id][t][k], (np.ndarray, np.float32)):
          results[img_id][t][k] = results[img_id][t][k].tolist()
  return results

if __name__ == '__main__':
  opt = opts().parse()
  test(opt)
