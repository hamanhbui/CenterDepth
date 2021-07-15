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

def get_model_scores(pred_boxes):
	"""Creates a dictionary of from model_scores to image ids.
	Args:
		pred_boxes (dict): dict of dicts of 'boxes' and 'scores'
	Returns:
		dict: keys are model_scores and values are image ids (usually filenames)
	"""
	model_score={}
	for img_id, val in pred_boxes.items():
		for score in val['score']:
			if score not in model_score.keys():
				model_score[score]=[img_id]
			else:
				model_score[score].append(img_id)
	return model_score

def calc_iou( gt_bbox, pred_bbox):
	'''
	This function takes the predicted bounding box and ground truth bounding box and 
	return the IoU ratio
	'''
	x_topleft_gt, y_topleft_gt, x_bottomright_gt, y_bottomright_gt= gt_bbox
	x_topleft_p, y_topleft_p, x_bottomright_p, y_bottomright_p= pred_bbox
	
	if (x_topleft_gt > x_bottomright_gt) or (y_topleft_gt> y_bottomright_gt):
		raise AssertionError("Ground Truth Bounding Box is not correct")
	if (x_topleft_p > x_bottomright_p) or (y_topleft_p> y_bottomright_p):
		raise AssertionError("Predicted Bounding Box is not correct",x_topleft_p, x_bottomright_p,y_topleft_p,y_bottomright_gt)
		 
	#if the GT bbox and predcited BBox do not overlap then iou=0
	if(x_bottomright_gt< x_topleft_p):
		# If bottom right of x-coordinate  GT  bbox is less than or above the top left of x coordinate of  the predicted BBox
		
		return 0.0
	if(y_bottomright_gt< y_topleft_p):  # If bottom right of y-coordinate  GT  bbox is less than or above the top left of y coordinate of  the predicted BBox
		
		return 0.0
	if(x_topleft_gt> x_bottomright_p): # If bottom right of x-coordinate  GT  bbox is greater than or below the bottom right  of x coordinate of  the predcited BBox
		
		return 0.0
	if(y_topleft_gt> y_bottomright_p): # If bottom right of y-coordinate  GT  bbox is greater than or below the bottom right  of y coordinate of  the predcited BBox
		
		return 0.0
	
	GT_bbox_area = (x_bottomright_gt -  x_topleft_gt + 1) * (  y_bottomright_gt -y_topleft_gt + 1)
	Pred_bbox_area =(x_bottomright_p - x_topleft_p + 1 ) * ( y_bottomright_p -y_topleft_p + 1)
	
	x_top_left =np.max([x_topleft_gt, x_topleft_p])
	y_top_left = np.max([y_topleft_gt, y_topleft_p])
	x_bottom_right = np.min([x_bottomright_gt, x_bottomright_p])
	y_bottom_right = np.min([y_bottomright_gt, y_bottomright_p])
	
	intersection_area = (x_bottom_right- x_top_left + 1) * (y_bottom_right-y_top_left  + 1)
	
	union_area = (GT_bbox_area + Pred_bbox_area - intersection_area)
   
	return intersection_area/union_area

def calc_precision_recall(image_results):
	"""Calculates precision and recall from the set of images
	Args:
		img_results (dict): dictionary formatted like:
			{
				'img_id1': {'true_pos': int, 'false_pos': int, 'false_neg': int},
				'img_id2': ...
				...
			}
	Returns:
		tuple: of floats of (precision, recall)
	"""
	true_positive=0
	false_positive=0
	false_negative=0
	for img_id, res in image_results.items():
		true_positive +=res['true_positive']
		false_positive += res['false_positive']
		false_negative += res['false_negative']
		try:
			precision = true_positive/(true_positive+ false_positive)
		except ZeroDivisionError:
			precision=0.0
		try:
			recall = true_positive/(true_positive + false_negative)
		except ZeroDivisionError:
			recall=0.0
	return (precision, recall)

def get_single_image_results(gt_boxes, pred_boxes, iou_thr):
	"""Calculates number of true_pos, false_pos, false_neg from single batch of boxes.
	Args:
		gt_boxes (list of list of floats): list of locations of ground truth
			objects as [xmin, ymin, xmax, ymax]
		pred_boxes (dict): dict of dicts of 'boxes' (formatted like `gt_boxes`)
			and 'scores'
		iou_thr (float): value of IoU to consider as threshold for a
			true prediction.
	Returns:
		dict: true positives (int), false positives (int), false negatives (int)
	"""
	all_pred_indices= range(len(pred_boxes))
	all_gt_indices=range(len(gt_boxes))
	if len(all_pred_indices)==0:
		tp=0
		fp=0
		fn=len(gt_boxes)
		return {'true_positive':tp, 'false_positive':fp, 'false_negative':fn}
	if len(all_gt_indices)==0:
		tp=0
		fp=len(pred_boxes)
		fn=0
		return {'true_positive':tp, 'false_positive':fp, 'false_negative':fn}
 
	gt_idx_thr=[]
	pred_idx_thr=[]
	ious=[]
	for ipb, pred_box in enumerate(pred_boxes):
		for igb, gt_box in enumerate(gt_boxes):
			iou= calc_iou(gt_box, pred_box)
 
			if iou >iou_thr:
				gt_idx_thr.append(igb)
				pred_idx_thr.append(ipb)
				ious.append(iou)
	iou_sort = np.argsort(ious)[::1]
	if len(iou_sort)==0:
		tp=0
		fp=len(pred_boxes)
		fn=len(gt_boxes)
		return {'true_positive':tp, 'false_positive':fp, 'false_negative':fn}
	else:
		gt_match_idx=[]
		pred_match_idx=[]
		for idx in iou_sort:
			gt_idx=gt_idx_thr[idx]
			pr_idx= pred_idx_thr[idx]
			# If the boxes are unmatched, add them to matches
			if(gt_idx not in gt_match_idx) and (pr_idx not in pred_match_idx):
				gt_match_idx.append(gt_idx)
				pred_match_idx.append(pr_idx)
		tp= len(gt_match_idx)
		fp= len(pred_boxes) - len(pred_match_idx)
		fn = len(gt_boxes) - len(gt_match_idx)
	return {'true_positive': tp, 'false_positive': fp, 'false_negative': fn}

def get_avg_precision_at_iou(gt_boxes, pred_bb, iou_thr=0.5):
	
	model_scores = get_model_scores(pred_bb)
	sorted_model_scores= sorted(model_scores.keys())
	# Sort the predicted boxes in descending order (lowest scoring boxes first):
	for img_id in pred_bb.keys():
		
		arg_sort = np.argsort(pred_bb[img_id]['score'])
		pred_bb[img_id]['score'] = np.array(pred_bb[img_id]['score'])[arg_sort].tolist()
		pred_bb[img_id]['bbox'] = np.array(pred_bb[img_id]['bbox'])[arg_sort].tolist()
	pred_boxes_pruned = deepcopy(pred_bb)
	
	precisions = []
	recalls = []
	model_thrs = []
	img_results = {}
	# Loop over model score thresholds and calculate precision, recall
	for ithr, model_score_thr in enumerate(sorted_model_scores[:-1]):
			# On first iteration, define img_results for the first time:
		print("Mode score : ", model_score_thr)
		img_ids = gt_boxes.keys() if ithr == 0 else model_scores[model_score_thr]
		for img_id in img_ids:
			   
			gt_boxes_img = gt_boxes[img_id]
			box_scores = pred_boxes_pruned[img_id]['score']
			start_idx = 0
			for score in box_scores:
				if score <= model_score_thr:
					pred_boxes_pruned[img_id]
					start_idx += 1
				else:
					break 
			# Remove boxes, scores of lower than threshold scores:
			pred_boxes_pruned[img_id]['score']= pred_boxes_pruned[img_id]['score'][start_idx:]
			pred_boxes_pruned[img_id]['bbox']= pred_boxes_pruned[img_id]['bbox'][start_idx:]
		# Recalculate image results for this image
			print(img_id)
			img_results[img_id] = get_single_image_results(gt_boxes_img, pred_boxes_pruned[img_id]['bbox'], iou_thr=0.5)
		# calculate precision and recall
		prec, rec = calc_precision_recall(img_results)
		precisions.append(prec)
		recalls.append(rec)
		model_thrs.append(model_score_thr)
	precisions = np.array(precisions)
	recalls = np.array(recalls)
	prec_at_rec = []
	for recall_level in np.linspace(0.0, 1.0, 11):
		try:
			args= np.argwhere(recalls>recall_level).flatten()
			prec= max(precisions[args])
			print(recalls,"Recall")
			print(      recall_level,"Recall Level")
			print(       args, "Args")
			print(       prec, "precision")
		except ValueError:
			prec=0.0
		prec_at_rec.append(prec)
	avg_prec = np.mean(prec_at_rec) 
	return {
		'avg_prec': avg_prec,
		'precisions': precisions,
		'recalls': recalls,
		'model_thrs': model_thrs}

class PrefetchDataset(torch.utils.data.Dataset):
	def __init__(self, opt, dataset, pre_process_func):
		self.images = dataset.images
		self.load_image_func = dataset.coco.loadImgs
		self.img_dir = dataset.img_dir
		self.pre_process_func = pre_process_func
		self.get_default_calib = dataset.get_default_calib
		self.opt = opt
		self._load_data = dataset._load_data
	
	def __getitem__(self, index):
		img_id = self.images[index]
		# img, anns, img_info, img_path = self._load_data(index)

		# ann = anns[0]
		# ann['bbox'][2] += ann['bbox'][0]
		# ann['bbox'][3] += ann['bbox'][1]

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
		# return img_id, ret, img_path, anns[0]
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
		img = cv2.putText(img, str(rs['dep'][0] * 2059.612), (ct_x, ct_y), 0, 0.5, (255, 0, 0))
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