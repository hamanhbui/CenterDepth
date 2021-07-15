import numpy as np
import cv2
from .image import transform_preds_with_trans, get_affine_transform
import numba

def generic_post_process(
	opt, dets, c, s, h, w, num_classes, calibs=None, height=-1, width=-1):
	if not ('scores' in dets):
		return [{}], [{}]
	ret = []

	for i in range(len(dets['scores'])):
		preds = []
		trans = get_affine_transform(
			c[i], s[i], 0, (w, h), inv=1).astype(np.float32)
		for j in range(len(dets['scores'][i])):
			if dets['scores'][i][j] < opt.out_thresh:
				break
			item = {}
			item['score'] = dets['scores'][i][j]
			item['class'] = int(dets['clses'][i][j]) + 1
			item['ct'] = transform_preds_with_trans(
				(dets['cts'][i][j]).reshape(1, 2), trans).reshape(2)

			if 'tracking' in dets:
				tracking = transform_preds_with_trans(
					(dets['tracking'][i][j] + dets['cts'][i][j]).reshape(1, 2), 
					trans).reshape(2)
				item['tracking'] = tracking - item['ct']

			if 'bboxes' in dets:
				bbox = transform_preds_with_trans(
					dets['bboxes'][i][j].reshape(2, 2), trans).reshape(4)
				item['bbox'] = bbox

			if 'dep' in dets and len(dets['dep'][i]) > j:
				item['dep'] = dets['dep'][i][j]
			
			preds.append(item)

		ret.append(preds)
	
	return ret