import os
import sys
import cv2
import json
import copy
import numpy as np
from detector import Detector

def save_img(img, results, calib):
	for rs in results:
		bbox = rs['bbox']
		# depth = rs["dep"][0]
		ct_x = int(bbox[0] + (bbox[2] - bbox[0])/2)
		ct_y = int(bbox[1] + (bbox[3] - bbox[1])/2)

		# locations = unproject_2d_to_3d((ct_x, ct_y), depth, calib)

		if rs["class"] == 1:
			img = cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)
		else:
			img = cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
		
		# img = cv2.putText(img, "X:" + str(locations[0]), (int(bbox[2]), int(bbox[1])), 0, 0.3, (255, 0, 0))
		# img = cv2.putText(img, "Y:" + str(locations[1]), (int(bbox[2]), ct_y), 0, 0.3, (255, 0, 0))
		# img = cv2.putText(img, str(depth), (int(bbox[2]), int(bbox[3])), 0, 0.3, (255, 0, 0))

		img = cv2.circle(img, (ct_x, ct_y), radius=1, color=(0, 0, 255), thickness=-1)
	
	return img

def unproject_2d_to_3d(pt_2d, depth, P):
  # pts_2d: 2
  # depth: 1
  # P: 3 x 4
  # return: 3
  z = depth - P[2, 3]
  x = (pt_2d[0] * depth - P[0, 3] - P[0, 2] * z) / P[0, 0]
  y = (pt_2d[1] * depth - P[1, 3] - P[1, 2] * z) / P[1, 1]
  pt_3d = np.array([x, y, z], dtype=np.float32).reshape(3)
  return pt_3d

def demo(opt):
	calib = np.array(
    [[1925.23395269, 0.0, 480.0, 0.0],
     [0.0, 1733.98554679, 272.0, 0.0],
     [0.0, 0.0, 1.0, 0.0]],
    dtype=np.float32)

	detector = Detector(opt)
	video = cv2.VideoWriter('test_30.avi', 0, fps = 5, frameSize = (960,544))

	with open(opt.test_meta_filenames) as json_file:
		data = json.load(json_file)
		for p in data["images"]:
			img = cv2.imread("data/simulated/images/" + p["file_name"])
			ret = detector.run(img)
			img = save_img(img, ret['results'], calib)
			video.write(img)

	# file1 = open('data/demo_unified/sample.txt', 'r')
	# Lines = file1.readlines()
	# for line in Lines:
	# 	file_name = line.strip()
	# 	file_name = file_name.replace("/home/ubuntu/vinscenes/", "/home/ubuntu/source-code/CenterDepth/data/demo_unified/")
	# 	img = cv2.imread(file_name)
	# 	img = cv2.resize(img, (960,544))
	# 	ret = detector.run(img)
	# 	img = save_img(img, ret['results'], calib)
	# 	video.write(img)
