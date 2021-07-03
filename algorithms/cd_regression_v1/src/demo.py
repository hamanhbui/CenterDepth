import os
import sys
import cv2
import json
import copy
import numpy as np
from detector import Detector

def save_img(img, results):
	for rs in results:
		bbox = rs['bbox']
		ct_x = int(bbox[0] + (bbox[2] - bbox[0])/2)
		ct_y = int(bbox[1] + (bbox[3] - bbox[1])/2)
		img = cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
		img = cv2.putText(img, str(rs['dep'][0]), (ct_x, ct_y), 0, 0.5, (255, 0, 0))
		img = cv2.circle(img, (ct_x, ct_y), radius=1, color=(0, 0, 255), thickness=-1)
	
	return img

def demo(opt):
	detector = Detector(opt)
	video = cv2.VideoWriter('test_30.avi', 0, 1, (960,544))

	with open('data/simulated/annotations/test_30.json') as json_file:
		data = json.load(json_file)
		for p in data["images"]:
			img = cv2.imread("data/simulated/images/" + p["file_name"])
			ret = detector.run(img)
			img = save_img(img, ret['results'])
			video.write(img)

	# file1 = open('data/demo_unified/sample.txt', 'r')
	# Lines = file1.readlines()
	# for line in Lines:
	# 	file_name = line.strip()
	# 	file_name = file_name.replace("/home/ubuntu/vinscenes/", "/home/ubuntu/source-code/CenterDepth/data/demo_unified/")
	# 	img = cv2.imread(file_name)
	# 	img = cv2.resize(img, (960,544))
	# 	ret = detector.run(img)
	# 	img = save_img(img, ret['results'])
	# 	video.write(img)
