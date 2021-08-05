import os
import sys
import cv2
import json
import copy
import numpy as np
from detector import Detector

def save_img(img, results, calib, filename, frame_id):
	out_lines = {"frame_id": frame_id, "anno": [], "calib": calib}
	for rs in results:
		bbox = rs['bbox']
		depth = rs["dep"][0] * ((calib[0][0] + calib[1][1])/2)

		top_left_3d = unproject_2d_to_3d((bbox[0], bbox[1]), depth, calib)
		bot_right_3d = unproject_2d_to_3d((bbox[2], bbox[3]), depth, calib)

		out_lines["anno"].append({
			"track_id": str(rs["tracking_id"]),
			"2d_top_left": [str(bbox[0]), str(bbox[1])],
			"2d_bot_right": [str(bbox[2]), str(bbox[3])],
			"3d_top_left": top_left_3d,
			"3d_bot_right": bot_right_3d,
		})

	cv2.imwrite("out/" + filename, img)
	filename = filename.replace(".jpeg", ".json")

	with open("out/" + filename, 'w', encoding='utf-8') as f:
		  json.dump(out_lines, f, ensure_ascii=False, indent=4)
	
	return img

def unproject_2d_to_3d(pt_2d, depth, P):
  # pts_2d: 2
  # depth: 1
  # P: 3 x 3
  # return: 3
  P = np.array(P)
  z = depth
  x = ((pt_2d[0] - P[0, 2]) * z) / P[0, 0]
  y = ((pt_2d[1] - P[1, 2]) * z) / P[1, 1]
  return [str(x), str(y), str(z)]

def demo(opt):
	calib = [[2617.9215, 0.0, 952.3042],
        [0.0, 2617.8467, 551.5737],
        [0.0, 0.0, 1.0]]

	detector = Detector(opt)

	file1 = open('data/demo_unified/sample.txt', 'r')
	Lines = file1.readlines()
	sub_fold = "2dod-highway-batch1_4058_813"
	frame_id = 1
	for line in Lines:
		file_name = line.strip()
		file_name = file_name.replace("/home/ubuntu/vinscenes/", "/home/ubuntu/source-code/CenterDepth/data/demo_unified/")
		if sub_fold != file_name.split("/")[9]:
			sub_fold = file_name.split("/")[9]
			detector.reset_tracking()

		if not os.path.exists("out/" + sub_fold + "/"):
				os.makedirs("out/" + sub_fold + "/")

		img = cv2.imread(file_name)
		ret = detector.run(img)
		img = save_img(img, ret['results'], calib, sub_fold + "/" + file_name.split("/")[12], frame_id)
		frame_id += 1
	

	# sub_fold = "test_cam30_BaseCurveTest_1_0"

	# with open(opt.test_meta_filenames) as json_file:
	# 	data = json.load(json_file)
	# 	for p in data["images"]:			
	# 		if sub_fold != p["file_name"].split("/")[0]:
	# 			sub_fold = p["file_name"].split("/")[0]
	# 			detector.reset_tracking()

	# 		if not os.path.exists("out/" + sub_fold + "/"):
	# 			os.makedirs("out/" + sub_fold + "/")

	# 		img = cv2.imread("data/simulated/images/" + p["file_name"])
	# 		ret = detector.run(img)

	# 		img = save_img(img, ret['results'], p["calib"], p["file_name"], p["frame_id"])
