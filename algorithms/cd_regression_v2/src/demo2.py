import os
import sys
import cv2
import json
import copy
import numpy as np
from detector import Detector

import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm

import torch
from torchvision import transforms

from .models.depth_decoder import DepthDecoder
from .models.pose_cnn import PoseCNN
from .models.pose_decoder import PoseDecoder
from .models.resnet_encoder import ResnetEncoder, ResnetEncoderMatching

from .models.layers import transformation_from_parameters

def load_and_preprocess_intrinsics(resize_width, resize_height):
	K = np.eye(4)
	K[:3, :3] = [[1925.23395269, 0.0, 480.0],
			[0.0, 1733.98554679, 272.0],
			[0.0, 0.0, 1.0]]

	# Convert normalised intrinsics to 1/4 size unnormalised intrinsics.
	# (The cost volume construction expects the intrinsics corresponding to 1/4 size images)
	K[0, :] *= resize_width // 4
	K[1, :] *= resize_height // 4

	invK = torch.Tensor(np.linalg.pinv(K)).unsqueeze(0)
	K = torch.Tensor(K).unsqueeze(0)

	if torch.cuda.is_available():
		return K.cuda(), invK.cuda()
	return K, invK

def load_and_preprocess_image(image_path, resize_width, resize_height):
	image = pil.open(image_path).convert('RGB')
	original_width, original_height = image.size
	image = image.resize((resize_width, resize_height), pil.LANCZOS)
	image = transforms.ToTensor()(image).unsqueeze(0)
	if torch.cuda.is_available():
		return image.cuda(), (original_height, original_width)
	return image, (original_height, original_width)


def save_img(img, results, calib, sigmoid_output_resized):
	for rs in results:
		bbox = rs['bbox']
		ct_x = int(bbox[0] + (bbox[2] - bbox[0])/2)
		ct_y = int(bbox[1] + (bbox[3] - bbox[1])/2)
		if ct_x > 543:
			ct_x = 543
		if ct_y > 959:
			ct_y = 959

		depth = sigmoid_output_resized[0][ct_y, ct_x] * 500

		locations = unproject_2d_to_3d((ct_x, ct_y), depth, calib)

		if rs["class"] == 1:
			img = cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)
		else:
			img = cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
		
		img = cv2.putText(img, "X:" + str(locations[0]), (int(bbox[2]), int(bbox[1])), 0, 0.3, (255, 0, 0))
		img = cv2.putText(img, "Y:" + str(locations[1]), (int(bbox[2]), ct_y), 0, 0.3, (255, 0, 0))
		img = cv2.putText(img, str(depth), (int(bbox[2]), int(bbox[3])), 0, 0.3, (255, 0, 0))

		img = cv2.circle(img, (ct_x, ct_y), radius=1, color=(0, 0, 255), thickness=-1)
	
	return img

def unproject_2d_to_3d(pt_2d, depth, P):
  # pts_2d: 2
  # depth: 1
  # P: 3 x 3
  # return: 3
  z = depth
  x = ((pt_2d[0] - P[0, 2]) * z) / P[0, 0]
  y = ((pt_2d[1] - P[1, 2]) * z) / P[1, 1]
  pt_3d = np.array([x, y, z], dtype=np.float32).reshape(3)
  return pt_3d

def demo(opt):
	model_path = "algorithms/cd_regression_v2/results/checkpoints/KITTI_MR/"

	# Loading pretrained model
	print("   Loading pretrained encoder")
	encoder_dict = torch.load(os.path.join(model_path, "encoder.pth"), map_location="cuda")
	encoder = ResnetEncoderMatching(18, False,
											 input_width=encoder_dict['width'],
											 input_height=encoder_dict['height'],
											 adaptive_bins=True,
											 min_depth_bin=encoder_dict['min_depth_bin'],
											 max_depth_bin=encoder_dict['max_depth_bin'],
											 depth_binning='linear',
											 num_depth_bins=96)

	filtered_dict_enc = {k: v for k, v in encoder_dict.items() if k in encoder.state_dict()}
	encoder.load_state_dict(filtered_dict_enc)

	print("   Loading pretrained decoder")
	depth_decoder = DepthDecoder(num_ch_enc=encoder.num_ch_enc, scales=range(4))

	loaded_dict = torch.load(os.path.join(model_path, "depth.pth"), map_location="cuda")
	depth_decoder.load_state_dict(loaded_dict)

	print("   Loading pose network")
	pose_enc_dict = torch.load(os.path.join(model_path, "pose_encoder.pth"),
							   map_location="cuda")
	pose_dec_dict = torch.load(os.path.join(model_path, "pose.pth"), map_location="cuda")

	pose_enc = ResnetEncoder(18, False, num_input_images=2)
	pose_dec = PoseDecoder(pose_enc.num_ch_enc, num_input_features=1,
									num_frames_to_predict_for=2)

	pose_enc.load_state_dict(pose_enc_dict, strict=True)
	pose_dec.load_state_dict(pose_dec_dict, strict=True)

	# Setting states of networks
	encoder.eval()
	depth_decoder.eval()
	pose_enc.eval()
	pose_dec.eval()
	if torch.cuda.is_available():
		encoder.cuda()
		depth_decoder.cuda()
		pose_enc.cuda()
		pose_dec.cuda()

	K, invK = load_and_preprocess_intrinsics(resize_width=encoder_dict['width'],
											 resize_height=encoder_dict['height'])
	
	with torch.no_grad():

		calib = np.array(
			[[1925.23395269, 0.0, 480.0],
			[0.0, 1733.98554679, 272.0],
			[0.0, 0.0, 1.0]],
			dtype=np.float32)

		# [[1925.23395269, 0.0, 480.0],
		# [0.0, 1733.98554679, 272.0],
		# [0.0, 0.0, 1.0]]
		
		# [[1234.35792236, 0.0, 480.0],
		# [0.0, 1111.7395857, 272.0],
		# [0.0, 0.0, 1.0]]

		detector = Detector(opt)
		video = cv2.VideoWriter('test_30.avi', 0, fps = 5, frameSize = (960,1088))
		
		# source_image_path = "/home/ubuntu/source-code/CenterDepth/data/demo_unified/inhouse_2d_selection//2dod-highway-batch1_4058_813/images/CAM_FRONT_LEFT/000000.jpeg"
		# file1 = open('data/demo_unified/sample.txt', 'r')
		# Lines = file1.readlines()
		# for line in Lines:
		# 	file_name = line.strip()
		# 	file_name = file_name.replace("/home/ubuntu/vinscenes/", "/home/ubuntu/source-code/CenterDepth/data/demo_unified/")
		# 	img = cv2.imread(file_name)
		# 	img = cv2.resize(img, (960,544))

		source_image_path = "/home/ubuntu/source-code/CenterDepth/data/simulated/images/cam30_test_long_1/1625114324.4163804.png"
		with open(opt.test_meta_filenames) as json_file:
			data = json.load(json_file)
			for p in data["images"]:
				file_name = "data/simulated/images/" + p["file_name"]
				img = cv2.imread("data/simulated/images/" + p["file_name"])
				ret = detector.run(img)

				target_image_path = file_name

				input_image, original_size = load_and_preprocess_image(target_image_path,
															resize_width=encoder_dict['width'],
															resize_height=encoder_dict['height'])

				source_image, _ = load_and_preprocess_image(source_image_path,
															resize_width=encoder_dict['width'],
															resize_height=encoder_dict['height'])

				pose_inputs = [source_image, input_image]
				pose_inputs = [pose_enc(torch.cat(pose_inputs, 1))]
				axisangle, translation = pose_dec(pose_inputs)
				pose = transformation_from_parameters(axisangle[:, 0], translation[:, 0], invert=True)

				output, lowest_cost, _ = encoder(current_image=input_image,
											lookup_images=source_image.unsqueeze(1),
											poses=pose.unsqueeze(1),
											K=K,
											invK=invK,
											min_depth_bin=encoder_dict['min_depth_bin'],
											max_depth_bin=encoder_dict['max_depth_bin'])

				output = depth_decoder(output)

				sigmoid_output = output[("disp", 0)]
				sigmoid_output_resized = torch.nn.functional.interpolate(
					sigmoid_output, (544, 960), mode="bilinear", align_corners=False)
				sigmoid_output_resized = sigmoid_output_resized.cpu().numpy()[:, 0]

				img = save_img(img, ret['results'], calib, sigmoid_output_resized)

				for plot_name, toplot in (('costvol_min', lowest_cost), ('disp', sigmoid_output_resized)):
					toplot = toplot.squeeze()
					normalizer = mpl.colors.Normalize(vmin=toplot.min(), vmax=np.percentile(toplot, 95))
					mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
					colormapped_im = (mapper.to_rgba(toplot)[:, :, :3] * 255).astype(np.uint8)
					im = pil.fromarray(colormapped_im)
					im.save("tmp.png")

				depth_img = cv2.imread("tmp.png")
				im_v = cv2.vconcat([img, depth_img])
				video.write(im_v)