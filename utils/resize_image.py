import os
import cv2
import numpy as np

def resize(in_dir, out_dir):
    for filename in os.listdir(in_dir):
        img = cv2.imread(in_dir + filename, cv2.IMREAD_UNCHANGED)
        # resize image
        resized = cv2.resize(img, (960, 544), interpolation = cv2.INTER_AREA)

        cv2.imwrite(out_dir + filename, resized)

# resize("data/data_2806_1k/cam30_curve_0/images/", "data/simulated/images/cam30_curve_0/")
resize("data/data_2806_1k/cam30_curve_1/images/", "data/simulated/images/cam30_curve_1/")
resize("data/data_2806_1k/cam30_curve_2/images/", "data/simulated/images/cam30_curve_2/")
resize("data/data_2806_1k/cam30_straight_0/images/", "data/simulated/images/cam30_straight_0/")
resize("data/data_2806_1k/cam30_straight_1/images/", "data/simulated/images/cam30_straight_1/")
resize("data/data_2806_1k/cam30_straight_2/images/", "data/simulated/images/cam30_straight_2/")

