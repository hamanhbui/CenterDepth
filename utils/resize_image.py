import os
import cv2
import numpy as np

def resize(in_dir, out_dir):
    for filename in os.listdir(in_dir):
        img = cv2.imread(in_dir + filename, cv2.IMREAD_UNCHANGED)
        # resize image
        resized = cv2.resize(img, (960, 544), interpolation = cv2.INTER_AREA)

        cv2.imwrite(out_dir + filename, resized)

for x in os.walk("data/simulated_original/data1407_val/cam30/"):
    for subfold in x[1]:
        if not os.path.exists("data/simulated/images/val_cam30_" + subfold + "/"):
            os.makedirs("data/simulated/images/val_cam30_" + subfold + "/")
        resize("data/simulated_original/data1407_val/cam30/" + subfold + "/images/", "data/simulated/images/val_cam30_" + subfold + "/")