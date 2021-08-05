import os
import numpy as np
import shutil

def resize(in_dir, out_dir):
    for filename in os.listdir(in_dir):
        shutil.copyfile(in_dir + filename, out_dir + filename)

for x in os.walk("data/simulated_v2/data2807_val/cam30/"):
    for subfold in x[1]:
        if not os.path.exists("data/simulated_v2/images/val_cam30_" + subfold + "/"):
            os.makedirs("data/simulated_v2/images/val_cam30_" + subfold + "/")
        resize("data/simulated_v2/data2807_val/cam30/" + subfold + "/images/", "data/simulated_v2/images/val_cam30_" + subfold + "/")