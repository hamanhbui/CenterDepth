import os
import numpy as np
import shutil

def resize(in_dir, out_dir):
    for filename in os.listdir(in_dir):
        shutil.copyfile(in_dir + filename, out_dir + filename)

for x in os.walk("data/simulated_v1_original/data1407_test/cam60/"):
    for subfold in x[1]:
        if not os.path.exists("data/simulated_v1_original/images/test_cam60_" + subfold + "/"):
            os.makedirs("data/simulated_v1_original/images/test_cam60_" + subfold + "/")
        resize("data/simulated_v1_original/data1407_test/cam60/" + subfold + "/images/", "data/simulated_v1_original/images/test_cam60_" + subfold + "/")