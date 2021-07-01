import os
import cv2
import numpy as np

def resize(in_dir, out_dir):
    for filename in os.listdir(in_dir):
        img = cv2.imread(in_dir + filename, cv2.IMREAD_UNCHANGED)
        # resize image
        resized = cv2.resize(img, (960, 544), interpolation = cv2.INTER_AREA)

        cv2.imwrite(out_dir + filename, resized)

resize("../data0107_test/cam30_test_curve_1/images/", "../data/simulated/images/cam30_test_curve_1/")
resize("../data0107_test/cam30_test_long_1/images/", "../data/simulated/images/cam30_test_long_1/")
resize("../data0107_test/cam425_test_curve_1/images/", "../data/simulated/images/cam425_test_curve_1/")
resize("../data0107_test/cam425_test_long_1/images/", "../data/simulated/images/cam425_test_long_1/")

