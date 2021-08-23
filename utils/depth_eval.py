import matplotlib.pyplot as plt

def plot_hostogram(pred_depth):
	# An "interface" to matplotlib.axes.Axes.hist() method
	n, bins, patches = plt.hist(x=pred_depth, bins='auto', color='#FF0000',
								alpha=0.7, rwidth=0.85)
	plt.grid(axis='y', alpha=0.75)
	plt.xlabel('Value')
	plt.ylabel('Frequency')
	plt.title('Ground Trurth Depth')
	maxfreq = n.max()
	# Set a clean upper y-axis limit.
	plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
	plt.savefig('ground_truth.png')

if __name__ == "__main__":

    gt_maps_dir = f'/home/nhoos/catkin_ws/data1407_test/'
    pre_maps_dir = '/home/nhoos/work_space/out'

    eval(gt_maps_dir, pre_maps_dir, cam, IOU_THRESHOLD)