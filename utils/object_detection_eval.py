import os
import json
import numpy as np
from scipy.optimize import linear_sum_assignment

def get_max_iou(pred_boxes, gt_box):
    """
    calculate the iou multiple pred_boxes and 1 gt_box (the same one)
    pred_boxes: multiple predict  boxes coordinate
    gt_box: ground truth bounding  box coordinate
    return: the max overlaps about pred_boxes and gt_box
    """
    # 1. calculate the inters coordinate
    if pred_boxes.shape[0] > 0:
        ixmin = np.maximum(pred_boxes[:, 0], gt_box[0])
        ixmax = np.minimum(pred_boxes[:, 2], gt_box[2])
        iymin = np.maximum(pred_boxes[:, 1], gt_box[1])
        iymax = np.minimum(pred_boxes[:, 3], gt_box[3])

        iw = np.maximum(ixmax - ixmin + 1., 0.)
        ih = np.maximum(iymax - iymin + 1., 0.)

    # 2.calculate the area of inters
        inters = iw * ih

    # 3.calculate the area of union
        uni = ((pred_boxes[:, 2] - pred_boxes[:, 0] + 1.) * (pred_boxes[:, 3] - pred_boxes[:, 1] + 1.) +
               (gt_box[2] - gt_box[0] + 1.) * (gt_box[3] - gt_box[1] + 1.) -
               inters)

    # 4.calculate the overlaps and find the max overlap ,the max overlaps index for pred_box
        iou = inters / uni
        iou_max = np.max(iou)
        nmax = np.argmax(iou)
        return iou, iou_max, nmax


def get_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.
    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    Returns
    -------
    float
        in [0, 1]
    """
    assert bb1['x1'] <= bb1['x2']
    assert bb1['y1'] <= bb1['y2']
    assert bb2['x1'] <= bb2['x2']
    assert bb2['y1'] <= bb2['y2']

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou

def tlbr2trbl(tl, br):
    x_min, y_min = tl
    x_max, y_max = br
    return [x_max, y_min], [x_min, y_max]

def trbl2tlbr(tr, bl):
    x_max, y_min = tr
    x_min, y_max = bl
    return [x_min, y_min], [x_max, y_max]

def parse_gt(gt_path):
    annos_dict = json.load(open(gt_path, "r"))
    gt_annos = []
    for anno in annos_dict['annos_bbox']:
        if anno['type_id'] != 4: # Not traffic sign
            continue
        tr = anno['top_right']
        bl = anno['bot_left']
        tl, br = trbl2tlbr(tr, bl)
        coor = tl + br
        coor = [float(x) for x in coor]
        coor_dict = {}
        coor_dict['x1'] = coor[0]
        coor_dict['y1'] = coor[1]
        coor_dict['x2'] = coor[2]
        coor_dict['y2'] = coor[3]
        gt_annos.append(coor_dict)
    return gt_annos

def parse_pre(pre_path):
    annos_dict = json.load(open(pre_path, "r"))
    pre_annos = []
    for anno in annos_dict['anno']:
        br = anno['2d_bot_right']    
        tl = anno['2d_top_left']
        # tr, bl = tlbr2trbl(tl, br)
        coor = tl + br
        coor = [float(x) for x in coor]
        coor_dict = {}
        coor_dict['x1'] = coor[0]
        coor_dict['y1'] = coor[1]
        coor_dict['x2'] = coor[2]
        coor_dict['y2'] = coor[3]
        pre_annos.append(coor_dict)
    return pre_annos

def eval(gt_maps_dir, pre_maps_dir, cam, IOU_THRESHOLD):
    TP = 0
    FP = 0
    FN = 0
    gt_maps_dir = os.path.join(gt_maps_dir, f"{cam}")
    list_gt_map = os.listdir(gt_maps_dir)

    for gt_map in list_gt_map:
        map_dir = os.path.join(gt_maps_dir, gt_map, 'json')
        pre_dir = os.path.join(pre_maps_dir, f'test_{cam}_{gt_map}')
        gt_dirs = os.listdir(map_dir)
        for gt in gt_dirs:
            gt_path = os.path.join(map_dir, gt)
            pre_path = os.path.join(pre_dir, gt)

            batch_TP = 0
            gt_annos = parse_gt(gt_path)
            pre_annos = parse_pre(pre_path)

            cost_matrix = np.ones((len(gt_annos), len(pre_annos)))
            for i in range(len(gt_annos)):
                for j in range(len(pre_annos)):
                    iou = get_iou(pre_annos[j], gt_annos[i])
                    if iou > IOU_THRESHOLD:
                        cost_matrix[i,j] = 1 - iou
                
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            for x,y in zip(row_ind,col_ind):
                if cost_matrix[x,y] < 1:
                    batch_TP+=1
            
            TP += batch_TP
            FP += len(pre_annos) - batch_TP
            FN += len(gt_annos) - batch_TP


    precision = TP / (TP + FP)
    recall = TP / (TP + FN)

    F1 = 2 * (precision * recall) / (precision + recall)

    print("IOU_THRESHOLD", IOU_THRESHOLD)
    print("TP", TP)
    print("FP", FP)
    print("FN", FN)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1",F1)

if __name__ == "__main__":

    IOU_THRESHOLD = 0.5
    cam = 'cam30'
    gt_maps_dir = f'/home/nhoos/catkin_ws/data1407_test/'
    pre_maps_dir = '/home/nhoos/work_space/out'

    eval(gt_maps_dir, pre_maps_dir, cam, IOU_THRESHOLD)
    
