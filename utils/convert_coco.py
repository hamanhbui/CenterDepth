import json
import numpy as np

test_meta = ["BaseCurveTest_1_0", "BaseCurveTest_1_1", "BaseCurveTest_2_0", "BaseCurveTest_2_1", "BaseCurveTest_3_0", "BaseCurveTest_3_1",
    "BaseJunction_1_0", "BaseJunction_1_1", "BaseJunction_2_0", "BaseJunction_2_1", "BaseJunction_3_0", "BaseJunction_3_1"]

train_meta = ["base_curve_1_0", "base_curve_1_1", "base_curve_1_2", "base_curve_2_0", "base_curve_2_1", "base_curve_2_2",
    "base_curve_3_0", "base_curve_3_1", "base_curve_3_2", "base_curve_4_0", "base_curve_4_1", "base_curve_4_2",
    "base_straight_1_0", "base_straight_1_1", "base_straight_1_2", "base_straight_2_0", "base_straight_2_1", "base_straight_2_2",
    "base_straight_3_0", "base_straight_3_1", "base_straight_3_2", "base_straight_4_0", "base_straight_4_1", "base_straight_4_2"]

val_meta = ["BaseCurveVal_1_0", "BaseCurveVal_1_1", "BaseCurveVal_2_0", "BaseCurveVal_2_1", "BaseCurveVal_3_0", "BaseCurveVal_3_1"]

def convert_2_COCO(mode, meta_datas, cam, data, calib):
    for idx in range(len(meta_datas)):
        meta_data = meta_datas[idx]
        with open('data/simulated_v1_original/data1407_'+ mode +'/cam' + str(cam) + '/' + meta_data + '/annotations.json') as json_file:
            new_data = json.load(json_file)
            for p in new_data["images"]:
                p["id"] = len(data["images"]) + p["id"]
                p["video_id"] += 1
                p["file_name"] = mode + "_cam" + str(cam) + "_" + meta_data + "/" + p["file_name"]
                p["calib"] = calib

            annotations = []
            for p in new_data["annotations"]:
                p["id"] = len(data["annotations"]) + len(annotations)
                p["image_id"] = len(data["images"]) + p["image_id"]
                if p["category_id"] == 1:
                    annotations.append(p)
            
            new_data["annotations"] = annotations
            
        data["videos"].extend([{"id": len(data["videos"]) + 1, "file_name": mode + '_cam' + str(cam) + '/' + meta_data}])
        data["images"].extend(new_data["images"])
        data["annotations"].extend(new_data["annotations"])
    
    return data

data = {"images": [], "annotations": [], "videos": [], "categories": [{ "id": 1, "name": "Traffic_sign"}]}
data = convert_2_COCO(mode = "test", meta_datas = test_meta, cam = 30, data = data, calib = [[3850.46790537, 0.0, 960.0, 0.0], [0.0, 3850.46790537, 604.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
data = convert_2_COCO(mode = "test", meta_datas = test_meta, cam = 60, data = data, calib = [[1662.82807514, 0.0, 960.0, 0.0], [0.0, 1662.82807514, 604.0, 0.0], [0.0, 0.0, 1.0, 0.0]])

with open('test.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)
