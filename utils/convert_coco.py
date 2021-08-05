import json
import numpy as np

test_meta = ["TestCurve1_0", "TestCurve1_1", "TestCurve1_2", "TestCurve2_0", "TestCurve2_1", "TestCurve2_2", "TestCurve3_0", "TestCurve3_1", "TestCurve3_2"]

train_meta = ["BaseCurve1_0", "BaseCurve1_1", "BaseCurve1_2", "BaseCurve1_3", "BaseCurve2_0", "BaseCurve2_1", "BaseCurve2_2", "BaseCurve2_3",
    "BaseCurve3_0", "BaseCurve3_1", "BaseCurve3_2", "BaseCurve3_3", "BaseCurve4_0", "BaseCurve4_1", "BaseCurve4_2", "BaseCurve4_3",
    "BaseCurve5_0", "BaseCurve5_1", "BaseCurve5_2", "BaseCurve5_3"]

val_meta = ["BaseCurve6_0", "BaseCurve6_1", "BaseCurve6_2", "BaseCurve6_3"]

def convert_2_COCO(mode, meta_datas, cam, data, calib):
    for idx in range(len(meta_datas)):
        meta_data = meta_datas[idx]
        with open('data/simulated_v2/data2807_'+ mode +'/cam' + str(cam) + '/' + meta_data + '/annotations.json') as json_file:
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
# data = convert_2_COCO(mode = "test", meta_datas = test_meta, cam = 30, data = data, calib = [[3850.46790537, 0.0, 960.0, 0.0], [0.0, 3850.46790537, 604.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
data = convert_2_COCO(mode = "train", meta_datas = train_meta, cam = 60, data = data, calib = [[1662.82807514, 0.0, 960.0, 0.0], [0.0, 1662.82807514, 604.0, 0.0], [0.0, 0.0, 1.0, 0.0]])

with open('train_60.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)
