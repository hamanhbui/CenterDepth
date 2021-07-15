import json
import numpy as np

height = 1208
width = 1920

x_scale = 960 / width
y_scale = 544 / height


test_meta = ["BaseCurveTest_1_0", "BaseCurveTest_1_1", "BaseCurveTest_2_0", "BaseCurveTest_2_1", "BaseCurveTest_3_0", "BaseCurveTest_3_1",
    "BaseJunction_1_0", "BaseJunction_1_1", "BaseJunction_2_0", "BaseJunction_2_1", "BaseJunction_3_0", "BaseJunction_3_1"]

train_meta = ["base_curve_1_0", "base_curve_1_1", "base_curve_1_2", "base_curve_2_0", "base_curve_2_1", "base_curve_2_2",
    "base_curve_3_0", "base_curve_3_1", "base_curve_3_2", "base_curve_4_0", "base_curve_4_1", "base_curve_4_2",
    "base_straight_1_0", "base_straight_1_1", "base_straight_1_2", "base_straight_2_0", "base_straight_2_1", "base_straight_2_2",
    "base_straight_3_0", "base_straight_3_1", "base_straight_3_2", "base_straight_4_0", "base_straight_4_1", "base_straight_4_2"]

val_meta = ["BaseCurveVal_1_0", "BaseCurveVal_1_1", "BaseCurveVal_2_0", "BaseCurveVal_2_1", "BaseCurveVal_3_0", "BaseCurveVal_3_1"]

def convert_2_COCO(mode, meta_datas, focal_length, cam, data):
    for idx in range(len(meta_datas)):
        meta_data = meta_datas[idx]
        with open('data/simulated_original/data1407_'+ mode +'/cam' + str(cam) + '/' + meta_data + '/annotations.json') as json_file:
            new_data = json.load(json_file)
            for p in new_data["images"]:
                p["height"] = 544
                p["width"] = 960
                p["id"] = len(data["images"]) + p["id"]
                p["video_id"] += 1
                p["file_name"] = mode + "_cam" + str(cam) + "_" + meta_data + "/" + p["file_name"]

            for p in new_data["annotations"]:
                p["id"] = len(data["annotations"]) + p["id"]
                p["image_id"] = len(data["images"]) + p["image_id"]
                p["bbox"][0] = int(np.round(p["bbox"][0] * x_scale))
                p["bbox"][1] = int(np.round(p["bbox"][1] * y_scale))
                p["bbox"][2] = int(np.round(p["bbox"][2] * x_scale))
                if p["bbox"][2] == 0:
                    p["bbox"][2] = 1
                p["bbox"][3] = int(np.round(p["bbox"][3] * y_scale))
                if p["bbox"][3] == 0:
                    p["bbox"][3] = 1
                p["category_id"] += 1
                p["depth"] = p["distance"] / focal_length
            
        data["videos"].extend([{"id": len(data["videos"]) + 1, "file_name": mode + '_cam' + str(cam) + '/' + meta_data}])
        data["images"].extend(new_data["images"])
        data["annotations"].extend(new_data["annotations"])
    
    return data

data = {"images": [], "annotations": [], "videos": [], "categories": [{ "id": 1, "name": "Traffic_light"}, { "id": 2, "name": "Traffic_sign"}, { "id": 3, "name": "Vehicle"}]}
data = convert_2_COCO(mode = "val", meta_datas = val_meta, focal_length = 3850.46790537, cam = 30, data = data)
data = convert_2_COCO(mode = "val", meta_datas = val_meta, focal_length = 1662.82807514, cam = 60, data = data)

with open('val.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)
