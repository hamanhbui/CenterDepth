import json
import numpy as np

height = 1208
width = 1920

x_scale = 960 / width
y_scale = 544 / height

def convert_2_COCO(meta_datas = ["cam30_straight_0", "cam30_straight_1", "cam30_straight_2", "cam30_curve_0", "cam30_curve_1", "cam30_curve_2"]):
    data = {"images": [], "annotations": [], "videos": [], "categories": [{ "id": 1, "name": "Traffic_light"}, { "id": 2, "name": "Traffic_sign"}]}
    train_data = {"images": [], "annotations": [], "videos": [], "categories": [{ "id": 1, "name": "Traffic_light"}, { "id": 2, "name": "Traffic_sign"}]}
    valid_data = {"images": [], "annotations": [], "videos": [], "categories": [{ "id": 1, "name": "Traffic_light"}, { "id": 2, "name": "Traffic_sign"}]}
    test_data = {"images": [], "annotations": [], "videos": [], "categories": [{ "id": 1, "name": "Traffic_light"}, { "id": 2, "name": "Traffic_sign"}]}
    for idx in range(len(meta_datas)):
        meta_data = meta_datas[idx]
        with open('data/data_2806_1k/' + meta_data + '/annotations.json') as json_file:
            new_data = json.load(json_file)
            for p in new_data["images"]:
                p["height"] = 544
                p["width"] = 960
                p["id"] = len(data["images"]) + p["id"]
                p["video_id"] += 1
                p["file_name"] = meta_data + "/" + p["file_name"]

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
                p["depth"] = p["distance"]
            
        data["videos"].extend([{"id": idx + 1, "file_name": meta_data}])
        data["images"].extend(new_data["images"])
        data["annotations"].extend(new_data["annotations"])
    
    return data

data = convert_2_COCO()
with open('data.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)
