"""
    Issue: never split train jsons (ds_info still has a proper train/val split) for ODinW datasets which had no val sets (only had train and test). Now need to split each one into custom_train.json and custom_val.json
    Turns out that the only dataset in this category is PascalVOC
"""
import os, json, pickle


with open("ds_info/PascalVOC.pkl", "rb") as f:
    ds_info = pickle.load(f)

with open("odinw/PascalVOC/train/annotations_without_background.json", "rb") as f:
    train_json = json.load(f)

print("Loaded JSONs")

train_ids = {int(i): True for i in ds_info["train"] if i != "img_pth"} # make dict for O(1) lookup

print(len(train_ids), len(train_json["images"]))

custom_train_json = {
    "images": [],
    "annotations": [],
    "categories": train_json["categories"]
}

custom_val_json = {
    "images": [],
    "annotations": [],
    "categories": train_json["categories"]
}


def distribute_data(json_key:str, id_key:str):
    for i, d in enumerate(train_json[json_key]):
        if d[id_key] in train_ids:
            custom_train_json[json_key].append(d)
        else:
            custom_val_json[json_key].append(d)
    
distribute_data("images", "id")
distribute_data("annotations", "image_id")

print(f"train: {len(custom_train_json['images'])} {len(custom_train_json['annotations'])}, val: {len(custom_val_json['images'])} {len(custom_val_json['annotations'])}")

with open("odinw/PascalVOC/custom_train.json", "w") as f:
    json.dump(custom_train_json, f)

with open("odinw/PascalVOC/custom_val.json", "w") as f:
    json.dump(custom_val_json, f)