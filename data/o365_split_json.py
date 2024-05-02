"""
Filter the train JSON by image ID and save
"""
import os, sys, json, pickle
import pandas as pd
import numpy as np
import torch


with open("O365/objects365_train.json", "rb") as f:
    train_json = json.load(f)

print("Loaded JSONs")

ids = pd.read_csv("O365/objects365_train_example_ids_and_filenames.csv", header=None, delimiter=" ")[0].tolist()
rng = np.random.default_rng(seed=0)
rng.shuffle(ids)

train_ids = {i: True for i in ids[:-30000]} # make dict for O(1) lookup

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

        if i % 100000 == 0:
            print(json_key, f"{i}/{len(train_json[json_key])}")
    
distribute_data("images", "id")
distribute_data("annotations", "image_id")

print(f"train: {len(custom_train_json['images'])} {len(custom_train_json['annotations'])}, val: {len(custom_val_json['images'])} {len(custom_val_json['annotations'])}")

with open("O365/custom_train.json", "w") as f:
    json.dump(custom_train_json, f)

with open("O365/custom_val.json", "w") as f:
    json.dump(custom_val_json, f)