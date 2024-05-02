"""
Split the train JSON into a train and val component
"""
import os, sys, json, pickle
import pandas as pd
import numpy as np
import torch

process_rank = int(sys.argv[1])
num_processes = int(sys.argv[2])

with open("LVIS/lvis_v1_train.json", "rb") as f:
    train_json = json.load(f)

rng = np.random.default_rng(seed=58008)
rng.shuffle(train_json["images"])

imgs = train_json["images"][:20000]

img_ids = [i["id"] for i in imgs]

process_annotations = []
process_ann_idxes = list(i for i in range(process_rank, len(train_json["annotations"]), num_processes))
for j,i in enumerate(process_ann_idxes):
    if i % 5000 == 0:
        print("annotations",  f"{j}/{len(process_ann_idxes)}")
    if train_json["annotations"][i]["image_id"] in img_ids:
        process_annotations.append(train_json["annotations"][i])

with open(f"temp2/{process_rank}.pkl", "wb") as f:
    pickle.dump(process_annotations, f)
