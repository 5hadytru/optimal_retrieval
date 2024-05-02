"""
Merge disparate annotations
"""
import os, sys, json, pickle
import pandas as pd
import numpy as np
import torch

num_processes = int(sys.argv[1])

with open("LVIS/lvis_v1_train.json", "rb") as f:
    train_json = json.load(f)

rng = np.random.default_rng(seed=58008)
rng.shuffle(train_json["images"])

imgs = train_json["images"][:20000]

val_json = {"images": imgs, "annotations": [], "categories": train_json["categories"]}

for i in range(num_processes):
    with open(f"temp2/{i}.pkl", "rb") as f:
        val_json["annotations"] += pickle.load(f)

with open(f"LVIS/custom_val.json", "w") as f:
    json.dump(val_json, f)
