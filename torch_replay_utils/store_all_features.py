import numpy as np
import faiss
import os, time
import torch
from torch.utils.data import Dataset
import h5py


# Get *ordered* list of feature matrix files
coreset_path = "../data/coresets/laion400m-data_0.4/CLIP_ViT_B_16_2B/"
hdf5_file_list = []
for i in range(19):
    for filename in os.listdir(coreset_path):
        if filename.startswith("feat") and filename.endswith(f"_{i}.hdf5"):
            hdf5_file_list.append(os.path.join(coreset_path, filename))
            break

feature_types = ['avg', 'image', 'text']

matrix_lists = [[], [], []]

for hdf5_file in hdf5_file_list:
    with h5py.File(hdf5_file, 'r') as f:
        for i, feature_type in enumerate(feature_types):
            print(f"{feature_type}, {hdf5_file}")
            matrix = f[feature_type][:]
            matrix_lists[i].append(matrix)

print('Saving features')
for i in range(len(feature_types)):
    matrix_combined = np.concatenate(matrix_lists[i], axis=0)
    print(feature_types[i], matrix_combined.shape)
    np.save(os.path.join(coreset_path, f"feat_{feature_types[i]}.npy"), matrix_combined)
