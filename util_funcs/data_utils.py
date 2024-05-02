import numpy as np
import time
import torch
import h5py, json
from torch.utils.data import DataLoader, Subset, Dataset, ConcatDataset, Sampler, dataset, TensorDataset
import math
import time
import os.path
from enum import Enum
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import wandb 
import pickle

# if __name__ != "__main__":
#     from util_funcs.device_utils import ToDeviceLoader


def load_json_file(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


class HDF5_Dataset(Dataset):
    """
    Object for randomly accessing items in a dataset composed of hdf5 files
    """
    def __init__(self, dir_pth:str, split:str, logit_shape=(3600,126)):
        if split == "train":
            self.matrix_keys = [
                ("images", torch.float32), 
                ("image_ids", torch.int), 
                ("imagenet_input_ids", torch.int), 
                ("imagenet_attention_masks", torch.int), 
                ("custom_input_ids", torch.int), 
                ("custom_attention_masks", torch.int), 
                ("boxes", torch.float32), 
                ("labels", torch.int)
            ]
        else:
            self.matrix_keys = [
                ("images", torch.float32), 
                ("image_ids", torch.int)
            ]

        self.hdf5_file_list = sorted(
            [os.path.join(dir_pth, fname) for fname in os.listdir(dir_pth) if f"_{split}_" in fname],
            key=lambda k: int(k.split(f"_{split}_")[-1])
        )

        self.padding_logits = torch.zeros(logit_shape)

        assert len(self.hdf5_file_list) > 0, dir_pth + " " + split

        self.start_indices = []

        # Calculate the starting index of each file in the combined dataset
        total_length = 0
        for hdf5_file in self.hdf5_file_list:
            self.start_indices.append(total_length)
            with h5py.File(hdf5_file, 'r') as f:
                total_length += f[self.matrix_keys[0][0]].shape[0]
        self.length = total_length

    def get_file_idx(self, global_idx):
        for file_i in range(len(self.start_indices)):
            if file_i == len(self.start_indices) - 1 or self.start_indices[file_i + 1] > global_idx:
                return file_i
        raise StopIteration

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        file_idx = self.get_file_idx(idx)
        local_idx = idx - self.start_indices[file_idx]

        # Read the data from the appropriate file and row
        hdf5_file = self.hdf5_file_list[file_idx]

        return_list = []
        with h5py.File(hdf5_file, 'r') as f:
            for key, return_dtype in self.matrix_keys:
                entire_matrix = f[key]
                row = entire_matrix[local_idx]
                item = torch.tensor(row, dtype=return_dtype)
                return_list.append(item)

        return return_list + [self.padding_logits, torch.tensor(False)] # can expand this as needed


"""
Few-shot learning utils
"""
def get_FSL_indices(ordered_labels, k_shots, ds_seed):
    """
    Given an ordered list of labels for a dataset, return a set of indices (!!!ordered!!!)
    which would turn the dataset into a k-shot learning dataset

    Can become an n-way dataset with the proper batch size
    """
    indices = []
    rng = np.random.default_rng(seed=ds_seed)

    unique_classes = np.unique(ordered_labels)
    rng.shuffle(unique_classes)

    for cls in unique_classes:
        cls_indices = np.where(ordered_labels == cls)[0]
        if cls_indices.size < k_shots:
            raise ValueError(f"Class {cls} has fewer than {k_shots} samples.")
        selected_indices = rng.choice(cls_indices, size=k_shots, replace=False)
        indices.extend(selected_indices)

    return indices

"""
DataLoader generation utils; putting it all together
"""
dataset_configs = { # changing the order of these keys would destroy reproducibility
    "all": ['plantdoc', 'ThermalCheetah', 'BCCD', 'AerialMaritimeDrone', 'OxfordPets', 'dice', 'brackishUnderwater', 'pothole', 'WildfireSmoke', 'ChessPieces', 'thermalDogsAndPeople', 'ShellfishOpenImages', 'Aquarium', 'EgoHands', 'AmericanSignLanguageLetters'],
    "GLIP": ["Raccoon", 'PascalVOC', "AerialMaritimeDrone", "Aquarium", "pistols", "ShellfishOpenImages", "EgoHands", "CottontailRabbits", "NorthAmericaMushrooms", "Packages", "thermalDogsAndPeople", "pothole", "VehiclesOpenImages"],
    "custom": ["brackishUnderwater", "ThermalCheetah"]
}

class Settings(Enum):
    """
    Data inflow regimes
    """
    SEQ_FEW_SHOT = 0
    SEQ_FULL_SHOT = 1
    ONLINE_FEW_SHOT = 2


class DistributedEvalSampler(Sampler):
    r"""
    DistributedEvalSampler is different from DistributedSampler.
    It does NOT add extra samples to make it evenly divisible.
    DistributedEvalSampler should NOT be used for training. The distributed processes could hang forever.
    See this issue for details: https://github.com/pytorch/pytorch/issues/22584
    shuffle is disabled by default

    DistributedEvalSampler is for evaluation purpose where synchronization does not happen every epoch.
    Synchronization should be done outside the dataloader loop.

    Sampler that restricts data loading to a subset of the dataset.

    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such a case, each
    process can pass a :class`~torch.utils.data.DistributedSampler` instance as a
    :class:`~torch.utils.data.DataLoader` sampler, and load a subset of the
    original dataset that is exclusive to it.

    .. note::
        Dataset is assumed to be of constant size.

    Arguments:
        dataset: Dataset used for sampling.
        num_replicas (int, optional): Number of processes participating in
            distributed training. By default, :attr:`rank` is retrieved from the
            current distributed group.
        rank (int, optional): Rank of the current process within :attr:`num_replicas`.
            By default, :attr:`rank` is retrieved from the current distributed
            group.
        shuffle (bool, optional): If ``True`` (default), sampler will shuffle the
            indices.
        seed (int, optional): random seed used to shuffle the sampler if
            :attr:`shuffle=True`. This number should be identical across all
            processes in the distributed group. Default: ``0``.

    .. warning::
        In distributed mode, calling the :meth`set_epoch(epoch) <set_epoch>` method at
        the beginning of each epoch **before** creating the :class:`DataLoader` iterator
        is necessary to make shuffling work properly across multiple epochs. Otherwise,
        the same ordering will be always used.

    Example::

        >>> sampler = DistributedSampler(dataset) if is_distributed else None
        >>> loader = DataLoader(dataset, shuffle=(sampler is None),
        ...                     sampler=sampler)
        >>> for epoch in range(start_epoch, n_epochs):
        ...     if is_distributed:
        ...         sampler.set_epoch(epoch)
        ...     train(loader)
    """

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=False, seed=0):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        # self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        # self.total_size = self.num_samples * self.num_replicas
        self.total_size = len(self.dataset)         # true value without extra samples
        indices = list(range(self.total_size))
        indices = indices[self.rank:self.total_size:self.num_replicas]
        self.num_samples = len(indices)             # true value without extra samples

        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))


        # # add extra samples to make it evenly divisible
        # indices += indices[:(self.total_size - len(indices))]
        # assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Arguments:
            epoch (int): _epoch number.
        """
        self.epoch = epoch

def get_loader(
    *,
    split:str,
    dataset_name:str, 
    setting:Settings, 
    batch_size:int, 
    TDS_model_id:str,
    big_ds_HDF5:bool,
    n_ways:int=None, 
    k_shots:int=None,
    return_replay_loader:bool=False,
    replay_buffer=None,
    assert_double:bool=False,
    subset_id:int=None
    ):
    """
    Get a DataLoader containing the proper dataset + Subsetted according to the setting as necessary
    """
    if return_replay_loader and split != "train":
        raise Exception()

    # online version incorporates n_ways and feeds support sets sequentially
    if setting == Settings.ONLINE_FEW_SHOT:
        if batch_size != (n_ways * k_shots):
            raise Exception("Onine N-way-K-shot must be reflected in batch size!")
        raise Exception("Must modify this function to include OFSL via dataset_name == master")
        master_train_set = None # will be a concatenated sequence of datasets -> shuffled sequence of support sets across all datasets
        master_ordered_labels = None # keep a set of ordered labels for the dataset; for FSL Subset

    ds = HDF5_Dataset(f"data/HDF5/{dataset_name}/{TDS_model_id}/", split)

    if subset_id is not None:
        with open(os.path.join("data", "subsets", f"{dataset_name}_{subset_id}.pkl"), "rb") as f:
            subset_indices = pickle.load(f)
        ds = Subset(ds, subset_indices)

    ds_len = len(ds)

    if return_replay_loader:
        ds = ConcatDataset([ds, replay_buffer])
        wandb.log({"ds_len": ds_len, "concat_ds_len": len(ds)})

    ds_len = len(ds)

    # sequence of datasets/loaders
    # if setting in [Settings.SEQ_FEW_SHOT, Settings.SEQ_FULL_SHOT]:
    #     # take a subset of the train set if doing sequential FSL
    #     if setting == Settings.SEQ_FEW_SHOT and split == "train":
    #         FSL_indices = get_FSL_indices(ordered_train_labels, k_shots, ds_seed)
    #         ds_rng.shuffle(FSL_indices) # do not do this in the online scenario
    #         dataset = Subset(dataset, FSL_indices)
    #         print(f"{dataset_name} FSL len: {len(dataset)} ({k_shots}-shot)")

    loader = DataLoader(
        ds, 
        shuffle=False, # DistributedSampler will shuffle
        batch_size=batch_size, 
        pin_memory=False, 
        sampler=DistributedSampler(ds) if split == "train" else DistributedEvalSampler(ds)
    )

    """  
    Old OFSL code

    # single dataset/loader for all datasets
    elif setting == Settings.ONLINE_FEW_SHOT:
        if master_train_set is None:
            master_train_set = train_set
            master_ordered_labels = ordered_train_labels
        else:
            master_train_set = ConcatDataset([master_train_set, train_set])

            # need to offset the current train_set's ordered labels
            unique_classes_so_far = len(np.unique(master_ordered_labels))
            master_ordered_labels.extend([label + unique_classes_so_far for label in ordered_train_labels])

    # if online FSL, need to create few shot version of the master dataset
    if setting == Settings.ONLINE_FEW_SHOT:
        FSL_indices = get_FSL_indices(master_ordered_labels, k_shots, ds_seed)
        master_train_set = Subset(master_train_set, FSL_indices)

        # one giant loader for all datasets; shuffle=False is important here
        dataloader_dict["train"][0] = {
            "name": "master",
            "loader": ToDeviceLoader(DataLoader(master_train_set, shuffle=False, batch_size=train_batch_size, pin_memory=True), device),
        }
        print(f"OFSL train set length: {len(master_train_set)}; num classes: {len(np.unique(master_ordered_labels))}")
    """

    return loader, ds_len

