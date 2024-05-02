"""
Utilities for generating coresets from object detection datasets
Coresets are stored as many hdf5 files
"""
import torch
from torch.utils.data import Dataset, TensorDataset
import time, pickle, os, sys, random
from transformers import OwlViTProcessor
import numpy as np
import h5py
from PIL import Image
import multiprocessing
import copy
import wandb

from importlib.machinery import SourceFileLoader
try:
    models = SourceFileLoader("models", "models/models.py" if __name__ != "__main__" else "../models/models.py").load_module()
    trainer_utils = SourceFileLoader("trainer_utils", "torch_replay_utils/trainer_utils.py" if __name__ != "__main__" else "trainer_utils.py").load_module()
except:
    # for multiprocessing
    models = SourceFileLoader("models", "../models/models.py").load_module()
    trainer_utils = SourceFileLoader("trainer_utils", "trainer_utils.py").load_module()


MAX_PROMPTS = 126 # 1 prepended padding prompt, 75 max prompts, 50 negatives
MAX_OBJECTS = 2023
CORESET_ROOT = "data/coresets"
REQUIRED_PROCESSES = 24
GPU_ID = None
N_INSTANCES = 50
N_FEATURES = 8
PROTOTYPE_N = 6
items_per_file = 2500


class Coreset(Dataset):
    """
    Object for randomly accessing items in a giant coreset composed of hdf5 files
    For storing a pre-specified set of parallel tensors (ex: Xy, features)
    Features are stored separate from Xy since they will all be loaded into RAM 
    """
    def __init__(self, coreset_name:str, data_type:str, model_name:str="OWL_B16", return_idx=False):
        if not data_type in ["features", "Xy"]:
            raise Exception(f"Invalid Coreset data_type: {data_type}")

        coreset_dir = os.path.join(CORESET_ROOT, coreset_name, model_name)
        hdf5_file_list = sorted(
            [os.path.join(coreset_dir, fname) for fname in os.listdir(coreset_dir) if fname.startswith(f"{coreset_name}_{data_type}_")],
            key=lambda f: int(f.split("_")[-1])
        )

        if data_type == "features":
            self.matrix_keys = [("cls_embeds", torch.float32), ("features", torch.float32), ("queries", torch.float32)]
        else:
            self.matrix_keys = [
                ("images", torch.float32),
                ("image_ids", torch.int),
                ("imagenet_input_ids", torch.long),
                ("imagenet_attention_masks", torch.long),
                ("boxes", torch.float32),
                ("labels", torch.int),
                ("logits", torch.float32)
            ]

        self.data_type = data_type
        self.return_idx = return_idx
        self.hdf5_file_list = hdf5_file_list
        self.start_indices = []

        # Calculate the starting index of each file in the combined dataset
        total_length = 0
        for hdf5_file in self.hdf5_file_list:
            self.start_indices.append(total_length)
            with h5py.File(hdf5_file, 'r') as f:
                total_length += f[self.matrix_keys[0][0]].shape[0]
        self.length = total_length

        self.dummy_text_inputs = torch.zeros((126,16), dtype=torch.long)

    def get_file_idx(self, global_idx):
        for file_i in range(len(self.start_indices)):
            if file_i == len(self.start_indices) - 1 or self.start_indices[file_i + 1] > global_idx:
                return file_i
        raise StopIteration

    def get_all_items(self):
        all_items = []
        for hdf5_file in self.hdf5_file_list:
            file_items = []
            with h5py.File(hdf5_file, 'r') as f:
                for key, return_dtype in self.matrix_keys:
                    entire_matrix = np.array(f[key])  # Load the entire matrix into a numpy array
                    tensor_matrix = torch.tensor(entire_matrix, dtype=return_dtype)  # Convert to tensor
                    file_items.append(tensor_matrix)
            all_items.append(file_items)

        # Concatenating along the first dimension (which usually represents the batch size)
        return [torch.cat([file_items[i] for file_items in all_items], dim=0) for i in range(len(self.matrix_keys))]

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

        if self.data_type == "Xy":
            if self.return_idx:
                return [idx] + return_list[:4] + [self.dummy_text_inputs, self.dummy_text_inputs] + return_list[4:] + [torch.tensor(True)]
            else:
                return return_list[:4] + [self.dummy_text_inputs, self.dummy_text_inputs] + return_list[4:] + [torch.tensor(True)]
        elif self.data_type == 'features':
            return return_list
        else:
            raise Exception(self.data_type)


def unravel_index_2D(indices:torch.Tensor, shape:torch.Size):
    """
    Torch implementation of numpy unravel_index for two dimensions
    """
    assert len(shape) == 2 and len(indices.shape) == 1

    shape = torch.tensor(shape)
    rows = indices // shape[1]
    cols = indices % shape[1]

    return (rows, cols)


def get_top_k_idxes(logits:torch.Tensor, exclusive_classes:bool, n_features:int):
    """
    Finds the top k scores and corresponding boxes within an image.

    The code applies on the image level; must use vmap for batching.
    
    Args:
    scores: [num_instances, num_classes] array of scores (i.e. logits or
    probabilities) to sort by.
    boxes: [num_instances, 4] Optional array of bounding boxes.
    k: Number of instances to return.
    exclusive_classes: If True, the top class for each box is returned. If
    False, classes are considered to be non-exclusive (multi-label setting),
    and the top-k computations happens globally across all scores, not just
    the maximum logit for each output token.

    Returns:
    Score, label, and box arrays of shape [top_k, ...] for the selected
    instances.
    """
    assert len(logits.shape) == 2

    k = n_features

    if exclusive_classes:
        k = min(k, logits.shape[0]) # k cannot be greater than the number of ViT tokens
        instance_top_scores, instance_class_ind = torch.max(logits, dim=1) # (num ViT tokens), (num ViT tokens)
        top_scores, instance_ind = torch.topk(instance_top_scores, k) # (k), (k)
        class_ind = instance_class_ind[instance_ind] # (k)
    else:
        k = min(k, logits.numel()) # k cannot be greater than the total number of class predictions
        top_scores, top_indices = torch.topk(logits.view(-1), k) # (k), (k)
        instance_ind, class_ind = unravel_index_2D(top_indices, logits.shape) # (k), (k)

    return instance_ind, class_ind


class CoresetGenerator():
    """
    Loading in an object detection dataset in the standard fashion such that it can be preprocessed and stored in HDF5 format
    Essentially for storing datasets in a preprocessed form on disk
    Assumptions:
        1. All images have the same number of prompts (and therefore input_ids of the same dimensionality); no need to pad input_ids
    """
    def __init__(self, info_dict_paths:list, processor, skip_existing=False):
        self.info_dicts = []
        for i in info_dict_paths:
            with open(i, "rb") as f:
                self.info_dicts.append(pickle.load(f)["train"])

        print("Loaded info dicts")

        self.skip_existing = skip_existing
        self.processor = processor


    def pad_single_sequence(self, sequence:torch.Tensor, *, out_len:int, padding_value:float):
        """
        Take an n-dimensional sequence (ex: 1D for labels and 2D for boxes) and pad it with padding_value-filled tensors

        Args:
            sequence: n-dim Tensor representing a single sequence (ex: labels for an image)
            out_len: length of each padded sequence
            batch_first: if True, the input and output tensor's first dimension will be batch size.
            padding_value: the value to be filled in the padded positions.
        Returns:
            A tensor of all input sequences padded to the length of the longest sequence in the input.
        """
        if len(sequence.shape) == 1:
            return torch.cat([sequence, torch.ones(out_len - sequence.size(0)) * padding_value])
        else:
            element_dim = sequence.shape[1:]
            return torch.cat([sequence, torch.full((out_len - sequence.size(0), *element_dim), padding_value)], dim=0)


    def get_labels(self, labels):
        """
        Process labels into a multi-hot tensor
        """
        padded_multihot = torch.zeros((MAX_OBJECTS, MAX_PROMPTS), dtype=torch.int)

        if labels is None: # no objects in image
            padded_multihot[:, 0] = 1
        elif isinstance(labels, torch.Tensor): # one label per object
            assert len(labels.shape) == 1
            
            padded_multihot[
                torch.arange(0, labels.size(0), dtype=torch.int), labels.to(torch.int) + 1
            ] = 1

            # add padding
            if labels.size(0) != MAX_OBJECTS:
                padded_multihot[
                    torch.arange(labels.size(0), MAX_OBJECTS, dtype=torch.int), torch.zeros((MAX_OBJECTS - labels.size(0)), dtype=torch.int)
                ] = 1

        elif isinstance(labels, list): # multiple labels per object
            for obj_i, obj_labels in enumerate(labels):
                assert isinstance(obj_labels, list), f"{type(obj_labels)}"

                padded_multihot[obj_i, obj_labels] = 1
            
            # add padding
            if len(labels) != MAX_OBJECTS:
                padded_multihot[
                    torch.arange(len(labels), MAX_OBJECTS, dtype=torch.int), torch.zeros((MAX_OBJECTS - len(labels)), dtype=torch.int)
                ] = 1
        else:
            raise Exception
        
        return padded_multihot
                

    def pad_boxes(self, boxes):
        """
        Pad boxes with zeros
        """
        padded_boxes = torch.zeros((MAX_OBJECTS, 4))

        if boxes is not None:
            assert len(boxes.shape) == 2, f"{boxes.shape}"

            padded_boxes[:boxes.size(0)] = boxes

        return padded_boxes


    def pad_prompts(self, input_ids, attention_masks):
        """
        Pad prompts and attention masks with zeroes
        """
        padded_input_ids = torch.zeros((MAX_PROMPTS, input_ids.size(-1)), dtype=torch.long)
        padded_attention_masks = torch.zeros((MAX_PROMPTS, attention_masks.size(-1)), dtype=torch.long)

        padded_input_ids[:input_ids.size(0)] = input_ids
        padded_attention_masks[:attention_masks.size(0)] = attention_masks

        return padded_input_ids, padded_attention_masks
    

    def pad_logits(self, logits):
        """
        Pad logits with zeroes
        """
        b, tokens, prompts = logits.shape
        padding = torch.zeros((b, tokens, MAX_PROMPTS - prompts), dtype=logits.dtype)
        return torch.cat([logits.cpu(), padding], dim=-1)


    def get_cls_denom(self, labels:torch.Tensor):
        if labels.numel() == 0:
            cls_denom = torch.tensor([1.])
        else:
            cls_denom = torch.maximum(torch.tensor([labels[...,1:].sum()]), torch.tensor([1.])).to(torch.float)

        return cls_denom
        

    def get_box_denom(self, boxes:torch.Tensor, labels:torch.Tensor):
        if boxes.numel() == 0:
            box_denom = torch.tensor([1.])
        else:
            n_labels_per_instance = torch.sum(labels[..., 1:], dim=-1)
            box_denom = torch.tensor([torch.sum(n_labels_per_instance > 0, dtype=torch.float)])
            box_denom = torch.maximum(box_denom, torch.tensor([1.]))

        return box_denom


    def add_denoms(self, y:dict):
        """
        Given local boxes and labels, add shared (across devices) denominators for box and label loss normalization
        The shared value of cls_denom is the average number of positive labels (1s) per-image across devices
        The shared value of box_denom is the average number of boxes per-image across devices
        """
        cls_denom = self.get_cls_denom(y["labels"])
        box_denom = self.get_box_denom(y["boxes"], y["labels"])

        y["cls_denom"] = cls_denom
        y["box_denom"] = box_denom


    @torch.no_grad()
    def process_img_dict(self, img_pth, img_dict, model, exclusive_classes:bool, loss_only:bool, compute_PEs:bool):
        # load in the PIL image, get label_names, and initialize labels and boxes (will generally need padding)
        current_image = Image.open(os.path.join("../data/", img_pth, img_dict["file_name"])).convert("RGB")
        current_prompts = [""] + img_dict["imagenet_prompts"]

        # run the image and prompts thru the HF processor
        preprocessed_inputs = self.processor(text=current_prompts, images=current_image, return_tensors="pt", truncation=True)

        current_labels = self.get_labels(img_dict["labels"] if "labels" in img_dict else None)
        current_boxes = self.pad_boxes(img_dict["boxes"] if "boxes" in img_dict else None)

        current_image = preprocessed_inputs["pixel_values"]

        current_imagenet_input_ids, current_imagenet_attention_masks = self.pad_prompts(
            preprocessed_inputs["input_ids"], preprocessed_inputs["attention_mask"])

        # get logits
        model_out = model(
            current_image.to(GPU_ID), 
            current_imagenet_input_ids.to(GPU_ID), 
            current_imagenet_attention_masks.to(GPU_ID)
        )
        coreset_logits = self.pad_logits(model_out["pred_logits"])

        # get features
        query_embeds = model.get_text_embeddings(
            current_imagenet_input_ids.to(GPU_ID), current_imagenet_attention_masks.to(GPU_ID))
        image_features = model.get_image_features(current_image.to(GPU_ID), return_cls_token=False)
        pred_boxes = model.get_boxes(image_features)
        pred_logits, class_embeds = model.get_logits(image_features, query_embeds, return_embeds=True)
        sig_logits = torch.nn.functional.sigmoid(pred_logits)
        
        top_k_feature_idxes, top_k_query_idxes = get_top_k_idxes(
            sig_logits.squeeze(0), exclusive_classes, N_FEATURES)

        assert list(class_embeds.shape[:2]) == list(image_features.shape[:2]), f"{class_embeds.shape[:2]} {image_features.shape[:2]}"

        top_k_class_embeds = class_embeds.squeeze(0)[top_k_feature_idxes]
        top_k_features = image_features.squeeze(0)[top_k_feature_idxes]
        top_k_query_embeds = query_embeds[top_k_query_idxes]

        assert len(top_k_class_embeds.unsqueeze(0).size()) == len(top_k_features.unsqueeze(0).size()), f"{top_k_class_embeds.unsqueeze(0).size()}, {top_k_features.unsqueeze(0).size()}"

        current_features = {
            "cls_embeds": top_k_class_embeds.unsqueeze(0), 
            "features": top_k_features.unsqueeze(0), 
            "queries": top_k_query_embeds.unsqueeze(0)
        }

        # get top-(min(n, # boxes for class)) class embeddings for each class for future prototype computation
        proto_embeds = {}
        if compute_PEs and "labels" in img_dict:
            for l_i, l in enumerate(img_dict["label_names"][:int(max(img_dict["labels"])) + 1]):
                n_embeds = min(PROTOTYPE_N, int(torch.sum(img_dict["labels"] == l_i))) # compute min(n, # boxes for class)
                actual_logit_idx = l_i + 1 # must account for padding prompt
                top_n_idxes = sig_logits[:, :, actual_logit_idx].topk(k=n_embeds, dim=1)[1].squeeze() # get indices of top n preds for this class
                proto_embeds[l] = torch.index_select(class_embeds, dim=1, index=top_n_idxes).squeeze(0) # get the vectors
                assert list(proto_embeds[l].shape) == [n_embeds, class_embeds.size(-1)], str(proto_embeds[l].shape)

        # get loss and (maybe) classes in the current image
        y = {"boxes": current_boxes, "labels": current_labels}
        self.add_denoms(y)
        y = {k: v.unsqueeze(0).to(GPU_ID) for k, v in y.items()}
        current_loss, _, __ = model.loss_func({"pred_logits": pred_logits, "pred_boxes": pred_boxes} | y)
        current_loss = current_loss.item()
        
        if not loss_only:
            try:
                last_positive_label_idx = int(max(img_dict["labels"])) if not isinstance(img_dict["labels"][0], list) else int(max([max(l) for l in img_dict["labels"] if len(l) > 0]))
                classes_in_image = img_dict["label_names"][:last_positive_label_idx]
            except KeyError:
                classes_in_image = []

            current_stats = {
                "loss": current_loss, 
                "classes": classes_in_image
            }
        else:
            current_stats = current_loss

        return (
            current_image, 
            current_imagenet_input_ids, 
            current_imagenet_attention_masks, 
            current_boxes, 
            current_labels, 
            coreset_logits, 
            current_features, 
            current_stats,
            proto_embeds
        )


    @torch.no_grad()
    def store(
        self, 
        coreset_name, 
        model_name, 
        model, 
        process_rank:int, 
        process_indices:list, 
        file_idxes:list, 
        items_per_file:int, 
        exclusive_classes:list,
        loss_only:list,
        compute_PEs:list
    ):
        """
        Store the coreset in HDF5 format
        """
        ds_dir = f"../data/coresets/{coreset_name}/"
        ds_root = os.path.join(ds_dir, model_name)
        os.makedirs(ds_root, exist_ok=True)

        start = time.time()

        process_items = sum([len(x) for x in process_indices])
        device = torch.device(f"cuda:{GPU_ID}")

        tds_dict = {
            "images": None,
            "image_ids": None,
            "imagenet_input_ids": None,
            "imagenet_input_ids": None,
            "imagenet_attention_masks": None,
            "boxes": None,
            "labels": None,
            "logits": None
        }

        all_features = {
            "cls_embeds": torch.zeros((process_items, N_FEATURES, model.query_emb_dim), device=device),
            "features": torch.zeros((process_items, N_FEATURES, model.num_features), device=device),
            "queries": torch.zeros((process_items, N_FEATURES, model.query_emb_dim), device=device)
        }

        proto_embeds = {}

        stats = {} # will hold loss + classes represented for each image

        current_len = 0
        total_len = 0
        files_saved = 0
        for train_dict_idx, train_dict in enumerate(self.info_dicts):
            img_pth = train_dict.pop("img_pth")

            # create in a reproducible order
            train_items = sorted(
                [(img_id, img_dict) for img_id, img_dict in train_dict.items()], 
                key=lambda item: int(item[0])
            )
            
            # start = time.time()
            for idx, (_, img_dict) in enumerate([train_items[i] for i in process_indices[train_dict_idx]]):
                (current_image, 
                current_imagenet_input_ids, 
                current_imagenet_attention_masks, 
                current_boxes, 
                current_labels, 
                current_logits, 
                current_features, 
                current_stats,
                current_proto_embeds) = self.process_img_dict(
                    img_pth, img_dict, model, exclusive_classes[train_dict_idx], loss_only[train_dict_idx], compute_PEs[train_dict_idx])
                
                proto_embeds[total_len] = {}
                for cls_name, embeds in current_proto_embeds.items():
                    proto_embeds[total_len][cls_name] = embeds

                for k, v in current_features.items():
                    all_features[k][total_len] = v

                stats[total_len] = current_stats

                # may need to initialize the TDS tensors
                if tds_dict["images"] is None:
                    tds_dict["images"] = torch.zeros((items_per_file, *current_image.size())).squeeze()
                    tds_dict["image_ids"] = torch.zeros((items_per_file,)).squeeze()
                    tds_dict["boxes"] = torch.zeros((items_per_file, *current_boxes.size())).squeeze()
                    tds_dict["labels"] = torch.zeros((items_per_file, *current_labels.size())).squeeze()
                    tds_dict["imagenet_input_ids"] = torch.zeros((items_per_file, *current_imagenet_input_ids.size()), dtype=torch.long).squeeze()
                    tds_dict["imagenet_attention_masks"] = torch.zeros((items_per_file, *current_imagenet_attention_masks.size()), dtype=torch.long).squeeze()
                    tds_dict["logits"] = torch.zeros((items_per_file, *current_logits.size())).squeeze()

                    for key, val in tds_dict.items():
                        print(key, val.size())

                # append the current datapoint's tensors to the TDS
                tds_dict["images"][current_len] = current_image.squeeze()
                tds_dict["image_ids"][current_len] = -1
                tds_dict["imagenet_input_ids"][current_len] = current_imagenet_input_ids.squeeze()
                tds_dict["imagenet_attention_masks"][current_len] = current_imagenet_attention_masks.squeeze()
                tds_dict["boxes"][current_len] = current_boxes.squeeze()
                tds_dict["labels"][current_len] = current_labels.squeeze()
                tds_dict["logits"][current_len] = current_logits.squeeze()

                current_len += 1
                total_len += 1

                if current_len == items_per_file:
                    self.store_data_in_hdf5(
                        tds_dict, 
                        items_per_file, 
                        os.path.join(ds_root, f"{coreset_name}_Xy_{file_idxes[files_saved]}")
                    )

                    torch.save(
                        proto_embeds,
                        os.path.join(ds_root, f"{coreset_name}_PEs_{process_rank}_{files_saved}.pth")
                    )

                    proto_embeds = {}

                    current_len = 0
                    files_saved += 1

                # print(time.time() - start, GPU_ID)
                # start = time.time()

                if idx % 5000 == 0:
                    print(f"{idx}/{len(process_indices[train_dict_idx])}, {train_dict_idx + 1}/{len(self.info_dicts)}")

        if current_len > 0:
            self.store_data_in_hdf5(
                {k: v[:current_len] for k, v in tds_dict.items()}, 
                current_len, 
                os.path.join(ds_root, f"{coreset_name}_Xy_{file_idxes[files_saved]}")
            )

            torch.save(
                proto_embeds,
                os.path.join(ds_root, f"{coreset_name}_PEs_{process_rank}_{files_saved}.pth")
            )

        # store all features for this process in one go
        self.store_data_in_hdf5(
            all_features, 
            total_len, 
            os.path.join(ds_root, f"{coreset_name}_features_{process_rank}")
        )      

        # store stats
        with open(f"{coreset_name}_stats_{process_rank}.pkl", "wb") as f:
            pickle.dump({"stats": stats}, f)  

        print(f"Took {(time.time() - start) / 60} minutes in total")


    def store_data_in_hdf5(self, data_dict:dict, expected_items:int, hdf5_file:str):
        # data_dict is a dict of tensors
        # NOTE: due to how we handle y.shape below, this currently only supports vector y (no scalars)
        if not all([val.shape[0] == expected_items for val in data_dict.values()]):
            raise Exception(f"Shape mismatch: {[val.shape for val in data_dict.values()]}")

        with h5py.File(hdf5_file, "w") as f:
            for key, val in data_dict.items():
                f.create_dataset(key, data=val.cpu().numpy(), chunks=tuple([1, *val.shape[1:]]))


def get_process_indices(cg:CoresetGenerator, total_processes:int, items_per_file:int):
    all_indices = [[None for j in range(len(cg.info_dicts))] for i in range(total_processes)]
    for d_i, d in enumerate(cg.info_dicts):
        for process_rank in range(total_processes):
            all_indices[process_rank][d_i] = range(process_rank, len(d) - 1, total_processes)

    all_file_idxes = [[] for _ in range(total_processes)]
    last_file_idx = -1
    for process_rank in range(total_processes):
        all_file_idxes[process_rank].append(last_file_idx + 1)
        process_items = sum([len(x) for x in all_indices[process_rank]]) - items_per_file
        while process_items > 0:
            all_file_idxes[process_rank].append(all_file_idxes[process_rank][-1] + 1)
            process_items -= items_per_file
        last_file_idx = all_file_idxes[process_rank][-1]

    print("Total items:", sum([sum([len(x) for x in y]) for y in all_indices]))
    print("All file idxes:", all_file_idxes)

    return all_indices, all_file_idxes


def get_process_start_idx(process_rank, all_indices):
    start_idx = 0
    for i in range(process_rank):
        start_idx += sum([len(x) for x in all_indices[i]])
    return start_idx


def store_coreset():
    model_name = sys.argv[2]
    process_rank = int(sys.argv[3])
    total_processes = int(sys.argv[4])
    
    global GPU_ID
    GPU_ID = int(sys.argv[5])

    if model_name == "OWL_B16":
        processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch16")
        model = models.OWL_B16().to(torch.device(f"cuda:{GPU_ID}"))
    elif model_name == "OWL_L14":
        processor = OwlViTProcessor.from_pretrained("google/owlvit-large-patch14")
        model = models.OWL_L14().to(torch.device(f"cuda:{GPU_ID}"))

    cg = CoresetGenerator(
        [os.path.join("../data/ds_info/", x) for x in ["VG.pkl", "O365.pkl"]],
        processor
    )

    all_indices, all_file_idxes = get_process_indices(cg, total_processes, items_per_file)

    print("Process items:", sum([len(x) for x in all_indices[process_rank]]))
    print("Process file idxes:", all_file_idxes[process_rank])

    model.precompute_box_bias(1)
    cg.store("O365_VG", model_name, model, process_rank, all_indices[process_rank], all_file_idxes[process_rank], items_per_file, [False, True], [True, False], [False, True])


def merge_stats(num_processes:int):
    o365_master, vg_master = {}, {}
    current_len = 0
    for i in range(num_processes):
        print(i)
        with open(f"O365_VG_stats_{i}.pkl", "rb") as f:
            p = pickle.load(f)["stats"]
        sorted_keys = sorted(list(p.keys()))
        for i in sorted_keys:
            if isinstance(p[i], dict):
                o365_master[current_len] = copy.deepcopy(p[i])
            else:
                vg_master[current_len] = p[i]
            current_len += 1

    with open(f"O365_stats.pkl", "wb") as f:
        pickle.dump(o365_master, f)

    with open(f"VG_stats.pkl", "wb") as f:
        pickle.dump(vg_master, f)


def apply_loss_threshold(stats:dict, threshold:float, loss_only:bool):
    if loss_only:
        return {k: v for k, v in stats.items() if v < threshold}
    else:
        return {k: copy.deepcopy(v) for k, v in stats.items() if v["loss"] < threshold}


def get_class_counts(all_classes:list, subset_stats:dict):
    class_counts = {}
    for img_dict in subset_stats.values():
        assert len(img_dict["classes"]) == len(set(img_dict["classes"])), str(img_dict["classes"])
        for cls_name in img_dict['classes']:
            if cls_name in class_counts:
                class_counts[cls_name] += 1
            else:
                class_counts[cls_name] = 1
    
    for cls_name in all_classes:
        if cls_name not in class_counts:
            class_counts[cls_name] = 0

    return class_counts


def fix_underrepresented_classes(base_class_counts:dict, full_stats:dict, subset_stats:dict, subset_class_counts:dict) -> list:
    new_indices = {} # dict for O(1) lookup

    global N_INSTANCES

    sorted_full_stats_keys = [k for k, v in sorted(
        [(i, j) for i, j in full_stats.items()], 
        key=lambda t: t[1]["loss"])
    ]

    sorted_full_stats = [(k, full_stats[k]) for k in sorted_full_stats_keys]

    print(sorted_full_stats[0])

    subset_class_counts_cpy = copy.deepcopy(subset_class_counts)

    underrep_count = 0
    for cls_name, cls_count in subset_class_counts_cpy.items():
        if cls_count < N_INSTANCES and cls_count < base_class_counts[cls_name]:
            underrep_count += 1
            new_cls_count = cls_count
            while new_cls_count < N_INSTANCES and new_cls_count < base_class_counts[cls_name]:
                for k, v in sorted_full_stats:
                    if (k not in subset_stats) and (k not in new_indices) and (cls_name in v["classes"]):
                        new_indices[k] = None
                        for c in v["classes"]:
                            subset_class_counts_cpy[c] += 1

                        new_cls_count += 1

    for k in new_indices:
        subset_stats[k] = full_stats[k]


def get_O365_indices(stats:dict):
    """
    If a class has less than n images in the coreset, make sure it has n or add max images
    """
    thresholds = [0.2, 0.225, 0.25, 0.275, 0.3]
    base_class_counts = get_class_counts([], stats)
    all_classes = list(base_class_counts.keys())
    indices = {}
    for thresh in thresholds:
        subset_stats = apply_loss_threshold(stats, thresh, False)
        print("O365", thresh, len(subset_stats))

        subset_class_counts = get_class_counts(all_classes, subset_stats)
        print(np.histogram(list(subset_class_counts.values()), bins=[0,1,5,20,50,100,300, 1000, 5000, 10000, 10000]))

        fix_underrepresented_classes(base_class_counts, stats, subset_stats, subset_class_counts)        
        print("---")
        subset_class_counts = get_class_counts(all_classes, subset_stats)
        print(np.histogram(list(subset_class_counts.values()), bins=[0,1,5,20,50,100,300, 1000, 5000, 10000, 10000]))
        subset_indices = list(subset_stats.keys())
        assert len(subset_indices) == len(set(subset_indices))
        print(len(subset_stats), len(subset_indices))

        indices[thresh] = subset_indices

    for thresh in indices.keys():
        indices[thresh] = [i for i in indices[thresh] if len(stats[i]["classes"]) > 0]
        print(thresh, "after no-class removal:", len(indices[thresh]))

    return indices


def get_VG_indices(stats:dict):
    print("--------------------")
    thresholds = [1.7]
    indices = {}
    for thresh in thresholds:
        filtered_stats = apply_loss_threshold(stats, thresh, True)
        print("VG", thresh, len(filtered_stats))
        indices[thresh] = list(filtered_stats.keys())
    
    return indices


def get_coreset_indices():
    with open(f"O365_stats.pkl", "rb") as f:
        o365_stats = pickle.load(f)
    with open(f"VG_stats.pkl", "rb") as f:
        VG_stats = pickle.load(f)

    for i in range(len(o365_stats) + len(VG_stats)):
        if i not in o365_stats and i not in VG_stats:
            raise Exception

    assert all([i in o365_stats or i in VG_stats for i in range(len(o365_stats) + len(VG_stats))]), "Missing indices"

    o365_indices = get_O365_indices(o365_stats)
    VG_indices = get_VG_indices(VG_stats)

    return o365_indices, VG_indices


def store_sweep_indices():
    all_o365_indices, all_VG_indices = get_coreset_indices()

    sweep_thresholds = [
        (0.2, 0.0),
        (0.225, 0.0),
        (0.25, 0.0),
        (0.275, 0.0),
        (0.3, 0.0)
    ]

    for thresh_pair in sweep_thresholds:
        o365_thresh, VG_thresh = thresh_pair

        o365_indices = all_o365_indices[o365_thresh] if o365_thresh != 0.0 else []
        VG_indices = all_VG_indices[VG_thresh] if VG_thresh != 0.0 else []

        print(thresh_pair, len(o365_indices) + len(VG_indices))

        with open(f"idx_o365_{o365_thresh}_VG_{VG_thresh}_N{N_INSTANCES}.pkl", "wb") as f:
            pickle.dump(o365_indices + VG_indices, f)


def store_feature_TDS():
    model_name = sys.argv[2]
    coreset_name = "O365_VG"

    global CORESET_ROOT
    CORESET_ROOT = "../data/coresets"
    ds = Coreset(coreset_name, "features", model_name)
    print(len(ds))
    all_features = ds.get_all_items()
    for tens in all_features:
        print(tens.size())
    assert len(all_features) == 3
    tds = TensorDataset(all_features[0], all_features[1], all_features[2])
    
    ds_dir = f"../data/coresets/{coreset_name}/"
    ds_root = os.path.join(ds_dir, model_name)
    torch.save(tds, os.path.join(ds_root, f"features_TDS.pth"))


def load_feature_TDS(coreset_name, model_name):
    ds_root = os.path.join("../" if __name__ == "__main__" else "", CORESET_ROOT, coreset_name, model_name)
    return torch.load(os.path.join(ds_root, f"features_TDS.pth"))


def load_prototypes(coreset_name, model_name, map_location):
    ds_root = os.path.join("../" if __name__ == "__main__" else "", CORESET_ROOT, coreset_name, model_name)
    return torch.load(os.path.join(ds_root, "prototypes.pth"), map_location=map_location)


def functional_compute_SVs(
    ev_features:torch.Tensor, # (B, N, F)
    ev_queries:torch.Tensor, # (B, N, Q)
    cand_features:torch.Tensor, # (C, M, F)
    cand_queries:torch.Tensor, # (C, M, Q)
    original_indices:torch.Tensor, # (C)
    SV_type:str, # ["batch", "buffer"]
    chunked:bool,
    chunk_sizes:list,
    K:int,
    avg_across_tokens:bool,
    device
    ):
    """
    Faithful kNN-SV computation; assumes a unified candidate set
    """
    if chunked: # controllably less memory-intensive
        feature_l2 = trainer_utils.batched_dist(ev_features, cand_features, "L2", chunk_sizes)
    else: # most memory-intensive but fastest
        feature_l2 = trainer_utils.naive_dist(ev_features, cand_features, "L2")

    # get the min distance of features for each image
    feature_l2, indices = feature_l2.min(dim=-1) # (B, N, C), (B, N, C)

    if chunked: # controllably less memory-intensive
        query_sim = trainer_utils.batched_sim_with_indices(ev_queries, cand_queries, indices, chunk_sizes)
    else: # most memory-intensive but fastest
        query_sim = trainer_utils.naive_sim_with_indices(ev_queries, cand_queries, indices)

    dist_stats = {
        "SV_PC_L2_min": feature_l2.min(),
        "SV_PC_L2_max": feature_l2.max(),
        "SV_PC_L2_mean": feature_l2.mean(),
        "SV_PC_L2_std": feature_l2.std(),
        "SV_PC_qsim_min": query_sim.min(),
        "SV_PC_qsim_max": query_sim.max(),
        "SV_PC_qsim_mean": query_sim.mean(),
        "SV_PC_qsim_std": query_sim.std()
    }

    # for each batch item, sort the images in descending order by feature similarity
    feature_l2, feature_indices = feature_l2.sort(dim=-1, descending=True) # (B, N, C), (B, N, C)

    # must permute coreset indices to maintain their validity
    coreset_indices = torch.broadcast_to(
        original_indices.unsqueeze(0).unsqueeze(0),
        feature_l2.size()
    ).gather(-1, feature_indices)

    assert torch.unique(coreset_indices, dim=-1).numel() == coreset_indices.numel()

    # now align query similarities with feature distances
    query_sim = query_sim.gather(-1, feature_indices)

    # compute ASV for each token for each image
    sorted_l2s = torch.stack([feature_l2, query_sim], dim=-1) # (B, N, C, 2)

    # assert (sorted_l2s[:,:,:,1][torch.isinf(sorted_l2s[:,:,:,0])] == 0.0).all()

    current_SVs = torch.zeros(sorted_l2s.shape[:-1]).to(device) # (B, N, C)
    n_cand = current_SVs.size(-1)
    last_SV = torch.zeros_like(current_SVs[:-1]) # (B, N)
    last_query_sim = torch.zeros_like(current_SVs[:-1]) # (B, N)
    for SV_i in range(sorted_l2s.size(-2)):
        if SV_i == 0:
            current_SVs[:,:,SV_i] = sorted_l2s[:,:,SV_i,1] / n_cand
        else:
            m = n_cand - SV_i
            current_SVs[:,:,SV_i] = last_SV + (((sorted_l2s[:,:,SV_i,1] - last_query_sim) / K) * (min(K, m) / m))

        last_SV = current_SVs[:,:,SV_i]
        last_query_sim = sorted_l2s[:,:,SV_i,1]

    rank_stats = {
        "SV_PC_avg_rank": current_SVs.max(dim=-1)[1].to(torch.float).mean() if SV_type == "buffer" else current_SVs.min(dim=-1)[1].to(torch.float).mean(),
        "SV_PC_min_rank": current_SVs.max(dim=-1)[1].to(torch.float).min() if SV_type == "buffer" else current_SVs.min(dim=-1)[1].to(torch.float).min(),
        "SV_PC_max_rank": current_SVs.max(dim=-1)[1].to(torch.float).max() if SV_type == "buffer" else current_SVs.min(dim=-1)[1].to(torch.float).max()
    }

    # first need to realign the SVs along dim 1
    aligned_coreset_indices, sort_indices = torch.sort(coreset_indices, dim=-1)
    aligned_SVs = current_SVs.gather(-1, sort_indices)

    # now need to min/max/avg each image's shapley value along the token/feature dimension
    if avg_across_tokens:
        SVs = torch.mean(aligned_SVs, dim=1)
    else:
        if SV_type == "batch":
            SVs, _ = torch.min(aligned_SVs, dim=1)
        elif SV_type == "buffer":
            SVs, _ = torch.max(aligned_SVs, dim=1)
        else:
            raise Exception(SV_type)

    coreset_indices = aligned_coreset_indices[0][0]

    return SVs, coreset_indices, rank_stats | dist_stats


def precompute_shapley_values(feature_type:str, coreset_name, model_name, o365_thresh, VG_thresh, coreset_N, n_tok):
    # load in the features and queries of the coreset
    full_features = load_feature_TDS(coreset_name, model_name)

    indices_pth = f"idx_o365_{o365_thresh}_VG_{VG_thresh}_N{coreset_N}.pkl"
    with open(indices_pth, "rb") as f:
        subset_indices = pickle.load(f)

    print(f"Features GB: {sum([t.numel() * 4 for t in full_features.tensors]) / 1000000000}")

    if feature_type == "cls_embeds":
        cpu_features = full_features.tensors[0][subset_indices, :n_tok]
    elif feature_type == "features":
        cpu_features = full_features.tensors[1][subset_indices, :n_tok]
    else:
        raise Exception
    cpu_queries = full_features.tensors[2][subset_indices, :n_tok]

    # launch distributed buffer SV computation
    n_devices = torch.cuda.device_count()
    n_indices_per_device = int(cpu_features.size(0) / n_devices) + 1
    device_indices = []
    for i in range(n_devices):
        start = i * n_indices_per_device
        end = (i+1) * n_indices_per_device
        end = end if end <= cpu_features.size(0) else cpu_features.size(0)
        device_indices.append(list(range(start, end)))

    args = [
        (cpu_features, cpu_queries, device_indices[i], i) for i in range(n_devices)
    ]
    
    multiprocessing.set_start_method('spawn', force=True)

    with multiprocessing.Pool(processes=n_devices) as pool:
        results = pool.starmap(local_precompute_SVs, args)
        pool.close()
        pool.join()

    """
    aggregated_stats = {}

    # Aggregate statistics from each process
    for process_stats in results:
        for key, value in process_stats.items():
            if key not in aggregated_stats:
                aggregated_stats[key] = []
            aggregated_stats[key].extend(value)

    for k, v in aggregated_stats.items():
        try:
            aggregated_stats[k] = torch.cat(v)
        except:
            aggregated_stats[k] = torch.tensor(v)
    torch.save(aggregated_stats, "pc_stats.pth")
    """
    
    pc_SVs = torch.zeros((cpu_features.size(0),))
    rank_weights = []
    for i in range(n_devices):
        d = torch.load(f'tmp_pc_{i}.pth', map_location=torch.device(0))
        rank_weight = torch.tensor(d["numel"] / cpu_features.size(0))
        rank_weights.append(rank_weight)
        pc_SVs += (d["avg_local_SVs"].to(torch.device(0)) * rank_weight.to(torch.device(0))).cpu()

    print(f"Rank weight sum: {sum(rank_weights)}")

    # store in a labeled file
    pc_SV_pth = f'pc_C_SVs_{feature_type}_o365_{o365_thresh}_VG_{VG_thresh}_N{coreset_N}_{n_tok}.pth'
    torch.save(pc_SVs, pc_SV_pth)
    print('Completed')


def local_precompute_SVs(cpu_features, cpu_queries, local_indices, rank):
    device = torch.device(rank)
    local_SVs = torch.zeros((len(local_indices), cpu_features.size(0)), device=device)

    global_features = cpu_features.to(device)
    global_queries = cpu_queries.to(device)

    local_features = global_features[local_indices]
    local_queries = global_queries[local_indices]

    batch_size = 500
    original_indices = torch.arange(0, global_features.size(0)).to(device)
    all_stats = {}
    for i in range(0, len(local_indices), batch_size):
        endpoint = i+batch_size if i+batch_size <= len(local_indices) else len(local_indices)
        current_features = local_features[i:endpoint]
        current_queries = local_queries[i:endpoint]
        
        SVs, indices, stats = functional_compute_SVs(
            current_features,
            current_queries,
            global_features,
            global_queries,
            original_indices=original_indices,
            SV_type="buffer",
            chunked=True,
            chunk_sizes=[100, 500],
            K=20,
            avg_across_tokens=False,
            device=device
        )

        for k in stats:
            if k not in all_stats:
                all_stats[k] = []
            all_stats[k].append(stats[k])

        assert torch.sum(torch.sort(indices)[0] == original_indices) == indices.numel()
        
        local_SVs[i:endpoint] = SVs
        print(rank, i)

    averaged_local_SVs = local_SVs.mean(dim=0)

    tmp_pc_SV_pth = f'tmp_pc_{rank}.pth'
    torch.save({"avg_local_SVs": averaged_local_SVs, "numel": len(local_indices)}, tmp_pc_SV_pth)

    sv_stats = {
        "SV_PC_min": averaged_local_SVs.min(),
        "SV_PC_max": averaged_local_SVs.max(),
        "SV_PC_mean": averaged_local_SVs.mean(),
        "SV_PC_std": averaged_local_SVs.std()
    } 
    
    for k, v in all_stats.items():
        try:
            torch.cat(v)
            all_stats[k] = torch.cat(v)
        except:
            all_stats[k] = torch.tensor(v)

    return all_stats | sv_stats


if __name__ == "__main__":
    # if sys.argv[1] == "Xyf":
    #     store_coreset()
    # elif sys.argv[1] == 'ftds':
    #     store_feature_TDS()
    # exit(0)
    
    precompute_shapley_values(
        sys.argv[1],
        "O365_VG",
        "OWL_L14",
        0.15,
        0.0,
        50,
        n_tok=int(sys.argv[2])
    )

    exit(0)

    # n_items = 206473
    # n_devices = 8
    # optimized = False
    # o365_thresh = 0.9
    # VG_thresh = 0.0
    # coreset_N = 50

    # # store in a labeled file
    # pc_SV_pth = f'pc_{"CO" if optimized else "C"}_SVs_o365_{o365_thresh}_VG_{VG_thresh}_N{coreset_N}.pth'
    # pc_SVs = torch.load(pc_SV_pth, map_location=torch.device(0)).cpu().numpy()

    # print(np.histogram(pc_SVs))