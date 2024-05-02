"""
    Code for properly converting all datasets to TensorDatasets
"""
import os, torch, pickle, json, time, sys
from torch.utils.data import TensorDataset
from PIL import Image
import numpy as np
from transformers import OwlViTProcessor
import copy
import h5py


MAX_PROMPTS = 126 # 1 prepended padding prompt, 75 max prompts, 50 negatives
MAX_OBJECTS = 2023


def get_max_prompt_lens():
    """
    Ensured (post-facto) that none of our prompts were truncated
    """
    for ds_info_path in [os.path.join("ds_info", ds_info_name) for ds_info_name in os.listdir("ds_info")]:
        ds_name = ds_info_path.split("/")[-1][:-4]
        print("----", ds_name)
        
        with open(ds_info_path, "rb") as f:
            d = pickle.load(f)
        
        if "train" not in d:
            print("No train:", ds_name)
            continue

        max_prompt_len = -1
        l = None
        
        for img_dict in d["train"].values():
            if isinstance(img_dict, str):
                continue
            for prompt in img_dict["imagenet_prompts"]:
                if len(prompt.split(" ")) > max_prompt_len:
                    max_prompt_len = len(prompt.split(" "))
                    l = prompt

            try:
                for prompt in img_dict["custom_prompts"]:
                    if len(prompt.split(" ")) > max_prompt_len:
                        max_prompt_len = len(prompt.split(" "))
                        l = prompt
            except KeyError:
                continue

        print(max_prompt_len, l)


def get_padding_stats():
    """
    Get the two stats that are necessary for padding each dataset, namely the max number of prompts for a single image (across all datasets) + the max
    number of objects in a single image
    """
    max_prompts_per_image = -1
    max_objects_per_image = -1
    max_prompts_ds = None
    max_objects_ds = None
    for ds_info_path in [os.path.join("ds_info", ds_info_name) for ds_info_name in os.listdir("ds_info")]:
        ds_name = ds_info_path.split("/")[-1][:-4]
        print("----", ds_name)
        
        with open(ds_info_path, "rb") as f:
            d = pickle.load(f)
        
        if "train" not in d:
            print("No train:", ds_name)
            continue


        max_prompts_per_image_ds = -1
        max_objects_per_image_ds = -1
        no_obj_count = 0
        no_prompt_count = 0
        
        for img_dict in d["train"].values():
            if isinstance(img_dict, str):
                continue
            try:
                if len(img_dict["labels"]) > max_objects_per_image_ds:
                    max_objects_per_image_ds = len(img_dict["labels"])
                if len(img_dict["labels"]) > max_objects_per_image:
                    max_objects_per_image = len(img_dict["labels"])
                    max_objects_ds = ds_name
            except KeyError:
                no_obj_count += 1
                assert "boxes" not in img_dict

            try:
                if len(img_dict["imagenet_prompts"]) > max_prompts_per_image_ds:
                    max_prompts_per_image_ds = len(img_dict["imagenet_prompts"])
                if len(img_dict["imagenet_prompts"]) > max_prompts_per_image:
                    max_prompts_per_image = len(img_dict["imagenet_prompts"])
                    max_prompts_ds = ds_name
            except KeyError:
                no_prompt_count += 1
                assert "boxes" not in img_dict and "labels" not in img_dict

        print(f"No obj: {no_obj_count} No prompt: {no_prompt_count}")
        print(f"Max prompts: {max_prompts_per_image_ds}")
        print(f"Max objects: {max_objects_per_image_ds}")
        
    print("--------------------")
    print(f"Max prompts: {max_prompts_per_image} ({max_prompts_ds})")
    print(f"Max objects: {max_objects_per_image} ({max_objects_ds})")


class DatasetTensorizer():
    """
    Loading in an object detection dataset in the standard fashion such that it can be preprocessed and stored in TDS or HDF5 format
    Essentially for storing datasets in a preprocessed form on disk
    Assumptions:
        1. All images have the same number of prompts (and therefore input_ids of the same dimensionality); no need to pad input_ids
    """
    def __init__(self, info_dict_path, processor, has_custom_prompts, skip_existing=False):
        with open(info_dict_path, "rb") as f:
            info_dict = pickle.load(f)

        self.dataset_name = info_dict_path.split("/")[-1].split(".")[0]
        self.skip_existing = skip_existing
        self.info_dict = info_dict
        self.train_dict = info_dict["train"] if "train" in info_dict else None
        self.processor = processor
        self.has_custom_prompts = has_custom_prompts

        print("------", self.dataset_name)


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


    def file_setup(self, model_name, split):
        ds_dir = f"../data/TensorDatasets/{self.dataset_name}/"
        ds_root = os.path.join(ds_dir, model_name)
        os.makedirs(ds_root, exist_ok=True)

        if self.skip_existing and os.path.isfile(os.path.join(ds_root, f"{self.dataset_name}_{split}.pth")):
            print("Skipping", os.path.join(ds_root, f"{self.dataset_name}_{split}.pth"))
            return None, True

        return ds_root, False


    def store_train_TDS(self, model_name):
        """
        Store the train set as a TensorDataset
        ****Assumes the whole tensorized/preprocessed dataset can fit in RAM****
        """
        ds_root, skip = self.file_setup(model_name, "train")

        if skip:
            return
        
        total_images = len(self.train_dict) - 1 # img_pth entry

        start = time.time()

        tds_dict = { # will be preallocated
            "images": None,
            "image_ids": -1 * torch.ones((total_images), dtype=torch.int),
            "imagenet_input_ids": None,
            "imagenet_input_ids": None,
            "imagenet_attention_masks": None,
            "boxes": None,
            "labels": None
        }

        if self.has_custom_prompts:
            tds_dict["custom_input_ids"] = None
            tds_dict["custom_attention_masks"] = None

        img_pth = self.train_dict.pop("img_pth")

        # create the TDS in a reproducible order
        train_items = sorted(
            [(img_id, img_dict) for img_id, img_dict in self.train_dict.items()], 
            key=lambda item: item[0])

        for idx, (img_id, img_dict) in enumerate(train_items):
            # load in the PIL image, get label_names, and initialize labels and boxes (will generally need padding)
            current_image = Image.open(os.path.join(img_pth, img_dict["file_name"]).replace("data/", "")).convert("RGB")

            current_imagenet_prompts = img_dict["imagenet_prompts"]

            if self.has_custom_prompts:
                current_custom_prompts = img_dict["custom_prompts"]
                current_prompts = [""] + current_imagenet_prompts + current_custom_prompts # includes padding prompt
            else:
                current_prompts = [""] + current_imagenet_prompts

            # run the image and prompts thru the HF processor
            try:
                preprocessed_inputs = self.processor(text=current_prompts, images=current_image, return_tensors="pt", truncation=False)
            except ValueError:
                assert self.dataset_name in ["plantdoc", "openPoetryVision"]
                for i in range(len(current_prompts)):
                    out = self.processor(text=current_prompts[i], return_tensors="pt", truncation=False)
                    if out["input_ids"].size(-1) > 16:
                        if current_prompts[i].startswith("a jpeg corrupted"):
                            current_prompts[i] = current_prompts[i].replace(" jpeg corrupted", "")
                        elif current_prompts[i].startswith("a photo of a hard to see"):
                            current_prompts[i] = current_prompts[i].replace(" hard to see", "")
                        elif current_prompts[i].startswith("a photo of the hard to see"):
                            current_prompts[i] = current_prompts[i].replace(" hard to see", "")
                        elif current_prompts[i].startswith("a black and white photo"):
                            current_prompts[i] = current_prompts[i].replace(" black and white", "")
                        elif current_prompts[i].startswith("a close-up photo"):
                            current_prompts[i] = current_prompts[i].replace(" close-up", "")    
                
                preprocessed_inputs = self.processor(text=current_prompts, images=current_image, return_tensors="pt", truncation=False)

            current_labels = self.get_labels(img_dict["labels"] if "labels" in img_dict else None)
            current_boxes = self.pad_boxes(img_dict["boxes"] if "boxes" in img_dict else None)

            current_image = preprocessed_inputs["pixel_values"]

            # organize input_ids
            if self.has_custom_prompts:
                num_each_prompt_type = int(len(preprocessed_inputs["input_ids"]) / 2)

                current_imagenet_input_ids = preprocessed_inputs["input_ids"][:num_each_prompt_type]
                current_imagenet_attention_masks = preprocessed_inputs["attention_mask"][:num_each_prompt_type]

                current_custom_input_ids = preprocessed_inputs["input_ids"][num_each_prompt_type:]
                current_custom_attention_masks = preprocessed_inputs["attention_mask"][num_each_prompt_type:]

                current_custom_input_ids, current_custom_attention_masks = self.pad_prompts(current_custom_input_ids, current_custom_attention_masks)

                assert len(current_imagenet_input_ids) != len(current_custom_input_ids), f"{len(current_imagenet_input_ids)}, {len(current_custom_input_ids)}"
            else:
                current_imagenet_input_ids = preprocessed_inputs["input_ids"]
                current_imagenet_attention_masks = preprocessed_inputs["attention_mask"]

            current_imagenet_input_ids, current_imagenet_attention_masks = self.pad_prompts(current_imagenet_input_ids, current_imagenet_attention_masks)

            # may need to initialize the TDS tensors
            if tds_dict["images"] is None:
                tds_dict["images"] = torch.zeros((total_images, *current_image.size())).squeeze()
                tds_dict["boxes"] = torch.zeros((total_images, *current_boxes.size())).squeeze()
                tds_dict["labels"] = torch.zeros((total_images, *current_labels.size())).squeeze()

                tds_dict["imagenet_input_ids"] = torch.zeros((total_images, *current_imagenet_input_ids.size()), dtype=torch.long).squeeze()
                tds_dict["imagenet_attention_masks"] = torch.zeros((total_images, *current_imagenet_attention_masks.size()), dtype=torch.long).squeeze()

                if self.has_custom_prompts:
                    tds_dict["custom_input_ids"] = torch.zeros((total_images, *current_custom_input_ids.size()), dtype=torch.long).squeeze()
                    tds_dict["custom_attention_masks"] = torch.zeros((total_images, *current_custom_attention_masks.size()), dtype=torch.long).squeeze()

                for key, val in tds_dict.items():
                    print(key, val.size())
                    assert not (key.startswith("custom") and not self.has_custom_prompts) 

            # append the current datapoint's tensors to the TDS
            tds_dict["images"][idx] = current_image.squeeze()
            tds_dict["image_ids"][idx] = img_id
            tds_dict["imagenet_input_ids"][idx] = current_imagenet_input_ids.squeeze()
            tds_dict["imagenet_attention_masks"][idx] = current_imagenet_attention_masks.squeeze()
            tds_dict["boxes"][idx] = current_boxes.squeeze()
            tds_dict["labels"][idx] = current_labels.squeeze()

            if self.has_custom_prompts:
                tds_dict["custom_input_ids"][idx] = current_custom_input_ids.squeeze()
                tds_dict["custom_attention_masks"][idx] = current_custom_attention_masks.squeeze()

            if idx % 250 == 0:
                print(f"Progress: {idx}/{len(train_items)}")

        assert idx == tds_dict["images"].size(0) - 1

        # save to disk as a TDS
        if self.has_custom_prompts:
            tds = TensorDataset(
                tds_dict["images"], 
                tds_dict["image_ids"], 
                tds_dict["imagenet_input_ids"], 
                tds_dict["imagenet_attention_masks"],
                tds_dict["custom_input_ids"],
                tds_dict["custom_attention_masks"],
                tds_dict["boxes"],
                tds_dict["labels"]
            )
        else:
            tds = TensorDataset(
                tds_dict["images"], 
                tds_dict["image_ids"], 
                tds_dict["imagenet_input_ids"], 
                tds_dict["imagenet_attention_masks"],
                tds_dict["boxes"],
                tds_dict["labels"]
            )
            
        torch.save(tds, os.path.join(ds_root, f"{self.dataset_name}_train.pth"))

        print(f"Took {(time.time() - start) / 60} minutes in total")


    def store_eval_TDS(self, model_name, img_pth, split, id_to_img_pth_12=False):
        """
        Store the val/test set as a TensorDataset
        ****Assumes the whole tensorized/preprocessed dataset can fit in RAM****
        """
        ds_root, skip = self.file_setup(model_name, split)

        if skip:
            return
        
        start = time.time()

        with open(self.info_dict["json_paths"][split].replace("data/", ""), "r") as f:
            j_images = json.load(f)["images"]

        idxes = list(range(len(j_images)))
        rng = np.random.default_rng(seed=10)
        rng.shuffle(idxes)

        total_images= len(j_images)

        tds_dict = { # will be preallocated
            "images": None,
            "image_ids": -1 * torch.ones((total_images), dtype=torch.int)
        }

        for idx in idxes:
            if not id_to_img_pth_12:
                current_img_pth = os.path.join(img_pth, j_images[idx]["file_name"]).replace("data/", "")
            else:
                current_img_pth = os.path.join(img_pth, "".join(["0" for _ in range(12 - len(str(j_images[idx]["id"])))]) + str(j_images[idx]["id"]) + ".jpg")

            img_id = j_images[idx]["id"]

            current_image = Image.open(current_img_pth).convert("RGB")
            preprocessed_inputs = self.processor(images=current_image, return_tensors="pt")

            current_image = preprocessed_inputs["pixel_values"]

            if tds_dict["images"] is None:
                tds_dict["images"] = torch.zeros((total_images, *current_image.size()))

                if total_images > 1:
                    tds_dict["images"] = tds_dict["images"].squeeze()
                else:
                    print(f"Only {total_images} {split} image!")

            tds_dict["images"][idx] = current_image
            tds_dict["image_ids"][idx] = img_id

        tds = TensorDataset(
            tds_dict["images"], 
            tds_dict["image_ids"]
        )
            
        torch.save(tds, os.path.join(ds_root, f"{self.dataset_name}_{split}.pth"))

        print(f"Took {(time.time() - start) / 60} minutes in total")


    """
    def store_as_hdf5(self, model_name, items_per_file:int, use_custom_prompts:bool=False):
        # Store the dataset as a set of HDF5 files
        # Does not assume the dataset can fit in RAM
        ds_dir = f"../data/coresets/{self.dataset_name}/"
        ds_root = os.path.join(ds_dir, model_name)
        os.makedirs(ds_root, exist_ok=True)

        # now loop thru the entire dataset and save to tensors
        start = time.time()
        file_count = 0
        current_len = 0 # current length of current HDF5 file
        buffed_images, buffed_imagenet_input_ids, buffed_custom_input_ids, buffed_attention_masks, buffed_boxes, buffed_labels = None, None, None, None, None, None
        for idx in range(len(self.image_files)):
            current_img_file = self.image_files[idx]
            current_img_dict = self.image_dicts[current_img_file]
            
            # load in the PIL image, get label_names, and initialize labels and boxes (will generally need padding)
            current_image = Image.open(os.path.join("../", self.img_pth, current_img_file)).convert("RGB")
            current_imagenet_prompts = current_img_dict["imagenet_prompts"]
            if use_custom_prompts: # Objects365 does not use custom prompts
                current_custom_prompts = current_img_dict["custom_prompts"]
            current_boxes = current_img_dict["boxes"] if "boxes" in current_img_dict else torch.Tensor([[-1.0,-1.0,-1.0,-1.0]])
            current_labels = current_img_dict["labels"] if "labels" in current_img_dict else torch.Tensor([])

            # pad labels and boxes as necessary
            if current_boxes.size(0) < self.max_labels_len:
                padded_boxes = self.pad_single_sequence(current_boxes, out_len=self.max_labels_len, padding_value=-1.0)
                padded_labels = self.pad_single_sequence(current_labels, out_len=self.max_labels_len, padding_value=-1.0)
            elif current_boxes.size(0) > self.max_labels_len:
                raise Exception

            # run the image and label names thru the HF processor
            current_prompts = current_imagenet_prompts if not use_custom_prompts else current_imagenet_prompts + current_custom_prompts 
            preprocessed_inputs = self.processor(text=current_prompts, images=current_image, return_tensors="pt")

            # get the datapoint's tensors
            current_image = preprocessed_inputs["pixel_values"]
            if use_custom_prompts:
                current_imagenet_input_ids = preprocessed_inputs["input_ids"][:int(len(preprocessed_inputs["input_ids"]) / 2)]
                current_custom_input_ids = preprocessed_inputs["input_ids"][int(len(preprocessed_inputs["input_ids"]) / 2):]
            else:
                current_imagenet_input_ids = preprocessed_inputs["input_ids"]
            current_attention_masks = preprocessed_inputs["attention_mask"]
            current_boxes = padded_boxes
            current_labels = padded_labels

            if use_custom_prompts and (len(current_imagenet_input_ids) != len(current_custom_input_ids)):
                raise Exception(f"{len(current_imagenet_input_ids)}, {len(current_custom_input_ids)}")

            # may need to initialize the TDS tensors
            if buffed_images is None:
                buffed_images = torch.zeros((items_per_file, *current_image.size())).squeeze()
                buffed_custom_input_ids = torch.zeros((items_per_file, *current_custom_input_ids.size())).squeeze()
                buffed_imagenet_input_ids = torch.zeros((items_per_file, *current_imagenet_input_ids.size())).squeeze()
                buffed_attention_masks = torch.zeros((items_per_file, *current_attention_masks.size())).squeeze()
                buffed_boxes = torch.zeros((items_per_file, *current_boxes.size())).squeeze()
                buffed_labels = torch.zeros((items_per_file, *current_labels.size())).squeeze()

                print(
                    buffed_images.size(),
                    buffed_custom_input_ids.size(),
                    buffed_imagenet_input_ids.size(),
                    buffed_attention_masks.size(),
                    buffed_boxes.size(),
                    buffed_labels.size()
                )
            
            # append the current datapoint's tensors to the TDS
            buffed_images[current_len] = current_image.squeeze()
            buffed_custom_input_ids[current_len] = current_custom_input_ids.squeeze()
            buffed_imagenet_input_ids[current_len] = current_imagenet_input_ids.squeeze()
            buffed_attention_masks[current_len] = current_attention_masks.squeeze()
            buffed_boxes[current_len] = current_boxes.squeeze()
            buffed_labels[current_len] = current_labels.squeeze()

            # get features for current image
            

            current_len += 1

            # may need to save the current file
        #     if current_len == items_per_file - 1:
        #         print(f"Dumping file {idx // current_len + 1}. Should match {file_count}")
        #         dump_start = time.time()
        #         file_path = os.path.join(ds_root, f"{self.dataset_name}_X_{items_per_file}_{file_count}.hdf5")
        #         self.store_data_in_hdf5({
        #             "images": f_images, 
        #             "input_ids": f_input_ids, 
        #             "attention_masks": f_attention_masks, 
        #             "boxes": f_boxes, 
        #             "labels": f_labels
        #         }, items_per_file, file_path)
        #         print(f"Took {time.time() - dump_start}s to dump")

        #         file_count += 1 
        #         current_len = 0
        
        # if current_len > 0:
        #     print("Performing final coreset dump")
        #     dump_start = time.time()
        #     file_path = os.path.join(ds_root, f"{self.dataset_name}_X_{int(current_len)}_{file_count}.hdf5")
        #     self.store_data_in_hdf5({
        #         "images": f_images, 
        #         "input_ids": f_input_ids, 
        #         "attention_masks": f_attention_masks, 
        #         "boxes": f_boxes, 
        #         "labels": f_labels
        #     }, current_len, file_path)
        #     print(f"Took {time.time() - dump_start}s to dump")


        print(f"Took {(time.time() - start) / 60} minutes in total")


    def store_data_in_hdf5(self, data_dict:dict, expected_items:int, hdf5_file:str):
        # data_dict is a dict of tensors
        # NOTE: due to how we handle y.shape below, this currently only supports vector y (no scalars)
        if not all([val.shape[0] == expected_items for val in data_dict.values()]):
            raise Exception(f"Shape mismatch: {[val.shape for val in data_dict.values()]}")

        with h5py.File(hdf5_file, "w") as f:
            for key, val in data_dict.items():
                f.create_dataset(key, data=val.numpy(), chunks=tuple([1, *val.shape[1:]]))
    """

def store_TDSes(ds_name_lst:list, skip_existing:bool):
    ds_info_paths = sorted([os.path.join("ds_info", info_dict) for info_dict in os.listdir("ds_info")])
    ds_info_paths = list(filter(
        lambda x: any([ds_name in x for ds_name in ds_name_lst]), ds_info_paths
    ))
    assert len(ds_info_paths) == len(ds_name_lst), f"{len(ds_info_paths)} {len(ds_name_lst)}"
    process_rank = int(sys.argv[1])
    num_processes = int(sys.argv[2])
    do_eval_TDSes = bool(int(sys.argv[3]))
    for i in range(process_rank, len(ds_info_paths), num_processes):
        ds_info_path = ds_info_paths[i]
        if all(ds_name not in ds_info_path for ds_name in ["O365", "VG", "LVIS"]):
            for model_id, processor_id in [("OWL_L14", "google/owlvit-large-patch14")]:
                processor = OwlViTProcessor.from_pretrained(processor_id)
                tensorizer = DatasetTensorizer(ds_info_path, processor, True, skip_existing=skip_existing)
                print(model_id)
                print("train")
                tensorizer.store_train_TDS(model_id)
                if do_eval_TDSes:
                    for split in ["val", 'test']:
                        print(split)
                        json_pth = tensorizer.info_dict["json_paths"][split]
                        img_pth = json_pth.replace(json_pth.split("/")[-1], "")
                        if "PascalVOC" in ds_info_path and split == "val":
                            img_pth += "train"
                        tensorizer.store_eval_TDS(model_id, img_pth, split)
        elif do_eval_TDSes and "VG" not in ds_info_path:    
            for model_id, processor_id in [("OWL_L14", "google/owlvit-large-patch14")]:
                processor = OwlViTProcessor.from_pretrained(processor_id)
                tensorizer = DatasetTensorizer(ds_info_path, processor, False, skip_existing=skip_existing)
                print(model_id)
                for split in ["val", 'test']:
                    print(split)
                    if tensorizer.dataset_name == "LVIS":
                        img_pth = f"LVIS/{split.replace('val', 'train').replace('test', 'val')}2017"
                    if tensorizer.dataset_name == 'O365':
                        img_pth = f"O365/{split.replace('val', 'train').replace('test', 'val')}"
                    tensorizer.store_eval_TDS(model_id, img_pth, split, id_to_img_pth_12=(tensorizer.dataset_name == "LVIS"))
        else:
            print("---- Skipping", ds_info_path)


def manually_store_TDSes_in_parallel():
    big_configs = [
        ("VG", "train", "OWL_B16", "google/owlvit-base-patch16"),
        ("VG", "train", "OWL_L14", "google/owlvit-large-patch14")
    ]

    process_rank = int(sys.argv[1])
    process_conf = big_configs[process_rank]

    print(process_conf)

    ds_name, split, model_id, processor_id = process_conf

    processor = OwlViTProcessor.from_pretrained(processor_id)

    for ds_info_path in sorted([os.path.join("ds_info", info_dict) for info_dict in os.listdir("ds_info")]):
        if ds_name in ds_info_path:
            if all(ds_name not in ds_info_path for ds_name in ["O365", "VG", "LVIS"]):
                tensorizer = DatasetTensorizer(ds_info_path, processor, True, skip_existing=False)
                if split == "train":
                    tensorizer.store_train_TDS(model_id)
                else:
                    json_pth = tensorizer.info_dict["json_paths"][split]
                    img_pth = json_pth.replace(json_pth.split("/")[-1], "")

                    if ds_name == "PascalVOC" and split == "val":
                        img_pth = json_pth.replace(json_pth.split("/")[-1], "train")

                    tensorizer.store_eval_TDS(model_id, img_pth, split)
            elif "VG" not in ds_name:
                tensorizer = DatasetTensorizer(ds_info_path, processor, False, skip_existing=False)
                json_pth = tensorizer.info_dict["json_paths"][split]
                if tensorizer.dataset_name == 'O365':
                    img_pth = f"O365/{split.replace('val', 'train').replace('test', 'val')}"
                tensorizer.store_eval_TDS(model_id, img_pth, split)


def TDS_to_HDF5(tds_pth:str, skip_existing:bool):
    hdf5_dir = tds_pth.replace("TensorDatasets", "HDF5").replace(tds_pth.split("/")[-1], "")
    print(hdf5_dir)
    dump_prefix = os.path.join(hdf5_dir, tds_pth.split("/")[-1].replace(".pth", ""))

    if os.path.isfile(dump_prefix + f"_0"):
        if skip_existing:
            print("Skipping", dump_prefix)
            return
        
    os.system(f"rm {dump_prefix}*")
    os.makedirs(hdf5_dir, exist_ok=True)

    start = time.time()
    tds = torch.load(tds_pth)
    if "train" in tds_pth.split("/")[-1].split("_")[-1]:
        if len(tds.tensors) == 8:
            tds_keys = ["images", "image_ids", "imagenet_input_ids", "imagenet_attention_masks", "custom_input_ids", "custom_attention_masks", "boxes", "labels"]
        else:
            tds_keys = ["images", "image_ids", "imagenet_input_ids", "imagenet_attention_masks", "boxes", "labels"]
    else:
        tds_keys = ["images", "image_ids"]

    assert len(tds_keys) == len(tds.tensors), f"{len(tds_keys)} {len(tds.tensors)}"

    hdf5_dict = {
        tds_keys[i]: tds.tensors[i] for i in range(len(tds.tensors))
    }

    if list(hdf5_dict["images"].shape[0:2]) == [1,1]:
        hdf5_dict["images"] = hdf5_dict["images"].squeeze(0)

    if len(tds_keys) > 2:
        assert all([hdf5_dict["images"].shape[1] == 3, len(hdf5_dict["image_ids"].shape) == 1, hdf5_dict["boxes"].shape[-1] == 4])
    else: 
        assert len(hdf5_dict["images"].shape) in [3,4], str(hdf5_dict["images"].shape)

    if len(tds) < 5000:
        store_data_in_hdf5(hdf5_dict, len(tds), dump_prefix + "_0")
    else:
        idx_ranges = [i for i in range(0, len(tds) + 1, 5000)]
        idx_ranges += [len(tds)] if len(tds) % 5000 > 0 else []
        for start_idx_i in range(len(idx_ranges) - 1):
            if start_idx_i == idx_ranges[-1]:
                break
            start_idx = idx_ranges[start_idx_i]
            end_idx = idx_ranges[start_idx_i + 1]
            hdf5_dict = {
                tds_keys[i]: tds.tensors[i][start_idx:end_idx] for i in range(len(tds.tensors))
            }
            store_data_in_hdf5(hdf5_dict, end_idx - start_idx, dump_prefix + f"_{start_idx_i}")
    print(f"Took {time.time() - start}s")


def store_data_in_hdf5(data_dict:dict, expected_items:int, hdf5_file:str):
    # data_dict is a dict of tensors
    # NOTE: due to how we handle y.shape below, this currently only supports vector y (no scalars)
    if not all([val.shape[0] == expected_items for val in data_dict.values()]):
        raise Exception(f"Shape mismatch: {[val.shape for val in data_dict.values()]}")

    with h5py.File(hdf5_file, "w") as f:
        for key, val in data_dict.items():
            f.create_dataset(key, data=val.numpy(), chunks=tuple([1, *val.shape[1:]]))


def convert_all_to_HDF5(tds_root:str, skip_existing:bool, skip_pattern:str=None):
    ds_dirs = [os.path.join(tds_root, d) for d in os.listdir(tds_root)]
    
    model_dirs = []
    for d in ds_dirs:
        model_dirs += [os.path.join(d, m) for m in os.listdir(d)]
    
    tds_paths = []
    for m in model_dirs:
        tds_paths += [os.path.join(m, t) for t in os.listdir(m)]

    process_rank = int(sys.argv[1])
    total_processes = int(sys.argv[2])

    do_eval = bool(int(sys.argv[3]))
    if not do_eval:
        tds_paths = list(filter(
            lambda tds: "val.pth" not in tds and "test.pth" not in tds,
            tds_paths
        ))

    if skip_pattern is not None:
        tds_paths = list(filter(
            lambda tds: skip_pattern not in tds,
            tds_paths
        ))

    for i in range(process_rank, len(tds_paths), total_processes):
        TDS_to_HDF5(tds_paths[i], skip_existing)


def convert_stragglers_to_HDF5(tds_root:str, skip_existing:bool):
    ds_dirs = [os.path.join(tds_root, d) for d in os.listdir(tds_root)]
    
    model_dirs = []
    for d in ds_dirs:
        model_dirs += [os.path.join(d, m) for m in os.listdir(d)]
    
    tds_paths = []
    for m in model_dirs:
        tds_paths += [os.path.join(m, t) for t in os.listdir(m)]

    stragglers = [
        ('test', 'PascalVOC', 'OWL_L14'),
        ('val', 'PascalVOC', 'OWL_L14'),
        ('train', 'PascalVOC', 'OWL_L14')
    ]

    tds_paths = list(filter(lambda t: any([
        s[0] in t and s[1] in t and s[2] in t for s in stragglers
    ]), tds_paths))

    assert len(stragglers) == len(tds_paths), f"{len(tds_paths)} {len(stragglers)}"

    process_rank = int(sys.argv[1])
    total_processes = int(sys.argv[2])

    for i in range(process_rank, len(tds_paths), total_processes):
        TDS_to_HDF5(tds_paths[i], skip_existing)


if __name__ == "__main__":
    if sys.argv[4] == "tds":
        #ds_names_lst = ["LVIS", "O365", "PascalVOC", 'plantdoc', 'ThermalCheetah', 'BCCD', 'AerialMaritimeDrone', 'OxfordPets', 'dice', 'brackishUnderwater', 'pothole', 'WildfireSmoke', 'ChessPieces', 'thermalDogsAndPeople', 'ShellfishOpenImages', 'Aquarium', 'pistols', 'EgoHands', 'openPoetryVision', 'AmericanSignLanguageLetters']
        ds_names_lst = ["PascalVOC"]
        store_TDSes(ds_names_lst, skip_existing=False)
    elif sys.argv[4] == "hdf5":
        #convert_stragglers_to_HDF5("TensorDatasets", skip_existing=False)
        convert_all_to_HDF5("TensorDatasets", False, skip_pattern="OWL_B16")


"""
---- DroneControl
No obj: 376 No prompt: 0
Max prompts: 58
Max objects: 3
---- boggleBoards
No obj: 0 No prompt: 0
Max prompts: 86
Max objects: 36
---- ThermalCheetah
No obj: 3 No prompt: 0
Max prompts: 52
Max objects: 7
---- PKLot
No obj: 189 No prompt: 0
Max prompts: 52
Max objects: 100
---- BCCD
No obj: 0 No prompt: 0
Max prompts: 53
Max objects: 30
---- O365_all
No obj: 74 No prompt: 0
Max prompts: 83
Max objects: 835
---- AerialMaritimeDrone
No obj: 0 No prompt: 0
Max prompts: 55
Max objects: 80
---- OxfordPets
No obj: 0 No prompt: 0
Max prompts: 87
Max objects: 2
---- CottontailRabbits
No obj: 0 No prompt: 0
Max prompts: 51
Max objects: 2
---- dice
No obj: 0 No prompt: 0
Max prompts: 56
Max objects: 45
---- Raccoon
No obj: 0 No prompt: 0
Max prompts: 51
Max objects: 3
---- NorthAmericaMushrooms
No obj: 0 No prompt: 0
Max prompts: 52
Max objects: 7
---- brackishUnderwater
No obj: 1772 No prompt: 0
Max prompts: 56
Max objects: 21
---- pothole
No obj: 0 No prompt: 0
Max prompts: 51
Max objects: 19
---- WildfireSmoke
No obj: 0 No prompt: 0
Max prompts: 51
Max objects: 1
---- ChessPieces
No obj: 0 No prompt: 0
Max prompts: 63
Max objects: 32
---- MountainDewCommercial
No obj: 0 No prompt: 0
Max prompts: 51
Max objects: 51
---- thermalDogsAndPeople
No obj: 11 No prompt: 0
Max prompts: 52
Max objects: 3
---- O365
No train: O365
---- ShellfishOpenImages
No obj: 0 No prompt: 0
Max prompts: 53
Max objects: 27
---- Aquarium
No obj: 1 No prompt: 0
Max prompts: 57
Max objects: 56
---- pistols
No obj: 0 No prompt: 0
Max prompts: 51
Max objects: 14
---- VehiclesOpenImages
No obj: 0 No prompt: 0
Max prompts: 55
Max objects: 9
---- EgoHands
No obj: 10 No prompt: 0
Max prompts: 54
Max objects: 4
---- openPoetryVision
No obj: 0 No prompt: 0
Max prompts: 93
Max objects: 5
---- AmericanSignLanguageLetters
No obj: 0 No prompt: 0
Max prompts: 76
Max objects: 1
---- plantdoc
No obj: 9 No prompt: 0
Max prompts: 80
Max objects: 42
---- websiteScreenshots
No obj: 0 No prompt: 0
Max prompts: 58
Max objects: 2023
---- MaskWearing
No obj: 0 No prompt: 0
Max prompts: 52
Max objects: 113
---- PascalVOC
No obj: 310 No prompt: 0
Max prompts: 70
Max objects: 42
---- Packages
No obj: 0 No prompt: 0
Max prompts: 51
Max objects: 2
---- LVIS
No train: LVIS
---- selfdrivingCar
No obj: 2844 No prompt: 0
Max prompts: 61
Max objects: 32
---- HardHatWorkers
No obj: 0 No prompt: 0
Max prompts: 53
Max objects: 35
--------------------
Max prompts: 93 (openPoetryVision)
Max objects: 2023 (websiteScreenshots)
"""