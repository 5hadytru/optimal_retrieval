"""
Code for generating ds_infos for ODinW. Laboriously executed.
NOTE: prompts are not added here
{
    [split]: {
        img_pth: str,
        [image_id]: {
            labels: Tensor
            label_names: List[str]
            file_name: str
            imagenet_prompts: List[str]
            custom_prompts: List[str]
        }
    },
    best_7_imagenet_prompts: Dict[List]
    custom_prompts: Dict[List]
    imagenet_prompts: Dict[List]
}
"""

import json, sys, os, pickle
import pprint
import torch
import numpy as np
from PIL import Image, ImageDraw

root = "odinw"

# change these
ds_name = "ChessPieces"
data_folder = os.path.join(ds_name, "Chess Pieces.v23-raw.coco")
single_folder = False
augment_queries = True

def query_aug(caption):
    return caption.replace("-", " ").replace("_", " ") + " chess piece"

splits = ["train", "val", "test"]
master_dict = {split: {} for split in splits}
master_dict["json_paths"] = {}

for split in splits:
    if ds_name == "PascalVOC": 
        ann_json = "annotations_without_background.json"
        if split in ["train", "val"]:
            split_path = os.path.join(root, data_folder, "train")
            ann_path = os.path.join(root, data_folder, f"custom_{split}.json")
        else:
            # use val anns
            split_path = os.path.join(root, data_folder, "valid")
            ann_path = os.path.join(root, data_folder, "valid", ann_json)

        master_dict[split]["img_pth"] = split_path

        with open(ann_path) as json_file:
            ann_dict = dict(json.load(json_file))
    else:
        if single_folder:
            ann_json = f"{split}_annotations_without_background.json"
            split_path = os.path.join(root, data_folder)
        else:
            ann_json = "annotations_without_background.json"
            if split == "val":
                split_path = os.path.join(root, data_folder, "valid")
            else:
                split_path = os.path.join(root, data_folder, split)

        ann_path = os.path.join(split_path, ann_json)
        master_dict[split]["img_pth"] = split_path

        with open(ann_path) as json_file:
            ann_dict = dict(json.load(json_file))

    master_dict["json_paths"][split] = ann_path

    """
    Get category id to name dict
    """
    cat_id_to_name = {}
    for cat in ann_dict["categories"]:
        cat_id_to_name[cat["id"]] = query_aug(str(cat["name"])) if augment_queries else str(cat["name"])

        if 0 in cat_id_to_name:
            raise Exception("Is 0 not background in this dataset??")

    if split == "train":
        pprint.pprint(cat_id_to_name)

    """
    Make an entry for each image along with all label names
    """
    for img_i, img in enumerate(ann_dict["images"]):
        master_dict[split][img['id']] = {}
        if ds_name == "ChessPieces":
            master_dict[split][img['id']]["label_names"] = [cat_id_to_name[i+1].lower() for i in range(1, len(cat_id_to_name))] # no background class
        else:
            master_dict[split][img['id']]["label_names"] = [cat_id_to_name[i+1].lower() for i in range(len(cat_id_to_name))] # no background class
        master_dict[split][img['id']]["width"] = img["width"]
        master_dict[split][img['id']]["height"] = img["height"]
        master_dict[split][img['id']]["file_name"] = img["file_name"]

    """
    Get image file to boxes, labels, and names dict
    """
    def convert_bbox(bbox:list, img_width, img_height) -> torch.Tensor:
        """
        xmin, ymin, width, height -> cx, cy, w, h
        Also scale the bbox coords to range [0,1]
        """
        xmin, ymin, width, height = bbox

        center_x = (xmin + (width / 2)) / img_width
        center_y = (ymin + (height / 2)) / img_height
        width = width / img_width
        height = height / img_height

        return torch.Tensor([center_x, center_y, width, height]).unsqueeze(0)

    for ann in ann_dict["annotations"]:
        image_id = ann["image_id"]

        # handle new bbox
        img_width = master_dict[split][image_id]["width"]
        img_height = master_dict[split][image_id]["height"]
        bbox = convert_bbox(ann["bbox"], img_width, img_height)
        if not "boxes" in master_dict[split][image_id]:
            master_dict[split][image_id]["boxes"] = bbox
        else:
            master_dict[split][image_id]["boxes"] = torch.cat((master_dict[split][image_id]["boxes"], bbox), dim=0)

        # handle new label
        if ds_name == "ChessPieces":
            if ann["category_id"] == 1:
                new_label = torch.Tensor([0])
            else:
                new_label = torch.Tensor([ann["category_id"] - 2])
        else:
            new_label = torch.Tensor([ann["category_id"] - 1])

        if not "labels" in master_dict[split][image_id]:
            master_dict[split][image_id]["labels"] = new_label
        else:
            master_dict[split][image_id]["labels"] = torch.cat((master_dict[split][image_id]["labels"], new_label)) # !! assuming no background !!


print("---")
print("train", len(master_dict["train"]))
print("val", len(master_dict["val"]))
print("test", len(master_dict["test"]))

"""
Validating on a single image
"""
def draw_bounding_boxes(image_path, bounding_boxes):
    image = Image.open(image_path)
    width, height = image.size
    draw = ImageDraw.Draw(image)

    for box in bounding_boxes:
        cx, cy, w, h = box
        x1 = (cx - (w / 2)) * width
        y1 = (cy - (h / 2)) * height
        x2 = (cx + (w / 2)) * width
        y2 = (cy + (h / 2)) * height
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)

    image.save("../hey.png")

all_train_keys = [int(key) for key in master_dict["train"] if key != "img_pth"]
img_info = master_dict["train"][6]
#bboxes = img_info["boxes"][0].unsqueeze(0)
bboxes = img_info["boxes"]
    
print([key for key in img_info])
draw_bounding_boxes(master_dict["train"]["img_pth"] + "/" + img_info["file_name"], bboxes)
print(img_info["labels"], img_info["label_names"])

"""
Save annotations to a dict
"""
with open(f"ds_info/{ds_name}.pkl", "wb") as f:
    pickle.dump(master_dict, f)


