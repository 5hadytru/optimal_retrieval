"""
Get the O365 info dict. Differences from the ODinW procedure include not doing this for val/test, ignoring forbidden annotations, adding custom cls -> cat ID map, and adding positive prompts (latter two were done post-facto for ODinW)
"""
import pickle, json, os, time
import pandas as pd
import torch
import numpy as np
from PIL import Image, ImageDraw
from pprint import pprint


IMAGENET_TEMPLATES = [
    lambda c: f'a bad photo of a {c}.',
    lambda c: f'a photo of many {c}.',
    lambda c: f'a sculpture of a {c}.',
    lambda c: f'a photo of the hard to see {c}.',
    lambda c: f'a low resolution photo of the {c}.',
    lambda c: f'a rendering of a {c}.',
    lambda c: f'graffiti of a {c}.',
    lambda c: f'a bad photo of the {c}.',
    lambda c: f'a cropped photo of the {c}.',
    lambda c: f'a tattoo of a {c}.',
    lambda c: f'the embroidered {c}.',
    lambda c: f'a photo of a hard to see {c}.',
    lambda c: f'a bright photo of a {c}.',
    lambda c: f'a photo of a clean {c}.',
    lambda c: f'a photo of a dirty {c}.',
    lambda c: f'a dark photo of the {c}.',
    lambda c: f'a drawing of a {c}.',
    lambda c: f'a photo of my {c}.',
    lambda c: f'the plastic {c}.',
    lambda c: f'a photo of the cool {c}.',
    lambda c: f'a close-up photo of a {c}.',
    lambda c: f'a black and white photo of the {c}.',
    lambda c: f'a painting of the {c}.',
    lambda c: f'a painting of a {c}.',
    lambda c: f'a pixelated photo of the {c}.',
    lambda c: f'a sculpture of the {c}.',
    lambda c: f'a bright photo of the {c}.',
    lambda c: f'a cropped photo of a {c}.',
    lambda c: f'a plastic {c}.',
    lambda c: f'a photo of the dirty {c}.',
    lambda c: f'a jpeg corrupted photo of a {c}.',
    lambda c: f'a blurry photo of the {c}.',
    lambda c: f'a photo of the {c}.',
    lambda c: f'a good photo of the {c}.',
    lambda c: f'a rendering of the {c}.',
    lambda c: f'a {c} in a video game.',
    lambda c: f'a photo of one {c}.',
    lambda c: f'a doodle of a {c}.',
    lambda c: f'a close-up photo of the {c}.',
    lambda c: f'a photo of a {c}.',
    lambda c: f'the origami {c}.',
    lambda c: f'the {c} in a video game.',
    lambda c: f'a sketch of a {c}.',
    lambda c: f'a doodle of the {c}.',
    lambda c: f'a origami {c}.',
    lambda c: f'a low resolution photo of a {c}.',
    lambda c: f'the toy {c}.',
    lambda c: f'a rendition of the {c}.',
    lambda c: f'a photo of the clean {c}.',
    lambda c: f'a photo of a large {c}.',
    lambda c: f'a rendition of a {c}.',
    lambda c: f'a photo of a nice {c}.',
    lambda c: f'a photo of a weird {c}.',
    lambda c: f'a blurry photo of a {c}.',
    lambda c: f'a cartoon {c}.',
    lambda c: f'art of a {c}.',
    lambda c: f'a sketch of the {c}.',
    lambda c: f'a embroidered {c}.',
    lambda c: f'a pixelated photo of a {c}.',
    lambda c: f'itap of the {c}.',
    lambda c: f'a jpeg corrupted photo of the {c}.',
    lambda c: f'a good photo of a {c}.',
    lambda c: f'a plushie {c}.',
    lambda c: f'a photo of the nice {c}.',
    lambda c: f'a photo of the small {c}.',
    lambda c: f'a photo of the weird {c}.',
    lambda c: f'the cartoon {c}.',
    lambda c: f'art of the {c}.',
    lambda c: f'a drawing of the {c}.',
    lambda c: f'a photo of the large {c}.',
    lambda c: f'a black and white photo of a {c}.',
    lambda c: f'the plushie {c}.',
    lambda c: f'a dark photo of a {c}.',
    lambda c: f'itap of a {c}.',
    lambda c: f'graffiti of the {c}.',
    lambda c: f'a toy {c}.',
    lambda c: f'itap of my {c}.',
    lambda c: f'a photo of a cool {c}.',
    lambda c: f'a photo of a small {c}.',
    lambda c: f'a tattoo of the {c}.',
]


BEST_7_IMAGENET_TEMPLATES = [
    lambda c: f'itap of a {c}.',
    lambda c: f'a bad photo of the {c}.',
    lambda c: f'a origami {c}.',
    lambda c: f'a photo of the large {c}.',
    lambda c: f'a {c} in a video game.',
    lambda c: f'art of the {c}.',
    lambda c: f'a photo of the small {c}.'
]


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


def main():
    with open("O365/custom_train.json", "r") as f: # pre-filtered and split JSON
        train_json = json.load(f)
    print("Loaded JSON")

    master_dict = {
        "train": {"img_pth": "O365/train/"},
        "custom_cls_name_to_coco_cat_id": {},
        "imagenet_prompts": {},
        "best_7_imagenet_prompts": {},
        'json_paths': {
            "train": "data/O365/custom_train.json",
            "val": "data/O365/custom_val.json",
            "test": "data/O365/objects365_val.json"
        }
    }

    # create map from category ids to category names + create a map from custom category names to category ids + add map from cat name to all prompts
    cat_id_to_name = {}
    for cat in train_json["categories"]:
        cat_id_to_name[cat["id"]] = cat["name"]
        master_dict["custom_cls_name_to_coco_cat_id"][cat["name"]] = cat["id"]
        master_dict["imagenet_prompts"][cat["name"]] = [template(cat["name"]) for template in IMAGENET_TEMPLATES]
        master_dict["best_7_imagenet_prompts"][cat["name"]] = [template(cat["name"]) for template in BEST_7_IMAGENET_TEMPLATES]

        if 0 in cat_id_to_name:
            raise Exception("Is 0 not background in this dataset??")

    # initialize all images
    master_dict["train"] = master_dict["train"] | {
        img["id"]: {
            "file_name": img["file_name"],
            "width": img["width"],
            "height": img["height"]
        } for img in train_json["images"]
    }

    """
    Add all annotations
    """
    with open("o365_forbidden_cat_ids.pkl", "rb") as f:
        cat_is_forbidden = pickle.load(f)

    rng = np.random.default_rng(seed=int(time.time()))

    for ann_i,ann in enumerate(train_json["annotations"]):
        if ann_i % 25000 == 0:
            print(ann_i)

        if cat_is_forbidden[ann["category_id"]]:
            continue

        image_id = ann["image_id"]
        label_name = cat_id_to_name[ann["category_id"]]

        # ensure label_name and its random prompt is represented in positives
        if "label_names" not in master_dict["train"][image_id]:
            master_dict["train"][image_id]["label_names"] = [label_name]

            master_dict["train"][image_id]["imagenet_prompts"] = [
                IMAGENET_TEMPLATES[rng.integers(len(IMAGENET_TEMPLATES), size=1)[0]](label_name)
            ]
        elif label_name not in master_dict["train"][image_id]["label_names"]:
            master_dict["train"][image_id]["label_names"].append(label_name)
            master_dict["train"][image_id]["imagenet_prompts"].append(IMAGENET_TEMPLATES[rng.integers(len(IMAGENET_TEMPLATES), size=1)[0]](label_name))

        # handle new bbox
        img_width = master_dict["train"][image_id]["width"]
        img_height = master_dict["train"][image_id]["height"]
        bbox = convert_bbox(ann["bbox"], img_width, img_height)
        if not "boxes" in master_dict["train"][image_id]:
            master_dict["train"][image_id]["boxes"] = bbox
        else:
            master_dict["train"][image_id]["boxes"] = torch.cat((master_dict["train"][image_id]["boxes"], bbox), dim=0)

        # handle new label
        new_label = torch.tensor([master_dict["train"][image_id]["label_names"].index(label_name)])
        if not "labels" in master_dict["train"][image_id]:
            master_dict["train"][image_id]["labels"] = new_label
        else:
            master_dict["train"][image_id]["labels"] = torch.cat((master_dict["train"][image_id]["labels"], new_label))

    with open("ds_info/O365.pkl", "wb") as f:
        pickle.dump(master_dict, f)

    print("---")
    print("ann per img:", len(train_json["annotations"]) / len(master_dict["train"]),  len(train_json["annotations"]) / len(train_json["images"]))
    print("train", len(master_dict["train"]))

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

    img_info = master_dict["train"][list(master_dict["train"].keys())[1000]]
    bboxes = img_info["boxes"]
        
    print([key for key in img_info])
    draw_bounding_boxes(os.path.join(master_dict["train"]["img_pth"], img_info["file_name"]), bboxes)
    print(img_info["labels"], img_info["label_names"])

    """
    Save annotations to a dict
    """
    with open(f"ds_info/O365.pkl", "wb") as f:
        pickle.dump(master_dict, f)

if __name__ == "__main__":
    main()