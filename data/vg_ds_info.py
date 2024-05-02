"""
Create a ds_info dict for VG (unlike ODinW, includes adding prompts)

{
    train: {
        img_pth: str,
        [image_id]: {
            labels: Tensor (multi-hot)
            label_names: List[str]
            file_name: str
            imagenet_prompts: List[str]
        }
    }
}
"""
import json, pickle, os, time, random
import pandas as pd
import numpy as np
import torch


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


def create_and_save():
    with open("VG/image_data.json", "rb") as f:
        image_data = json.load(f)

    with open("VG/objects.json", "rb") as f:
        objects = json.load(f)

    master_dict = {
        "train": {"img_pth": "VG/VG_100K/"}
    }

    train_ids = pd.read_csv("VG/visual_genome_train_example_img_ids.csv", header=None, delimiter=" ")[0].tolist() # pre-de-duplicated

    master_dict["train"] = master_dict["train"] | {img_id: None for img_id in train_ids}

    # initialize each image's data dictionary
    for img_dict in image_data:
        try:
            img_id = img_dict["image_id"]
            master_dict["train"][img_id]
            master_dict["train"][img_id] = {
                "file_name": f"{img_id}.jpg",
                "width": img_dict["width"],
                "height": img_dict["height"],
            }
        except KeyError:
            continue

    assert all(val != None for val in master_dict["train"].values())

    # add all annotations
    with open("vg_forbidden_cat_ids.pkl", "rb") as f:
        cat_is_forbidden = pickle.load(f)

    for d_idx, d in enumerate(objects):
        # d contains all annotations for an image
        img_id = d["image_id"]

        try:
            master_dict["train"][img_id]
        except KeyError:
            continue

        bboxes = None
        labels = []

        # get all unique label names for this image
        label_names = []
        for o in d["objects"]:
            label_names += o["names"]
        img_label_names = list(set(label_names))

        label_map = {img_label_names[i]: i for i in range(len(img_label_names))}

        for o in d["objects"]:
            if any(cat_is_forbidden[cat_name] for cat_name in o["names"]):
                continue

            bbox = convert_bbox(
                [float(o['x']), float(o['y']), float(o['w']), float(o['h'])], 
                master_dict["train"][img_id]['width'],
                master_dict["train"][img_id]['height']
            )

            if bboxes is None:
                bboxes = bbox
            else:
                bboxes = torch.cat((bboxes, bbox), dim=0)

            labels.append(
                [label_map[name] for name in o["names"]])

        # add a random prompt for each label name, in order
        rng = np.random.default_rng(seed=int(time.time()))
        master_dict["train"][img_id]["labels"] = labels
        master_dict["train"][img_id]["label_names"] = img_label_names
        master_dict["train"][img_id]["imagenet_prompts"] = [random.choice(IMAGENET_TEMPLATES)(l) for l in img_label_names]
        master_dict["train"][img_id]["boxes"] = bboxes

        print(d_idx)

    with open("ds_info/VG.pkl", "wb") as f:
        pickle.dump(master_dict, f)
    
if __name__ == "__main__":
    create_and_save()