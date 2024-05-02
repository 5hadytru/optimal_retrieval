import os, sys, pickle, random
import numpy as np


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


def count_label_freqs(ds_info:dict, total_label_count:int, label_counts):
    for i,d in enumerate(ds_info.values()):
        if isinstance(d, str):
            continue

        if "label_names" not in d:
            continue

        for label_name in d["label_names"]:
            if label_name not in label_counts:
                label_counts[label_name] = 1
            else:
                label_counts[label_name] += 1
            total_label_count += 1

        if i % 20000 == 0:
            print(f"{i}/{len(ds_info)}")

    return total_label_count, label_counts


def gen_neg_label_probs():
    """
    Make a dict mapping valid (non-forbidded) label names within VG and O365 to their probabilities of being selected
    as negatives
    Forbidden labels are already not present in any annotations
    """
    with open("ds_info/O365_all.pkl", "rb") as f:
        o365_dsi = pickle.load(f)

    with open("ds_info/VG.pkl", "rb") as f:
        vg_dsi = pickle.load(f)

    print("Loaded ds_infos")
    
    total_label_count = 0
    label_counts = {}

    total_label_count, label_counts = count_label_freqs(o365_dsi["train"], total_label_count, label_counts)
    total_label_count, label_counts = count_label_freqs(vg_dsi["train"], total_label_count, label_counts)

    label_probs = {label_name: label_count / total_label_count for label_name, label_count in label_counts.items()}

    with open("o365_vg_neg_label_probs.pkl", "wb") as f:
        pickle.dump(label_probs, f)


def load_neg_label_probs():
    with open("o365_vg_neg_label_probs.pkl", "rb") as f:
        neg_label_probs_dict = pickle.load(f)
        neg_labels = list(neg_label_probs_dict.keys())
        neg_label_probs = np.array(list(neg_label_probs_dict.values()))
    
    return neg_labels, neg_label_probs


def remove_existing_negs(img_dict):
    if 'labels' in img_dict and len(img_dict["labels"]) > 0:
        last_positive_prompt_idx = int(max(img_dict["labels"])) if not isinstance(img_dict["labels"][0], list) else int(max([max(l) for l in img_dict["labels"] if len(l) > 0]))
        if last_positive_prompt_idx < len(img_dict["imagenet_prompts"]):
            img_dict["label_names"] = img_dict["label_names"][:last_positive_prompt_idx + 1]
            img_dict["imagenet_prompts"] = img_dict["imagenet_prompts"][:last_positive_prompt_idx + 1] 

            try:
                img_dict["custom_prompts"] = img_dict["custom_prompts"][:last_positive_prompt_idx + 1] 
            except KeyError:
                pass
    else:
        img_dict["label_names"] = []
        img_dict["imagenet_prompts"] = []
        img_dict["custom_prompts"] = []


def sample_random_negatives(positive_labels:dict, neg_labels:list, neg_label_probs:np.ndarray, n=50):
    """
    Given a list of labels represented in an image, get a list of n labels not present in the image, 
    sampled in proportion to their presence in the dataset
    """
    negative_label_names = []
    rng = np.random.default_rng(seed=0)
    negatives = rng.choice(len(neg_labels), size=n + 50, p=neg_label_probs, replace=False) # +50 gives us a cushion if a bunch of the choices are somehow in the image
    for neg_i in negatives:
        candidate_neg = neg_labels[neg_i]

        if len(negative_label_names) == n:
            break

        # prevent negatives which are substrings or matches of positives and vice versa
        if any([((pl in candidate_neg or candidate_neg in pl) and len(pl) > 2) for pl in positive_labels]):
            #print([(candidate_neg, pl) for pl in positive_labels if ((pl in candidate_neg or candidate_neg in pl) and len(pl) > 2)])
            continue

        # special case
        if (candidate_neg in ["person", "human"]) and ("human" in positive_labels or "person" in positive_labels):
            continue

        negative_label_names.append(candidate_neg)

    assert len(negative_label_names) == n, f"{len(negative_label_names)} {len(positive_labels)}"
    assert len(negative_label_names) == len(set(negative_label_names)), f"{len(negative_label_names)} {len(set(negative_label_names))}"

    return negative_label_names


def main():
    """
    For each ds_info that has a 'train' dict, go thru each image's dict and add 50 negative prompts

    Negative labels (label_names/categories) are selected according to their frequency in the dataset + are not added if the 
    label is already a positive for the image or if it is a substring of one of the positives (and vice versa)
    """
    #gen_neg_label_probs()

    neg_labels, neg_label_probs = load_neg_label_probs()

    ds_info_paths = sorted([os.path.join("ds_info", ds_info_name) for ds_info_name in os.listdir("ds_info")])
    for dsi_idx, ds_info_path in enumerate(ds_info_paths):
        ds_name = ds_info_path.split("/")[-1][:-4]

        if bool(int(sys.argv[1])) and dsi_idx < int(sys.argv[2]):
            print("-------- Skipping", ds_name, dsi_idx)
            continue
        print("--------", ds_name, dsi_idx)

        with open(ds_info_path, "rb") as f:
            d = pickle.load(f)
        
        if "train" not in d:
            print("No train:", ds_name)
            continue

        ds_has_custom_prompts = any("custom_prompts" in img_dict for img_dict in d["train"].values())

        for idx, img_id in enumerate(d["train"].keys()):
            if isinstance(img_id, str):
                continue

            if "label_names" in d['train'][img_id]:
                neg_label_names = sample_random_negatives(
                    {ln: None for ln in d['train'][img_id]["label_names"]}, # dict for O(1) lookup
                    neg_labels, 
                    neg_label_probs
                )
            else:
                d['train'][img_id]["label_names"] = []
                neg_label_names = sample_random_negatives(
                    {},
                    neg_labels, 
                    neg_label_probs
                )

            neg_prompts = []
            for ln in neg_label_names:
                neg_prompts.append(random.choice(IMAGENET_TEMPLATES)(ln))

            d['train'][img_id]["label_names"] += neg_label_names

            if "imagenet_prompts" in d['train'][img_id]:
                d['train'][img_id]["imagenet_prompts"] += neg_prompts
            else:
                d['train'][img_id]["imagenet_prompts"] = neg_prompts

            if ds_has_custom_prompts:
                if "custom_prompts" in d['train'][img_id]:
                    d['train'][img_id]["custom_prompts"] += neg_prompts
                else:
                    d['train'][img_id]["custom_prompts"] = neg_prompts

            if idx % 1000 == 0:
                print(f"- {idx}/{len(d['train'])}")

        with open(ds_info_path, "wb") as f:
            pickle.dump(d, f)


if __name__ == "__main__":
    main()