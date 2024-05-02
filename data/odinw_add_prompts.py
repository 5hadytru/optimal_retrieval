import pickle, os, random


global_templates = {
    "boggleBoards": [
        lambda cls_name: f"A photo of a '{cls_name.split(' ')[-1]}' piece from the game Boggle",
        lambda cls_name: f"A photo of a '{cls_name.split(' ')[-1]}' piece from a Boggle board",
        lambda cls_name: f"A '{cls_name.split(' ')[-1]}' piece from a Boggle board",
        lambda cls_name: f"A '{cls_name.split(' ')[-1]}' piece from the game Boggle"
    ],
    "MountainDewCommercial": [
        lambda cls_name: f"A photo of a {cls_name}",
        lambda cls_name: f"A photo of a {cls_name}, a type of soda"
    ],
    "ThermalCheetah": [
        lambda cls_name: f"A thermal photo of a {cls_name}",
        lambda cls_name: f"A photo of a {cls_name} taken with a thermal camera"
    ],
    "PKLot": [
        lambda cls_name: f"A photo of an {cls_name}",
        lambda cls_name: f"An {cls_name}"
    ],
    "BCCD": [
        lambda cls_name: f"A photo of a {cls_name} under a microscope",
        lambda cls_name: f"{cls_name} under a microscope",
    ],
    "AerialMaritimeDrone": [
        lambda cls_name: f"An overhead photo of a {cls_name} by the water",
        lambda cls_name: f"An overhead photo of a {cls_name}",
        lambda cls_name: f"A photo of a {cls_name} taken by a drone"
    ],
    "OxfordPets": [
        lambda cls_name: f"A photo of a {cls_name}, a type of pet",
        lambda cls_name: f"A photo of my pet {cls_name}"
    ],
    "CottontailRabbits": [
        lambda _: f"A photo of a rabbit"
    ],
    "dice": [
        lambda cls_name: f'A photo of {cls_name}',
        lambda cls_name: f'A {cls_name[2:]}'
    ],
    "Raccoon": [
        lambda cls_name: f"A photo of a {cls_name} with a striped tail",
        lambda cls_name: f"A photo of a {cls_name}",
        lambda cls_name: f"A {cls_name}"
    ],
    "NorthAmericaMushrooms": [
        lambda cls_name: f"A photo of a {cls_name}"
    ],
    "brackishUnderwater": [
        lambda cls_name: f"An underwater photo of a {cls_name}"
    ],
    "pothole": [
        lambda cls_name: f"A {cls_name} in the road"
    ],
    "WildfireSmoke": [
        lambda cls_name: f"Wildfire {cls_name}",
        lambda cls_name: f"Wildfire {cls_name} in the distance",
        lambda cls_name: f"A photo of {cls_name}"
    ],
    "ChessPieces": [
        lambda cls_name: f"A {cls_name[2:]}",
        lambda cls_name: f"A photo of {cls_name[2:]}"
    ],
    "thermalDogsAndPeople": [
        lambda cls_name: f"A thermal infrared photo of a {cls_name}"
    ],
    "ShellfishOpenImages": [
        lambda cls_name: f"A {cls_name}",
        lambda cls_name: f"A photo of a {cls_name}"
    ],
    "pistols": [
        lambda cls_name: f"A {cls_name}",
        lambda cls_name: f"A photo of a {cls_name}"
    ],
    "VehiclesOpenImages": [
        lambda cls_name: f"A {cls_name}",
        lambda cls_name: f"A photo of a {cls_name}"
    ],
    "EgoHands": [
        lambda cls_name: f"A photo of {cls_name}"
    ],
    "openPoetryVision": [
        lambda cls_name: f"A screenshot of text in {cls_name}",
        lambda cls_name: f"Text typed in {cls_name}"
    ],
    "AmericanSignLanguageLetters": [
        lambda cls_name: f"A photo of {cls_name}",
        lambda cls_name: cls_name
    ],
    "plantdoc": [
        lambda cls_name: f"A photo of a {cls_name}"
    ],
    "MaskWearing": [
        lambda cls_name: f"A person with {cls_name}"
    ],
    "Packages": [
        lambda _: "A photo of a package on a porch",
        lambda _: "A package on a porch",
        lambda _: "A package on the porch"
    ],
    "selfdrivingCar": [
        lambda cls_name: f"A {cls_name}",
        lambda cls_name: f"A photo of a {cls_name}"
    ],
    "websiteScreenshots": [
        lambda cls_name: f"A {cls_name} on a webpage",
        lambda cls_name: f"A screenshot of a {cls_name} on a webpage",
        lambda cls_name: f"A {cls_name} on a website",
        lambda cls_name: f"A screenshot of a {cls_name} on a website"
    ],
    "PascalVOC": [
        lambda cls_name: f"A photo of a {cls_name}",
        lambda cls_name: f"A {cls_name}"
    ]
}

def dronecontrol_template(cls_name):
    if "null" in cls_name:
        return [
            "An irrelevant object",
            "Irrelevant object"
        ]
    else:
        return [
            f"A person indicating {cls_name} to a drone",
            f"A person giving {cls_name} to a drone",
            f"A person showing {cls_name} to a drone",
            f"A low resolution photo of a person indicating {cls_name} to a drone",
            f"A low resolution photo of a person giving {cls_name} to a drone",
            f"A low resolution photo of a person showing {cls_name} to a drone",
            f"A photo of a person indicating {cls_name} to a drone",
            f"A photo of a person giving {cls_name} to a drone",
            f"A photo of a person showing {cls_name} to a drone"
        ]


def aquarium_template(cls_name):
    if "penguin" in cls_name:
        return [
            "A photo of a black and white penguin",
            "A black and white penguin in an aquarium",
            "A photo of a black and white penguin in an aquarium"
        ]
    elif "puffin" in cls_name:
        return [
            "A puffin, a type of bird, with an orange beak",
            "A photo of a puffin with an orange beak",
            "A puffin with an orange beak in an aquarium"
        ]
    elif "stingray" in cls_name:
        return [
            "A photo of a flat and round stingray in an aquarium",
            "A photo of a stingray, which is flat and round",
            "A stingray, which is flat and round"
        ]
    else:
        return [
            f"A photo of a {cls_name} in an aquarium",
            f"A photo of a {cls_name}",
            f"A {cls_name} in an aquarium"
        ]
    

def hardhat_template(cls_name):
    if "head" in cls_name:
        return [
            f"A photo of a {cls_name} with no helmet",
            f"A {cls_name} with no helmet"
        ]
    elif "helmet" in cls_name:
        return [
            f"A photo of a head with a helmet",
            f"A head with a helmet"
        ]
    else:
        return [
            f"A photo of a person",
            f"A person"
        ]

conditional_templates = {
    "Aquarium": aquarium_template,
    "DroneControl": dronecontrol_template,
    "HardHatWorkers": hardhat_template
}

imagenet_templates = [
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

best_7_imagenet_templates = [
    lambda c: f'itap of a {c}.',
    lambda c: f'a bad photo of the {c}.',
    lambda c: f'a origami {c}.',
    lambda c: f'a photo of the large {c}.',
    lambda c: f'a {c} in a video game.',
    lambda c: f'art of the {c}.',
    lambda c: f'a photo of the small {c}.'
]

def main():
# for each ds_info dictionary in the folder, first add the label name to all 7 prompts list 
# then go thru each training example and add full prompts via random template sampling (for both a 
# random ImageNet template and a random custom template for ablation)
    count = 0
    for ds_info_pth in [os.path.join("ds_info", info_dict_f) for info_dict_f in os.listdir("ds_info")]:
        ds_name = ds_info_pth.split("/")[-1].split(".pkl")[0]

        if "Chess" not in ds_name:
            continue

        with open(ds_info_pth, "rb") as f:
            ds_dict = pickle.load(f)

        # get all unique label names
        all_label_names = []
        for img_id, img_dict in ds_dict["train"].items():
            if img_id == "img_pth":
                continue
            all_label_names += img_dict["label_names"]

        unique_label_names = list(set(all_label_names))

        print(ds_name, f"{count + 1}/18", len(unique_label_names), "classes")

        # add all imagenet prompts to the root of the dict
        ds_dict["imagenet_prompts"] = {label_name: [template(label_name) for template in imagenet_templates] for label_name in unique_label_names}
        ds_dict["best_7_imagenet_prompts"] = {label_name: [template(label_name) for template in best_7_imagenet_templates] for label_name in unique_label_names}

        # add ALL custom prompts to the root of the dict
        ds_dict["custom_prompts"] = {label_name: [] for label_name in unique_label_names}
        for label_name in unique_label_names:
            if ds_name in global_templates:
                for template in global_templates[ds_name]:
                    ds_dict["custom_prompts"][label_name].append(template(label_name))
            else:
                for prompt in conditional_templates[ds_name](label_name):
                    ds_dict["custom_prompts"][label_name].append(prompt)

        # for each training image add random prompts for each label_name, in order!
        for img_name, img_dict in ds_dict["train"].items():
            if img_name == "img_pth":
                continue

            assert len(set(img_dict["label_names"])) == len(img_dict["label_names"])

            img_dict["imagenet_prompts"] = []
            img_dict["custom_prompts"] = []
            
            for label_name in img_dict["label_names"]:
                img_dict["imagenet_prompts"].append(random.choice(ds_dict["imagenet_prompts"][label_name]))
                img_dict["custom_prompts"].append(random.choice(ds_dict["custom_prompts"][label_name]))

            assert len(img_dict["imagenet_prompts"]) == len(img_dict["label_names"])
            
            assert len(img_dict["custom_prompts"]) == len(img_dict["label_names"])

        with open(ds_info_pth, "wb") as f:
            pickle.dump(ds_dict, f)

        count += 1

if __name__ == '__main__':
    main()