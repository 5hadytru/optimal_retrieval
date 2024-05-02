import json, pickle, os


dataset_to_COCO_GT = {
    'DroneControl': lambda split: f"data/odinw/DroneControl/Drone Control.v3-raw.coco/{split.replace('val', 'valid')}/annotations_without_background.json", 
    'boggleBoards': lambda split: f"data/odinw/boggleBoards/416x416AutoOrient/export/{split}_annotations_without_background.json",
    'ThermalCheetah': lambda split: f"data/odinw/ThermalCheetah/{split.replace('val', 'valid')}/annotations_without_background.json",
    'PKLot': lambda split: f"data/odinw/PKLot/640/{split.replace('val', 'valid')}/annotations_without_background.json", 
    'BCCD': lambda split: f"data/odinw/BCCD/BCCD.v3-raw.coco/{split.replace('val', 'valid')}/annotations_without_background.json",
    'AerialMaritimeDrone': lambda split: f"data/odinw/AerialMaritimeDrone/tiled/{split.replace('val', 'valid')}/annotations_without_background.json",
    'OxfordPets': lambda split: f"data/odinw/OxfordPets/by-breed/{split.replace('val', 'valid')}/annotations_without_background.json",
    'CottontailRabbits': lambda split: f"data/odinw/CottontailRabbits/{split.replace('val', 'valid')}/annotations_without_background.json",
    'dice': lambda split: f"data/odinw/dice/mediumColor/export/{split}_annotations_without_background.json", 
    'Raccoon': lambda split: f"data/odinw/Raccoon/Raccoon.v2-raw.coco/{split.replace('val', 'valid')}/annotations_without_background.json", 
    'NorthAmericaMushrooms': lambda split: f"data/odinw/NorthAmericaMushrooms/North American Mushrooms.v1-416x416.coco/{split.replace('val', 'valid')}/annotations_without_background.json", 
    'brackishUnderwater': lambda split: f"data/odinw/brackishUnderwater/1920x1080/{split.replace('val', 'valid')}/annotations_without_background.json", 
    'pothole': lambda split: f"data/odinw/pothole/{split.replace('val', 'valid')}/annotations_without_background.json", 
    'WildfireSmoke': lambda split: f"data/odinw/WildfireSmoke/{split.replace('val', 'valid')}/annotations_without_background.json", 
    'ChessPieces': lambda split: f"data/odinw/ChessPieces/Chess Pieces.v23-raw.coco/{split.replace('val', 'valid')}/annotations_without_background.json", 
    'thermalDogsAndPeople': lambda split: f"data/odinw/thermalDogsAndPeople/{split.replace('val', 'valid')}/annotations_without_background.json", 
    'ShellfishOpenImages': lambda split: f"data/odinw/ShellfishOpenImages/raw/{split.replace('val', 'valid')}/annotations_without_background.json", 
    'Aquarium': lambda split: f"data/odinw/Aquarium/Aquarium Combined.v2-raw-1024.coco/{split.replace('val', 'valid')}/annotations_without_background.json", 
    'pistols': lambda split: f"data/odinw/pistols/export/{split}_annotations_without_background.json", 
    'VehiclesOpenImages': lambda split: f"data/odinw/VehiclesOpenImages/416x416/{split.replace('val', 'valid')}/annotations_without_background.json", 
    'EgoHands': lambda split: f"data/odinw/EgoHands/specific/{split.replace('val', 'valid')}/annotations_without_background.json", 
    'openPoetryVision': lambda split: f"data/odinw/openPoetryVision/512x512/{split.replace('val', 'valid')}/annotations_without_background.json", 
    'AmericanSignLanguageLetters': lambda split: f"data/odinw/AmericanSignLanguageLetters/American Sign Language Letters.v1-v1.coco/{split.replace('val', 'valid')}/annotations_without_background.json", 
    'plantdoc': lambda split: f"data/odinw/plantdoc/416x416/{split.replace('val', 'valid')}/annotations_without_background.json", 
    'MaskWearing': lambda split: f"data/odinw/MaskWearing/raw/{split.replace('val', 'valid')}/annotations_without_background.json", 
    'PascalVOC': lambda split: f"data/odinw/PascalVOC/{split.replace('val', 'custom_val.json').replace('train', 'custom_train.json').replace('test', 'valid/annotations_without_background.json')}", 
    'Packages': lambda split: f"data/odinw/Packages/Raw/{split.replace('val', 'valid')}/annotations_without_background.json", 
    'selfdrivingCar': lambda split: f"data/odinw/selfdrivingCar/fixedLarge/export/{split}_annotations_without_background.json",
    'MountainDewCommercial': lambda split: f"data/odinw/MountainDewCommercial/{split.replace('val', 'valid')}/annotations_without_background.json",
    'websiteScreenshots': lambda split: f"data/odinw/websiteScreenshots/{split.replace('val', 'valid')}/annotations_without_background.json",
    'HardHatWorkers': lambda split: f"data/odinw/HardHatWorkers/raw/{split.replace('val', 'valid')}/annotations_without_background.json",
    "O365": lambda split: f'data/O365/{split.replace("train", "custom_train").replace("val", "custom_val").replace("test", "objects365_val")}.json',
    "LVIS": lambda split: f'data/LVIS/{split.replace("val", "custom_val.json").replace("test", "lvis_v1_val.json")}',
}


def add_wh(ds_info:dict, json_dict, split:str):
    if split not in ds_info:
        ds_info[split] = {}
    
    for img_dict in json_dict["images"]:
        if img_dict["id"] in ds_info[split]:
            ds_info[split][img_dict["id"]].update({"width": img_dict["width"], "height": img_dict["height"]})
        else:
            ds_info[split][img_dict["id"]] = {"width": img_dict["width"], "height": img_dict["height"]}


for ds_info_path in [os.path.join("ds_info", ds_info_name) for ds_info_name in os.listdir("ds_info")]:
    ds_name = ds_info_path.split("/")[-1][:-4]
    print("----", ds_name)

    if ds_name not in ["O365", "LVIS"] or ds_name == "VG":
        continue
    
    with open(ds_info_path, "rb") as f:
        d = pickle.load(f)
    
    if ds_name == "LVIS":
        d["classes_are_exclusive"] = False
    else:
        d["classes_are_exclusive"] = True

    if "json_paths" not in d:
        d["json_paths"] = {"val": dataset_to_COCO_GT[ds_name]("val"), "test": dataset_to_COCO_GT[ds_name]("test")}

    has_wh_already = False
    if "val" in d:
        test_key = list(key for key in d["val"].keys() if not isinstance(d["val"][key], str))[0]
        if "width" in d["val"][test_key]:
            has_wh_already = True
    
    if has_wh_already:
        print("Saving ds_info and skipping", ds_name)
        with open(ds_info_path, "wb") as f:
            pickle.dump(d, f)
        continue
    else:
        print("Adding wh to", ds_name)

    with open(dataset_to_COCO_GT[ds_name]("val").replace("data/", ""), "r") as f:
        val_json = json.load(f)

    add_wh(d, val_json, "val")

    with open(dataset_to_COCO_GT[ds_name]("test").replace("data/", ""), "r") as f:
        test_json = json.load(f)

    add_wh(d, test_json, "test")
    
    with open(ds_info_path, "wb") as f:
        pickle.dump(d, f)