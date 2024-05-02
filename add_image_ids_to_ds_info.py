import os, pickle, json

dataset_to_COCO_GT = {
    'DroneControl': lambda split: f"data/odinw/DroneControl/Drone Control.v3-raw.coco/{split.replace('val', 'valid')}/annotations_without_background.json", 
    'boggleBoards': lambda split: f"data/odinw/boggleBoards/416x416AutoOrient/export/{split}_annotations_without_background.json",
    'ThermalCheetah': lambda split: f"data/odinw/ThermalCheetah/{split.replace('val', 'valid')}/annotations_without_background.json",
    'PKLot': lambda split: f"data/odinw/PKLot/640/{split.replace('val', 'valid')}/annotations_without_background.json", 
    'BCCD': lambda split: f"data/odinw/BCCD/BCCD.v3-raw.coco/{split.replace('val', 'valid')}/annotations_without_background.json",
    'AerialMaritimeDrone': lambda split: f"data/odinw/AerialMaritimeDrone/large/{split.replace('val', 'valid')}/annotations_without_background.json",
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
    'PascalVOC': lambda split: f"data/odinw/PascalVOC/{split.replace('val', 'valid')}/annotations_without_background.json", 
    'Packages': lambda split: f"data/odinw/Packages/Raw/{split.replace('val', 'valid')}/annotations_without_background.json", 
    'selfdrivingCar': lambda split: f"data/odinw/selfdrivingCar/fixedLarge/export/{split}_annotations_without_background.json",
    "O365": lambda split: f'data/O365/zhiyuan_objv2_{split.replace("val", "train").replace("test", "val")}.json'
}
for i in range(5):
    print("-=-=-=-=-=-=-=-=-=-=-=-")

for i, ds_info_pth in enumerate([os.path.join("data/ds_info", ds_info_dict) for ds_info_dict in os.listdir("data/ds_info")]):
    ds_name = ds_info_pth.split("/")[-1].split(".pkl")[0]
    print("------------", ds_name, ds_info_pth, i)

    if "O365" not in ds_info_pth:
        continue

    ds_info_dict = None
    with open(ds_info_pth, "rb") as f:
        ds_info_dict = pickle.load(f)

    for split in ["val", "test"]: 
        if ds_name in ["PascalVOC"]:
            coco_split = {"val": "train", "test": "val"}[split] 
        else:
            coco_split = split

        with open(dataset_to_COCO_GT[ds_info_pth.split("/")[-1].split(".pkl")[0]](coco_split), "rb") as f:
            ds_coco_dict = json.load(f)

        missing_count = 0
        for i, image_dict in enumerate(ds_coco_dict["images"]):
            try:
                if not "O365" in ds_name:
                    ds_info_dict[split][image_dict["file_name"]]["image_id"] = image_dict["id"]
                else:
                    ds_info_dict_item = ds_info_dict[split].pop(image_dict["file_name"].split("/")[-1])
                    ds_info_dict[split][image_dict["file_name"]] = ds_info_dict_item
                    ds_info_dict[split][image_dict["file_name"]]["image_id"] = image_dict["id"]
            except KeyError:
                # print(i, image_dict["file_name"])
                missing_count += 1

        print(ds_name, split, f"missing {missing_count}/{i+1}")
        if missing_count > 0:
            if missing_count != (len(ds_coco_dict["images"]) - (len(ds_info_dict[split]) - 1)):
                raise Exception

    with open(ds_info_pth, "wb") as f:
        pickle.dump(ds_info_dict, f)
