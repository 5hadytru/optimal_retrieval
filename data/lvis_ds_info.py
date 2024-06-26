"""
    LVIS ds_info only needs best 7 imagenet prompts, a custom class name to coco cat ID map, and a cat_is_rare map (LVIS is the only dataset which is test-only)
"""
import os, sys, json, pickle
import pandas as pd
import numpy as np
import torch


BEST_7_IMAGENET_TEMPLATES = [
    lambda c: f'itap of a {c}.',
    lambda c: f'a bad photo of the {c}.',
    lambda c: f'a origami {c}.',
    lambda c: f'a photo of the large {c}.',
    lambda c: f'a {c} in a video game.',
    lambda c: f'art of the {c}.',
    lambda c: f'a photo of the small {c}.'
]

# From annotation JSON files at https://www.lvisdataset.org/dataset:
LVIS_RARE_CLASSES = [
    'applesauce', 'apricot', 'arctic_(type_of_shoe)', 'armoire', 'armor', 'ax',
    'baboon', 'bagpipe', 'baguet', 'bait', 'ballet_skirt', 'banjo', 'barbell',
    'barge', 'bass_horn', 'batter_(food)', 'beachball', 'bedpan', 'beeper',
    'beetle', 'Bible', 'birthday_card', 'pirate_flag', 'blimp', 'gameboard',
    'bob', 'bolo_tie', 'bonnet', 'bookmark', 'boom_microphone', 'bow_(weapon)',
    'pipe_bowl', 'bowling_ball', 'boxing_glove', 'brass_plaque', 'breechcloth',
    'broach', 'bubble_gum', 'horse_buggy', 'bulldozer', 'bulletproof_vest',
    'burrito', 'cabana', 'locker', 'candy_bar', 'canteen', 'elevator_car',
    'car_battery', 'cargo_ship', 'carnation', 'casserole', 'cassette',
    'chain_mail', 'chaise_longue', 'chalice', 'chap', 'checkbook',
    'checkerboard', 'chessboard', 'chime', 'chinaware', 'poker_chip',
    'chocolate_milk', 'chocolate_mousse', 'cider', 'cigar_box', 'clarinet',
    'cleat_(for_securing_rope)', 'clementine', 'clippers_(for_plants)', 'cloak',
    'clutch_bag', 'cockroach', 'cocoa_(beverage)', 'coil', 'coloring_material',
    'combination_lock', 'comic_book', 'compass', 'convertible_(automobile)',
    'sofa_bed', 'cooker', 'cooking_utensil', 'corkboard', 'cornbread',
    'cornmeal', 'cougar', 'coverall', 'crabmeat', 'crape', 'cream_pitcher',
    'crouton', 'crowbar', 'hair_curler', 'curling_iron', 'cylinder', 'cymbal',
    'dagger', 'dalmatian', 'date_(fruit)', 'detergent', 'diary', 'die',
    'dinghy', 'tux', 'dishwasher_detergent', 'diving_board', 'dollar',
    'dollhouse', 'dove', 'dragonfly', 'drone', 'dropper', 'drumstick',
    'dumbbell', 'dustpan', 'earplug', 'eclair', 'eel', 'egg_roll',
    'electric_chair', 'escargot', 'eyepatch', 'falcon', 'fedora', 'ferret',
    'fig_(fruit)', 'file_(tool)', 'first-aid_kit', 'fishbowl', 'flash',
    'fleece', 'football_helmet', 'fudge', 'funnel', 'futon', 'gag', 'garbage',
    'gargoyle', 'gasmask', 'gemstone', 'generator', 'goldfish',
    'gondola_(boat)', 'gorilla', 'gourd', 'gravy_boat', 'griddle', 'grits',
    'halter_top', 'hamper', 'hand_glass', 'handcuff', 'handsaw',
    'hardback_book', 'harmonium', 'hatbox', 'headset', 'heron', 'hippopotamus',
    'hockey_stick', 'hookah', 'hornet', 'hot-air_balloon', 'hotplate',
    'hourglass', 'houseboat', 'hummus', 'popsicle', 'ice_pack', 'ice_skate',
    'inhaler', 'jelly_bean', 'jewel', 'joystick', 'keg', 'kennel', 'keycard',
    'kitchen_table', 'knitting_needle', 'knocker_(on_a_door)', 'koala',
    'lab_coat', 'lamb-chop', 'lasagna', 'lawn_mower', 'leather', 'legume',
    'lemonade', 'lightning_rod', 'limousine', 'liquor', 'machine_gun',
    'mallard', 'mallet', 'mammoth', 'manatee', 'martini', 'mascot', 'masher',
    'matchbox', 'microscope', 'milestone', 'milk_can', 'milkshake',
    'mint_candy', 'motor_vehicle', 'music_stool', 'nailfile', 'neckerchief',
    'nosebag_(for_animals)', 'nutcracker', 'octopus_(food)', 'octopus_(animal)',
    'omelet', 'inkpad', 'pan_(metal_container)', 'pantyhose', 'papaya',
    'paperback_book', 'paperweight', 'parchment', 'passenger_ship',
    'patty_(food)', 'wooden_leg', 'pegboard', 'pencil_box', 'pencil_sharpener',
    'pendulum', 'pennant', 'penny_(coin)', 'persimmon', 'phonebook',
    'piggy_bank', 'pin_(non_jewelry)', 'ping-pong_ball', 'pinwheel',
    'tobacco_pipe', 'pistol', 'pitchfork', 'playpen', 'plow_(farm_equipment)',
    'plume', 'pocket_watch', 'poncho', 'pool_table', 'prune', 'pudding',
    'puffer_(fish)', 'puffin', 'pug-dog', 'puncher', 'puppet', 'quesadilla',
    'quiche', 'race_car', 'radar', 'rag_doll', 'rat', 'rib_(food)',
    'river_boat', 'road_map', 'rodent', 'roller_skate', 'Rollerblade',
    'root_beer', 'safety_pin', 'salad_plate', 'salmon_(food)', 'satchel',
    'saucepan', 'sawhorse', 'saxophone', 'scarecrow', 'scraper', 'seaplane',
    'sharpener', 'Sharpie', 'shaver_(electric)', 'shawl', 'shears',
    'shepherd_dog', 'sherbert', 'shot_glass', 'shower_cap',
    'shredder_(for_paper)', 'skullcap', 'sling_(bandage)', 'smoothie', 'snake',
    'softball', 'sombrero', 'soup_bowl', 'soya_milk', 'space_shuttle',
    'sparkler_(fireworks)', 'spear', 'crawfish', 'squid_(food)', 'stagecoach',
    'steak_knife', 'stepladder', 'stew', 'stirrer', 'string_cheese', 'stylus',
    'subwoofer', 'sugar_bowl', 'sugarcane_(plant)', 'syringe', 'Tabasco_sauce',
    'table-tennis_table', 'tachometer', 'taco', 'tambourine', 'army_tank',
    'telephoto_lens', 'tequila', 'thimble', 'trampoline', 'trench_coat',
    'triangle_(musical_instrument)', 'truffle_(chocolate)', 'vat', 'turnip',
    'unicycle', 'vinegar', 'violin', 'vodka', 'vulture', 'waffle_iron',
    'walrus', 'wardrobe', 'washbasin', 'water_heater', 'water_gun', 'wolf'
]


print(len(LVIS_RARE_CLASSES))

with open("LVIS/lvis_v1_val.json", "rb") as f:
    j = json.load(f)

ds_info = {
    "best_7_imagenet_prompts": {},
    "custom_cls_name_to_coco_cat_id": {},
    "json_paths": {"test": "data/LVIS/lvis_v1_val.json", "val": "data/LVIS/custom_val.json"}
}

for cat in j["categories"]:
    custom_cls_name = cat["name"].replace("_", " ").replace("-", " ")
    ds_info['custom_cls_name_to_coco_cat_id'][custom_cls_name] = cat["id"]
    ds_info["best_7_imagenet_prompts"][custom_cls_name] =  [template(custom_cls_name) for template in BEST_7_IMAGENET_TEMPLATES]

with open("ds_info/LVIS.pkl", "wb") as f:
    pickle.dump(ds_info, f)