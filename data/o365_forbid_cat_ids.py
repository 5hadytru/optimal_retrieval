import json, os, pickle
from fuzzywuzzy import process
import tensorflow as tf
from typing import Optional, Sequence, Tuple, Union

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

# The list below contains labels from Object365 and Visual Genome that are close
# to LVIS "rare" labels. Annotations with these labels must be removed from the
# training data for accurate "zero-shot" evaluation. The list was created by
# finding all O365/VG labels that contain LVIS labels as a substring (after
# removing space, underscore and dash). This catches close but non-identical
# labels such as "apple sauce" vs. "applesauce", "leather" vs "brown leather",
# or singular vs. plural. False positives were manually removed from the list.
O365_AND_VG_FORBIDDEN = [
    'apple cider', 'apple sauce', 'apricots', 'ax tool', 'axe', 'baguette',
    'baguettes', 'balsamic vinegar', 'barbell weights', 'barbells', 'barges',
    'baseball mascot', 'bbq cooker', 'beach ball', 'bean casserole',
    'bear mascot', 'bed pan', 'beef stew', 'beige fleece', 'big rat',
    'bird mascot', 'black fleece', 'black funnel', 'black garbage',
    'black leather', 'black leather corner', 'black pistol', 'black satchel',
    'blackleather', 'blue bonnet', 'blue pennant', 'blue plume', 'blue snake',
    'bobber', 'book mark', 'bookmarker', 'bookmarks', 'bottle liquor',
    'breakfast quiche', 'broken spear', 'brown gorilla', 'brown leather',
    'building gargoyle', 'burritos', 'cabana roof', 'cabanas', 'camera flash',
    'carnations', 'carrot stew', 'casserole dish', 'cassette disc',
    'cassette tape', 'cement cylinder', 'chaps', 'charcoal cooker',
    'check book', 'checker board', 'checkerboard pattern', 'chess board',
    'chime is hanging', 'chime is still', 'chimes', 'chocolate eclair',
    'clementines', 'clock pendulum', 'clothes hamper', 'coffee stirrer',
    'coil burner', 'coil heater', 'coil pipe', 'coil samples', 'coil wire',
    'coiled cable', 'coiled wire', 'coils', 'cooker plate', 'cooker unit',
    'cookers', 'cork board', 'corn bread', 'coveralls', 'crab meat', 'croutons',
    'cylinder figure', 'cylinder object', 'cylinders', 'cymbals',
    'dark leather', 'detergent bottle', 'diary cover', 'dish detergent',
    'dishwashing detergent', 'dog kennel', 'doll house', 'dollar bill',
    'dollars', 'doves', 'dragon fly', 'drum stick', 'drumsticks', 'dumb bell',
    'dust pan', 'ear plug', 'ear plugs', 'earplugs', 'egg casserole',
    'electric shears', 'electrical coil', 'exhaust funnel', 'eye patch',
    'fedora hat', 'fence kennel', 'fish bowl', 'flag pennants',
    'flash from camera', 'flashes', 'fleece jacket', 'fleece liner',
    'footlocker', 'fudge center', 'futon cushion', 'game board', 'garbage heap',
    'garbage pail', 'garbage pails', 'garbage pile', 'gargoyles', 'gas mask',
    'gemstones', 'glass cylinder', 'glass of lemonade', 'gold chime',
    'gorillas', 'gourds', 'grape popsicle', 'green fleece', 'green gourds',
    'green shawl', 'grey fleece', 'handcuffs', 'head jewels', 'head set',
    'headsets', 'heatin coil', 'hole puncher', 'hot plate', 'hot plates',
    'hour glass', 'house boat', 'iridescent shears', 'jewels', 'joysticks',
    'kegs', 'key card', 'kitchen shears', 'koala bear', 'laundry detergent',
    'laundry hamper', 'leather patch', 'leather satchel', 'leather square',
    'leather strip', 'legumes', 'liquor bottle', 'liquor bottles',
    'liquor spirit', 'liquorbottle', 'lockers', 'mascots', 'match box',
    'meat stew', 'metal shears', 'microphone headset', 'nail file',
    'nutcracker doll', 'omelet part', 'omelete', 'omelette', 'omeletter',
    'one dollar', 'paint scraper', 'panty hose', 'papayas', 'paper weight',
    'peg board', 'pencil sharpener', 'pendulums', 'pennant banner', 'pennants',
    'persimmons', 'phone book', 'pin wheel', 'pin wheels', 'pinwheels',
    'pistol in waistband', 'pitch fork', 'pitcher of lemonade', 'play snake',
    'polo mallet', 'poncho hood', 'potato masher', 'propane cylinder', 'prunes',
    'radar beacon', 'radar dish', 'radar equipment', 'red coils', 'red leather',
    'red poncho', 'red spear', 'redthimble', 'rice cooker', 'sand barge',
    'sauce pan', 'sauce pans', 'saw horse', 'saw horses', 'sawhorse bench',
    'sawhorses', 'scissors shears', 'scrape', 'sea plane', 'sheep shears',
    'silver armor', 'silver funnel', 'sketched handcuffs', 'skull cap',
    'sliced gourds', 'slow cooker', 'small baguette', 'spears',
    'spinach quiche', 'step ladder', 'stick mallet', 'stirrers',
    'storage locker', 'stuffed gorilla', 'sub woofer', 'tambourines',
    'tan leather', 'tangy lemonade', 'telephone books', 'there is an axe',
    'toy snake', 'trainstep ladder', 'turnip roots', 'turnips', 'tux jacket',
    'tuxedo', 'tuxedo jacket', 'tuxedos', 'two rats', 'vats', 'video cassettes',
    'vodka bottle', 'vodka bottles', 'vultures', 'wash basin', 'wash basins',
    'waste barge', 'white armor', 'white cylinder', 'white fleece',
    'white pegboard', 'white shears', 'wii joystick', 'wind chime',
    'wind chimes', 'windchime', 'windchimes', 'wolf head', 'wood armoire',
    'wooden axe', 'woolen fleece', 'yellow bulldozer'
]

# Adding NOT_PROMPTABLE_MARKER to a query will exclude it from having a prompt
# template (e.g. 'a photo of a {}') added during training:
NOT_PROMPTABLE_MARKER = '#'

def _canonicalize_string_tf(
    string: Union[str, Sequence[str], tf.Tensor]) -> tf.Tensor:
  """Brings text labels into a standard form."""

  string = tf.strings.lower(string)

  # Remove all characters that are not either alphanumeric, or dash, or space,
  # or NOT_PROMPTABLE_MARKER:
  string = tf.strings.regex_replace(
      string, f'[^a-z0-9-{NOT_PROMPTABLE_MARKER} ]', ' ')
  string = tf.strings.regex_replace(string, r'\s+', ' ')
  string = tf.strings.regex_replace(string, r'-+', '-')
  string = tf.strings.strip(string)

  # Remove characters that equal the promptability-maker but appear somewhere
  # other than the start of the string:
  string = tf.strings.regex_replace(
      string, f'([^^]){NOT_PROMPTABLE_MARKER}+', r'\1')

  return string


def remove_promptability_marker(x: tf.Tensor) -> tf.Tensor:
  """Removes any promptability-marker-character from a tensor of strings."""
  return tf.strings.regex_replace(x, NOT_PROMPTABLE_MARKER, '')


def _is_forbidden_label(labels: tf.Tensor) -> tf.Tensor:
  """Checks which elements of string tensor 'labels' are forbidden."""
  forbidden_labels = LVIS_RARE_CLASSES + O365_AND_VG_FORBIDDEN

  # Canonicalize both query and forbidden labels:
  forbidden_labels = _canonicalize_string_tf(forbidden_labels)
  labels = _canonicalize_string_tf(labels)

  # Remove dashes, which are not removed by _canonicalize_string and may differ
  # between query and forbidden labels:
  forbidden_labels = tf.strings.regex_replace(forbidden_labels, '-', '')
  labels = tf.strings.regex_replace(labels, '-', '')

  # Need unique set for tf.lookup.StaticHashTable:
  forbidden_labels, _ = tf.unique(forbidden_labels)

  forbidden_labels_table = tf.lookup.StaticHashTable(
      tf.lookup.KeyValueTensorInitializer(
          forbidden_labels, tf.ones_like(forbidden_labels, dtype=tf.bool)),
      default_value=False)
  return forbidden_labels_table.lookup(remove_promptability_marker(labels))


with open("tmp/objects365_val.json", "rb") as f:
    val_json = json.load(f)

o365_cats_lst = [cat["name"] for cat in val_json["categories"]]
o365_cats = tf.constant(o365_cats_lst, dtype=tf.string)
forbidden_cats = o365_cats[_is_forbidden_label(o365_cats)]

if tf.executing_eagerly():
    forbidden_cats = forbidden_cats.numpy().tolist()
else:
    with tf.compat.v1.Session() as sess:
        forbidden_cats = sess.run(forbidden_cats).tolist()

forbidden_cats = [s.decode('utf-8') for s in forbidden_cats]

print(forbidden_cats)

forbidden_cat_ids = [cat["id"] for cat in val_json["categories"] if cat["name"] in forbidden_cats]

cat_is_forbidden = {
   cat["id"]: True if cat["id"] in forbidden_cat_ids else False for cat in val_json["categories"]
}

print(len(forbidden_cats), len(forbidden_cat_ids))

with open("o365_forbidden_cat_ids.pkl", "wb") as f:
   pickle.dump(cat_is_forbidden, f)