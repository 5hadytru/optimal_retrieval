import pickle, sys, os, copy
from itertools import product
import pprint

# L14: [8e-8, 8e-9]
# B16: [2e-7, 8e-9]

MODELS = ["OWL_B16", "OWL_L14"]

SWEEP_NAME = sys.argv[1]
SUPPOSED_LEN = int(sys.argv[2])

BASE_SWEEPS = [
    {
        "ss_lrs": { 
            "OWL_L14": [[2e-5, 2e-6], [8e-6, 8e-7], [4.4e-6, 4.4e-7], [1e-6, 1e-7]],
            "OWL_B16": [[2e-6, 8e-8], [1.1e-6, 4.4e-8], [2e-7, 8e-9], [1.1e-7, 4.4e-9]]
        },
        "lw_lr_decay_mults": [[0.99, 0.99], [0.98, 0.98], [0.97, 0.97], [0.96, 0.96], [0.94, 0.94]],
        "warmup_epochs": [2],
        "lr_sched": ["cos warm"],
        "epochs": [7],
        "train_batch": {
            "OWL_L14": [None],
            "OWL_B16": [None]
        },
        "subset_id": [None]
    }
]

DS_SWEEPS = {
    'ThermalCheetah': [{}],
    'BCCD': [{}],
    'AerialMaritimeDrone': [{}],
    'OxfordPets': [{}],
    'dice': [{}],
    'brackishUnderwater': [{
        "subset_id": [0]
    }],
    'PascalVOC': [{
        "subset_id": [0]
    }],
    'pothole': [{}],
    'WildfireSmoke': [{}],
    'ChessPieces': [{}],
    'thermalDogsAndPeople': [{}],
    'ShellfishOpenImages': [{}],
    'Aquarium': [{}],
    'pistols': [{
        "subset_id": [0]
    }],
    'EgoHands': [{
        "subset_id": [0]
    }],
    'openPoetryVision': [{
        "subset_id": [0]
    }],
    'AmericanSignLanguageLetters': [{
        "subset_id": [0]
    }],
    'plantdoc': [{}],
}

def gen_hp_sweep_dicts():
    sweep_dir = f"configs/ds_specific/sweeps/{SWEEP_NAME}"
    assert len(DS_SWEEPS) == SUPPOSED_LEN, len(DS_SWEEPS)
    os.makedirs(sweep_dir, exist_ok=True)
    for ds_name, all_mods_dicts in DS_SWEEPS.items():
        sweep = {model: [] for model in MODELS} # list of dicts for each model
        if all([m is None for m in all_mods_dicts]):
            print("-")
            continue
        for base_sweep_i, mods_dict in enumerate(all_mods_dicts):
            if mods_dict is None:
                continue
            for model in MODELS:
                sweep_lists = {}
                # get a {key: list of values} dict for this model 
                base_sweep = BASE_SWEEPS[base_sweep_i]
                for key, val in base_sweep.items():
                    if key in mods_dict:
                        if isinstance(val, dict):
                            sweep_lists[key] = mods_dict[key][model]
                        else:
                            sweep_lists[key] = mods_dict[key]
                    else:
                        if isinstance(val, dict):
                            sweep_lists[key] = val[model]
                        else:
                            sweep_lists[key] = val
                
                keys = list(sweep_lists.keys())
                values = [sweep_lists[key] for key in keys]
                permutations = product(*values)

                permuted_dicts = []

                for perm in permutations:
                    perm_dict = {key: value for key, value in zip(keys, perm)}
                    permuted_dicts.append(perm_dict)

                mid_idx = len(permuted_dicts) // 2
                first_half = permuted_dicts[:mid_idx]
                second_half = permuted_dicts[mid_idx:]
                reordered_list = [x for pair in zip(first_half, second_half) for x in pair]

                if len(permuted_dicts) % 2 != 0:
                    reordered_list.append(second_half[-1])

                sweep[model].extend(reordered_list)

        print(ds_name, [len(sweep[model_name]) for model_name in MODELS])

        with open(os.path.join(sweep_dir, f"{ds_name}.pkl"), "wb") as f:
            pickle.dump(sweep, f)

if __name__ == "__main__":
    gen_hp_sweep_dicts()


# BASE_SWEEPS = [
#     {
#         "ss_lrs": { 
#             "OWL_L14": [[8e-8, 8e-9], [8e-7, 8e-8], [2e-7, 2e-8]],
#             "OWL_B16": [[2e-7, 8e-9], [2e-6, 8e-8], [2e-7, 2e-8]]
#         },
#         "lw_lr_decay_mults": [[0.97, 0.97], [0.92, 0.92]],
#         "warmup_epochs": [3],
#         "lr_sched": ["cos warm"],
#         "epochs": [30],
#         "train_batch": {
#             "OWL_L14": [None],
#             "OWL_B16": [None]
#         }
#     },
#     {
#         "ss_lrs": { 
#             "OWL_L14": [[8e-8, 8e-9], [8e-7, 8e-8], [2e-7, 2e-8]],
#             "OWL_B16": [[2e-7, 8e-9], [2e-6, 8e-8], [2e-7, 2e-8]]
#         },
#         "lw_lr_decay_mults": [[0.97, 0.97], [0.92, 0.92]],
#         "warmup_epochs": [3],
#         "lr_sched": ["cos warm"],
#         "epochs": [30],
#         "train_batch": {
#             "OWL_L14": [None],
#             "OWL_B16": [None]
#         }
#     }
# ]