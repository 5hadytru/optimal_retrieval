import pickle, os
from pprint import pprint

for i in os.listdir("."):
    if any([x in i for x in ["dice", "BCCD", "Chess", "Shellfish", "Cheetah", "Aerial", "American", "brack"]]):
        with open(i, "rb") as f:
            d = pickle.load(f)
        for j in d["OWL_L14"]:
            j["epochs"] = 7
            j["warmup_epochs"] = 2

        with open(i, "wb") as f:
          pickle.dump(d, f)