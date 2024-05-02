import pickle
import os, sys

"""
class_info = {
    id: {
        "name": str,
        "indices": List[int], # indices wrt the coreset/subset not the full dataset
        "img_losses": List[float],
        "proto_dists": List[float] # added in prototype_utils.py
    }
}
"""

def main():
    subset_params = sys.argv[1] # 0.15,0.0,50
    ot, vgt, ni = subset_params.split(",")
    indices_pth = f"idx_o365_{ot}_VG_{vgt}_N{ni}.pkl"
    with open(indices_pth, "rb") as f:
        subset_indices = pickle.load(f)

    with open("O365_stats.pkl", "rb") as f:
        full_stats = pickle.load(f)

    class_info = {}
    cls_name_to_id = {}

    # NOTE: this procedure assumes that all coreset indices are in O365; no VG
    for subset_idx, full_idx in enumerate(subset_indices):
        stats_dict = full_stats[full_idx]
        assert len(set(stats_dict["classes"])) == len(stats_dict["classes"])
        for cls_name in stats_dict["classes"]:
            if cls_name not in cls_name_to_id:
                cls_id = 0 if len(class_info) == 0 else max(class_info.keys()) + 1
                class_info[cls_id] = {"name": cls_name, "img_losses": [stats_dict["loss"]], "indices": [subset_idx]}
                cls_name_to_id[cls_name] = cls_id
            else:
                class_info[cls_name_to_id[cls_name]]["img_losses"].append(stats_dict["loss"])
                class_info[cls_name_to_id[cls_name]]["indices"].append(subset_idx)

    dump_pth = f"idx_o365_{ot}_VG_{vgt}_N{ni}_class_info.pkl"
    with open(dump_pth, "wb") as f:
        pickle.dump(class_info, f)


if __name__ == "__main__":
    main()