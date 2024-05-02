"""
Contains functions for computing prototypes and for storing prototype scores for class embeddings on an image-level,
for specific coreset subsets
"""
import torch
import os, sys
import pickle


def merge_dicts(num_processes:int, model_name:str, merge_type:str):
    master = {}
    current_len = 0

    for p_i in range(num_processes):
        print("Merging", str(p_i))
        base_pth = f"../data/coresets/O365_VG/{model_name}"
        file_i = 0
        if merge_type == "bugged":
            while True:
                if not os.path.exists(os.path.join(base_pth, f"O365_VG_PEs_{p_i}_{file_i + 1}.pth")):
                    break
                file_i += 1
            file_pth = os.path.join(base_pth, f"O365_VG_PEs_{p_i}_{file_i}.pth")
            d = torch.load(file_pth, map_location=torch.device("cpu"))
            sorted_keys = sorted(list(d.keys()))
            for i in sorted_keys:
                master[current_len] = d[i]
                current_len += 1
        elif merge_type == "intended":
            file_pth = os.path.join(base_pth, f"O365_VG_PEs_{p_i}_{file_i}.pth")
            while os.path.exists(file_pth):
                print(file_i)
                d = torch.load(file_pth, map_location=torch.device("cpu"))
                sorted_keys = sorted(list(d.keys()))
                for i in sorted_keys:
                    master[current_len] = d[i]
                    current_len += 1
                file_i += 1
                file_pth = os.path.join(base_pth, f"O365_VG_PEs_{p_i}_{file_i}.pth")

    return master


def compute_prototypes(all_embeds:dict):
    # for each class, assemble all class embeddings into single tensors on CPU (requires lots of RAM)
    cls_to_full_indices = {}
    for idx, embeds_dict in all_embeds.items():
        for cls_name in embeds_dict.keys():
            if cls_name in cls_to_full_indices:
                cls_to_full_indices[cls_name].append(idx)
            else:
                cls_to_full_indices[cls_name] = [idx]
            
    cls_to_embeddings = {}
    for cls_name, indices in cls_to_full_indices.items():
        print(cls_name, "cat")
        cls_to_embeddings[cls_name] = torch.cat([all_embeds[i][cls_name] for i in indices], dim=0)

    # send to device and compute the average
    prototypes = {}
    for cls_name, embeddings in cls_to_embeddings.items():
        print(cls_name, "proto")
        prototypes[cls_name] = torch.mean(embeddings.to(torch.device(0)), dim=0)
        assert list(prototypes[cls_name].shape) == [embeddings.size(-1)]

    return prototypes


def add_proto_dists(protoypes:dict, all_embeddings:dict, subset_params:str):
    # open subset indices list and class info
    ot, vgt, ni = subset_params.split(",")
    indices_pth = f"idx_o365_{ot}_VG_{vgt}_N{ni}.pkl"
    class_info_pth = f"idx_o365_{ot}_VG_{vgt}_N{ni}_class_info.pkl"
    with open(indices_pth, "rb") as f:
        subset_indices = pickle.load(f)
    with open(class_info_pth, "rb") as f:
        class_info = pickle.load(f)

    # for each index in each entry in class info, compute and append prototype distance
    def compute_min_proto_dist(subset_idx, cls_name):
        full_idx = subset_indices[subset_idx]
        idx_embeds = all_embeddings[full_idx][cls_name] # (n, embed dim)
        proto = protoypes[cls_name] # (embed dim,)
        assert len(idx_embeds.size()) == 2
        assert list(proto.size()) == [idx_embeds.size(-1)]
        proto_dists = 1.0 - torch.nn.functional.cosine_similarity(proto.to(torch.device(0)), idx_embeds.to(torch.device(0)))
        assert list(proto_dists.size()) == [idx_embeds.size(0)], str(proto_dists.size())
        return torch.min(proto_dists)        

    for cls_id, cls_dict in class_info.items():
        print("Scoring", cls_id)
        c = cls_dict["name"]
        proto_dists = [compute_min_proto_dist(i, c) for i in cls_dict["indices"]]
        cls_dict["proto_dists"] = proto_dists
    
    with open(class_info_pth, "wb") as f:
        pickle.dump(class_info, f)


def main():
    num_processes = int(sys.argv[1])
    model_name = sys.argv[2]
    subset_params = sys.argv[3] # 0.15,0.0,50
    merge_type = sys.argv[4]

    assert merge_type in ["bugged", "intended"]

    all_embeddings = merge_dicts(num_processes, model_name, merge_type)
    prototypes = compute_prototypes(all_embeddings)

    torch.save(prototypes, f"../data/coresets/O365_VG/{model_name}/prototypes.pth")

    add_proto_dists(prototypes, all_embeddings, subset_params)


if __name__ == "__main__":
    main()