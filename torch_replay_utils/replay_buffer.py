import enum
import torch
import numpy as np
import wandb
from torch_replay_utils.coreset_utils import Coreset, load_feature_TDS, load_prototypes
import pickle
import time
import random
from torch import distributed as dist
from torch.utils.data import Subset
from pylibraft.common import DeviceResources
from pylibraft.neighbors.brute_force import knn


log_rank_0 = lambda d: wandb.log(d) if dist.get_rank() == 0 else None
print_rank_0 = lambda d: print(d) if dist.get_rank() == 0 else None

class ReplayBuffer:
    """
    Acts as an API for a replay buffer, which is a torch Dataset (wrapping a bunch of HDF5 files, for instance)
    """
    def __init__(self, device, cfg):
        self.device = device
        self.cfg = cfg
        self.rng = np.random.default_rng(seed=int(time.time()) + random.randint(0,10000))
        self._coreset = self.init_coreset()
        self.class_info, self.cls_name_map = self.load_class_info(shuffle=True)
        self.init_strat_specific_data(cfg.main.retrieval_strat)
        self.valid_indices = torch.arange(len(self._coreset), dtype=torch.long, device=torch.device(dist.get_rank()))
        self.current_indices = torch.tensor([], dtype=torch.long, device=torch.device(dist.get_rank()))

        if dist.get_rank() == 0:
            print("Coreset len:", len(self._coreset))


    def __len__(self):
        return len(self._coreset)
    

    def init_strat_specific_data(self, retrieval_strat:str):
        """
        Initialize attributes which are specific to certain retrieval strats
        """
        if retrieval_strat in ['RandomSUB']:    
            self.idx_probs = self.get_SUB_idx_probs()

        if retrieval_strat in ["RandomUB", "GRASP", "LWUB", "SWIL", "C_ASER"]:
            self.ub_class_i = 0
            self.ub_class_order = np.arange(len(self.class_info.keys()), dtype=np.int32)
            self.rng.shuffle(self.ub_class_order)

        if retrieval_strat in ["SWIL", "C_ASER"]:
            prototype_dict = load_prototypes(self.cfg.main.coreset_name, self.cfg.main.model_name, map_location=self.device)
            sorted_cls_names = sorted(prototype_dict.keys())
            self.prototypes = torch.stack([prototype_dict[c] for c in sorted_cls_names])
            self.prototype_i_to_cls_id = [self.cls_name_map[c] for c in sorted_cls_names]

            # logging probability histograms for each *dataset*
            self.swil_probs = [None]

            self.swil_stats_id = time.time() + random.randint(0,100000)

            self.swil_hist_dump_pth = f"tmp/stats/SWIL_{self.swil_stats_id}_{dist.get_rank()}.pkl"
            print_rank_0(f"Dumping stats at {self.swil_hist_dump_pth}")


    def get_SUB_idx_probs(self):
        """
        Precompute idx probabilities for hyper-efficient soft uniform-balanced retrieval
        """
        scaled_probs = {}
        for class_id, class_dict in self.class_info.items():
            indices = class_dict["indices"]
            scaled_freq = np.power(len(indices), self.cfg.main.sub_exp)
            for idx in indices:
                scaled_probs[int(idx)] = scaled_probs.get(int(idx), 0) + 1.0 / scaled_freq

        # Normalize probabilities to sum to 1
        total_prob_sum = sum(scaled_probs.values())
        normalized_probs = np.array([scaled_probs[idx] / total_prob_sum for idx in range(len(self._coreset))])

        return normalized_probs


    def load_class_info(self, shuffle:bool):
        self.class_info_pth = f"torch_replay_utils/idx_o365_{self.cfg.main.o365_thresh}_VG_{self.cfg.main.VG_thresh}_N{self.cfg.main.coreset_N}_class_info.pkl"
        with open(self.class_info_pth, "rb") as f:
            d = pickle.load(f)
    
        if shuffle:
            for v in d.values():
                self.rng.shuffle(v["indices"])

        cls_name_map = {class_dict["name"]: class_id for class_id, class_dict in d.items()}

        return d, cls_name_map


    def init_features(self, tds_cpu:torch.utils.data.TensorDataset, indices=None):
        if indices is None:
            tensors_cpu = tuple(t for t in tds_cpu.tensors)
        else:
            tensors_cpu = tuple(t[indices] for t in tds_cpu.tensors)

        if dist.get_rank() == 0:
            print(f"Features GB: {sum([t.numel() * 4 for t in tensors_cpu]) / 1000000000}")

        self._cls_embeds = tensors_cpu[0].to(self.device) if not self.cfg.main.bf16 else tensors_cpu[0].to(torch.bfloat16).to(self.device)
        self._image_features = tensors_cpu[1].to(self.device) if not self.cfg.main.bf16 else tensors_cpu[1].to(torch.bfloat16).to(self.device)
        self._query_embeds = tensors_cpu[2].to(self.device) if not self.cfg.main.bf16 else tensors_cpu[2].to(torch.bfloat16).to(self.device)

        self._cls_embeds = self._cls_embeds[:, :self.cfg.main.n_tok, :]
        self._image_features = self._image_features[:, :self.cfg.main.n_tok, :]
        self._query_embeds = self._query_embeds[:, :self.cfg.main.n_tok, :]


    def clear_features(self):
        del self._cls_embeds
        del self._image_features
        del self._query_embeds


    def init_coreset(self):
        full_coreset = Coreset(self.cfg.main.coreset_name, "Xy", self.cfg.main.model_name)
        full_features = load_feature_TDS(self.cfg.main.coreset_name, self.cfg.main.model_name)

        assert len(full_coreset) == len(full_features), f"{len(full_coreset)} {len(full_features)}"

        if self.cfg.main.base_coreset:
            self.init_features(full_features)
            return full_coreset

        indices_pth = f"torch_replay_utils/idx_o365_{self.cfg.main.o365_thresh}_VG_{self.cfg.main.VG_thresh}_N{self.cfg.main.coreset_N}.pkl"
        with open(indices_pth, "rb") as f:
            subset_indices = pickle.load(f)

        coreset_subset = Subset(full_coreset, subset_indices)
        self.init_features(full_features, indices=subset_indices)

        return coreset_subset


    def post_epoch_logs(self):
        """
        Do whatever buffer-related post-epoch logging here 
        """
        pass


    def reset_dedup(self):
        self.current_indices = torch.tensor([], dtype=torch.long, device=torch.device(dist.get_rank()))
        self.valid_indices = torch.arange(len(self._coreset), dtype=torch.long, device=torch.device(dist.get_rank()))

        if self.cfg.main.retrieval_strat == "SWIL":
            print_rank_0(f"Dumping stats at {self.swil_hist_dump_pth}")
            with open(self.swil_hist_dump_pth, "wb") as f:
                pickle.dump(self.swil_probs, f)
            self.swil_probs.append(None)


    def append_indices(self, global_indices:torch.Tensor):
        """
        Append indices to a globally-representative tensor on rank 0
        self.current_indices should be pre-allocated for maximum speed
        """
        assert dist.get_rank() == 0

        if self.cfg.main.reset_dedup_on == "batch":
            # never deduplicate
            return

        self.valid_indices = torch.tensor(np.setdiff1d(self.valid_indices.cpu().numpy(), global_indices.cpu().numpy()))
        self.current_indices = torch.cat([self.current_indices, global_indices])

        if self.cfg.main.reset_dedup_on == "proportion" and (self.cfg.main.dedup_prop * len(self._coreset)) < len(self.current_indices):
            self.reset_dedup()



    def load_batch(self, indices:torch.Tensor):
        """
        Given a list of coreset indices and dict of reserve indices, store them properly -> return a list of stacked tensors representing
        the batch gotten via 'indices'
        """
        #start = time.time()
        batch_elements = [self._coreset[i] for i in indices]
        # print_rank_0(["load batch:", time.time() - start])

        return [torch.stack([x[i] for x in batch_elements]) for i in range(len(batch_elements[0]))]

    def get_random_retrieval_indices(self, n, probs=None) -> torch.Tensor:
        """
        Return n random buffer indices
        """
        start = time.time()

        assert dist.get_rank() == 0
        
        if probs is None:
            possible_indices = np.arange(len(self._coreset), dtype=np.int64)
            invalid_indices = self.current_indices.cpu().numpy()
            sampling_pool = np.setdiff1d(possible_indices, invalid_indices)
            selected_indices = self.rng.choice(sampling_pool, size=n, replace=False)
        else:
            selected_indices = np.array([], dtype=np.int32)
            invalid_indices = self.current_indices.cpu().numpy()
            while len(selected_indices) < n:
                idxes = self.rng.choice(len(self._coreset), size=n - len(selected_indices), p=probs, replace=False)
                idxes = np.setdiff1d(idxes, np.concatenate([selected_indices, invalid_indices]))
                selected_indices = np.concatenate([selected_indices, idxes])
            
        print(f"random retrieval {n}:", time.time() - start)

        return torch.tensor(
            selected_indices, 
            dtype=torch.long, 
            device=torch.device(dist.get_rank())
        )
    

    def get_SUB_retrieval_indices(self, n:int):
        """
        Return n buffer indices with the Soft Uniform Balanced idx probabilities
        n will contain no duplicates w.r.t self.current_indices and itself
        """
        start = time.time()

        assert dist.get_rank() == 0

        selected_indices = np.array([], dtype=np.int32)
        invalid_indices = self.current_indices.cpu().numpy()

        while len(selected_indices) < n:
            idxes = self.rng.choice(len(self._coreset), size=n - len(selected_indices), p=self.idx_probs, replace=False)
            idxes = np.setdiff1d(idxes, np.concatenate([selected_indices, invalid_indices]))
            selected_indices = np.concatenate([selected_indices, idxes])

        print(f"SUB retrieval {n}:", time.time() - start)

        return torch.tensor(
            selected_indices, 
            dtype=torch.long, 
            device=torch.device(dist.get_rank())
        )
    

    def get_probs(self, class_id:int, prob_type:str):
        # NOTE: if you are going to use the array / sum(array) procedure, your values must all be positive!
        assert prob_type in ["GRASP", "LWUB", None]

        if prob_type == "GRASP":
            inverse_proto_dists = np.power(self.class_info[class_id]["proto_dists"] + 1e-8, -1 * self.cfg.main.ub_coeff)
            return inverse_proto_dists / np.sum(inverse_proto_dists)
        elif prob_type == "LWUB":
            inverse_img_losses = np.power(self.class_info[class_id]["img_losses"], -1 * self.cfg.main.ub_coeff)
            return inverse_img_losses / np.sum(inverse_img_losses)
        return None


    def get_UB_retrieval_indices(self, n:int, prob_type:str=None):
        """
        Return n random buffer indices while sampling each class equally 
        n will contain no duplicates w.r.t self.current_indices and itself
        Can also leverage per-idx values present in self.class_info to sample from each class in a weighted-prob manner
        """
        start = time.time()

        assert dist.get_rank() == 0

        selected_indices = np.array([], dtype=np.int32)
        invalid_indices = self.current_indices.cpu().numpy()

        def reset():
            self.ub_class_i = 0
            self.rng.shuffle(self.ub_class_order)

        while len(selected_indices) < n:
            if self.ub_class_i == len(self.ub_class_order):
                reset()

            current_class = self.ub_class_order[self.ub_class_i]
            all_invalid_indices = np.concatenate([selected_indices, invalid_indices])
            current_probs = self.get_probs(current_class, prob_type)

            # if current_probs is not None:
            #     wandb.log({
            #         "prob_mean": np.mean(current_probs),
            #         "prob_min": np.min(current_probs),
            #         "prob_max": np.max(current_probs),
            #         "prob_std": np.std(current_probs),
            #         "prob_n": current_probs.size,
            #     })

            class_indices = self.class_info[current_class]["indices"]
            n_cand_indices = len(np.setdiff1d(class_indices, all_invalid_indices))

            if n_cand_indices > 0:
                while True:
                    current_selection = self.rng.choice(class_indices, p=current_probs)
                    if len(np.setdiff1d(np.array([current_selection]), all_invalid_indices)) > 0:
                        break
                selected_indices = np.append(selected_indices, current_selection)

            self.ub_class_i += 1

        print(f"UB retrieval {n}:", time.time() - start)

        return torch.tensor(
            selected_indices, 
            dtype=torch.long, 
            device=torch.device(dist.get_rank())
        )


    def calculate_entropy(self, probabilities:np.ndarray):
        """
        Calculate the normalized Shannon entropy of a 1D numpy array of probabilities.

        Parameters:
        - probabilities: A 1D numpy array containing the probabilities of each outcome.

        Returns:
        - The normalized Shannon entropy of the distribution.
        """
        # Ensure the probabilities sum to 1 (within a reasonable tolerance)
        if not np.isclose(probabilities.sum(), 1):
            raise ValueError("The probabilities must sum to 1.")
        
        # Ensure there are no negative probabilities
        if np.any(probabilities < 0):
            raise ValueError("Probabilities cannot be negative.")
        
        # Calculate the Shannon entropy
        entropy = -np.sum(probabilities * np.log2(probabilities, where=(probabilities!=0)))
        
        # Calculate the maximum entropy
        n = len(probabilities)  # Number of outcomes
        max_entropy = np.log2(n)
        
        # Normalize the entropy
        normalized_entropy = entropy / max_entropy
        
        return normalized_entropy


    @torch.no_grad()
    def get_SWIL_retrieval_indices(
            self, 
            n:int,
            all_batch_features:torch.Tensor, # (B * world size, n_tok, embed dim)
            prob_type:str,
            adaptivity:str,
            inverted=False,
            ent_thresh=None,
            log_stats=False
        ):
        """
        Get indices according to Similarity Weighted Interleaved Learning (https://www.pnas.org/doi/10.1073/pnas.2115229119)
        """
        assert n % all_batch_features.size(0) == 0, f"{n} {all_batch_features.size(0)}"

        if inverted:
            sample_n_classes = 1 # 1 class per sample in the batch
            n_samples_per_class = n // all_batch_features.size(0) # *maximum* samples per class
        else:
            sample_n_classes = n // all_batch_features.size(0)
            n_samples_per_class = 1

        assert dist.get_rank() == 0

        if len(self.prototypes.size()) == 2:
            self.prototypes = self.prototypes.unsqueeze(1).unsqueeze(2) # (n classes, 1, 1, embed dim)

        # compute prototype distances
        proto_cos_dists = 1 - torch.nn.functional.cosine_similarity(
            self.prototypes, 
            all_batch_features.unsqueeze(0), # (1, batch size, n_tok, embed dim)
            dim=-1
        ) # (n_classes, batch size, n_tok)

        # take the mean or minimum prototype similarity for each image in the batch
        if self.cfg.main.swil_reduce_type == "avg":
            proto_cos_dists = proto_cos_dists.mean(dim=-1) # (n classes, B * world size)
        elif self.cfg.main.swil_reduce_type == "min":
            proto_cos_dists = proto_cos_dists.min(dim=-1)[0]
        else:
            raise Exception
        
        # sample classes and samples for each of the B * world size images + sample indices
        invalid_indices = self.current_indices.cpu().numpy()
        selected_indices = np.array([], dtype=np.int32)
        batch_idx_subsets = []
        max_iter = 1000000
        batch_class_probs = None
        for batch_i in range(proto_cos_dists.size(1)):
            batch_idx_subsets.append(np.array([], dtype=np.int64))
            for _ in range(sample_n_classes):
                all_invalid_indices = np.concatenate([invalid_indices, selected_indices])
                current_proto_dists = proto_cos_dists[:, batch_i] # (n classes,)
                inverse_proto_dists = np.power(current_proto_dists.cpu().numpy() + 1e-8, -1 * self.cfg.main.swil_cls_coeff)
                class_probs = inverse_proto_dists / np.sum(inverse_proto_dists)

                # if only using SWIL for images with low-entropy class prob distributions, use UB with GRASP instead
                if adaptivity == "img" and self.calculate_entropy(class_probs) > ent_thresh:
                    selected_idx = self.get_UB_retrieval_indices(n_samples_per_class, "GRASP").cpu().numpy()
                    selected_indices = np.append(selected_indices, selected_idx)
                    batch_idx_subsets[batch_i] = np.append(batch_idx_subsets[batch_i], np.array([selected_idx]))
                    all_invalid_indices = np.concatenate([invalid_indices, selected_indices])
                    log_rank_0({"ada": 1})
                    continue
                else:
                    log_rank_0({"ada": 0})

                # for logging stats
                if batch_class_probs is None:
                    batch_class_probs = np.expand_dims(class_probs, axis=1)
                else:
                    batch_class_probs = np.append(
                        batch_class_probs, 
                        np.expand_dims(class_probs, axis=1), 
                        axis=1
                    )

                # select class to sample from
                iters = 0
                while True:
                    selected_class_id = self.prototype_i_to_cls_id[int(self.rng.choice(len(class_probs), p=class_probs))]
                    if len(np.setdiff1d(self.class_info[selected_class_id]["indices"], all_invalid_indices)) > 0:
                        break
                    iters += 1
                    if iters > max_iter:
                        raise Exception("Class selection timed out")

                # sample from the class
                idx_probs = self.get_probs(selected_class_id, prob_type)
                for __ in range(n_samples_per_class):
                    # we have exhausted this class and will therefore not sample the maximum number of samples for it
                    if len(np.setdiff1d(self.class_info[selected_class_id]["indices"], all_invalid_indices)) == 0:
                        break

                    iters = 0
                    while True:
                        selected_idx = self.rng.choice(self.class_info[selected_class_id]["indices"], p=idx_probs)
                        if len(np.setdiff1d(np.array([selected_idx]), all_invalid_indices)) > 0:
                            break
                        iters += 1
                        if iters > max_iter:
                            raise Exception("Sample selection timed out")
                    selected_indices = np.append(selected_indices, selected_idx)
                    batch_idx_subsets[batch_i] = np.append(batch_idx_subsets[batch_i], np.array([selected_idx]))
                    all_invalid_indices = np.concatenate([invalid_indices, selected_indices])

        # log stats
        if log_stats:
            if self.swil_probs[-1] is None:
                self.swil_probs[-1] = batch_class_probs
            else:
                self.swil_probs[-1] = np.append(self.swil_probs[-1], batch_class_probs, axis=1)
                print(self.swil_probs[-1].shape)

        if inverted:
            return torch.tensor(
                selected_indices, 
                dtype=torch.long, 
                device=torch.device(dist.get_rank())
            ), [
                torch.tensor(
                    batch_subset, 
                    dtype=torch.long, 
                    device=torch.device(dist.get_rank())
                ) for batch_subset in batch_idx_subsets
            ]
        else:
            return torch.tensor(
                selected_indices, 
                dtype=torch.long, 
                device=torch.device(dist.get_rank())
            )


    def get_features(self, indices:torch.Tensor, cls_embeds:bool):
        """
        Get + organize features at the given indices
        """
        assert len(list(indices.size())) == 1

        if cls_embeds:
            return self._cls_embeds[indices], self._query_embeds[indices]
        else:
            return self._image_features[indices], self._query_embeds[indices]
        

    def get_query_embeds(self, indices=None):
        """
        Get all or some queries
        """
        if indices is None:
            return self._query_embeds
        else:
            return self._query_embeds[indices], indices
        

    def load_pc_SVs(self, cls_embeds, o365_thresh, VG_thresh, coreset_N):
        """
        Load pre-computed SVs into an attribute
        """
        pc_SV_pth = f'torch_replay_utils/pc_C_SVs_{"cls_embeds" if cls_embeds else "features"}_o365_{o365_thresh}_VG_{VG_thresh}_N{coreset_N}_{self.cfg.main.n_tok}.pth'
        pc_SVs = torch.load(pc_SV_pth, map_location=torch.device(dist.get_rank()))

        return pc_SVs
    

    def search_index(
            self, 
            queries:torch.Tensor,
            k:int
        ):
        """
        Return kNN distances and indices (indices into the entire coreset) for queries
        This is handled by the buffer for clean and proper deduplication; similar to get_SUB...
        """
        n_tok = self._image_features.size(1)
        feature_dim = self._image_features.size(-1)

        valid_features = self._image_features[self.valid_indices, :, :]
        assert valid_features.size(0) == self.valid_indices.numel()

        searchable_features = valid_features.view(-1, feature_dim) # (num valid idxes * n_tok, F)

        distances, indices = knn(searchable_features, queries, k, metric='l2')
        distances, indices = torch.tensor(distances), torch.tensor(indices)
        print_rank_0([distances.dtype, indices.dtype])
        row_indices = self.valid_indices[indices // n_tok]
        col_indices = indices % n_tok

        print_rank_0([self._image_features[self.valid_indices, :, :].size(), distances.size(), row_indices.size(), col_indices.size()])

        return distances, row_indices, col_indices
        
