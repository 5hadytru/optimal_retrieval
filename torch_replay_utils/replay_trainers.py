from abc import ABC, abstractmethod
import torch
import time
from torch_replay_utils.replay_buffer import ReplayBuffer
from torch_replay_utils import opt_utils
from enum import Enum
import wandb
import functools
import faiss
import faiss.contrib.torch_utils
import torch.distributed as dist
from util_funcs.main_utils import prepare_batch
import torch_replay_utils.trainer_utils as trainer_utils
import numpy as np
import pickle
import random


class RetrievalTypes(Enum):
    NNR = 0
    RANDOM = 1
    NONE = 2

log_rank_0 = lambda d: wandb.log(d) if dist.get_rank() == 0 else None
print_rank_0 = lambda d: print(d) if dist.get_rank() == 0 else None


class ReplayTrainer(ABC):
    """
    Base class for creating a class which implements a specific retrieval strategy

    Must be coupled with the training loop since the retrieval strategy determines when/what data is replayed and stored
    """
    def __init__(self, device, cfg):
        self.cfg = cfg
        self.device = device
        self.replay_buffer = ReplayBuffer(device, cfg)
        self.retrieval_each_batch = cfg.main.retrieval_each_batch

        self.post_epoch_data = {}

        self.init_post_epoch_data()

        self.mixed_forward_pass = opt_utils.DER(cfg)
        self.batch_forward_pass = opt_utils.Basic(cfg)
        self.forward_pass_no_grad = opt_utils.NoGrad(cfg)


    def post_epoch_logs(self):
        """
        Logs for after every epoch
        """
        if dist.get_rank() == 0:
            wandb.log(self.post_epoch_data)

        for key in self.post_epoch_data:
            self.post_epoch_data[key] = 0

        self.replay_buffer.post_epoch_logs()
        
    def reset_dedup(self):
        """
        After each dataset, we clear the replay buffer
        """
        self.replay_buffer.reset_dedup()

    @abstractmethod
    def init_post_epoch_data(self):
        """
        Initialize the dictionary containing post-epoch wandb logs
        """
        pass


    def get_batch(self, local_indices:torch.Tensor, global_indices:torch.Tensor, append_indices:bool):
        if append_indices:
            if dist.get_rank() == 0:
                self.replay_buffer.append_indices(global_indices)

        return self.replay_buffer.load_batch(local_indices)


    def train_batch(self, *, inputs:dict, y:dict, model, optimizer, epoch):
        if epoch == 0 or self.retrieval_each_batch:
            loss_mean = self.forward_pass_retrieval(inputs, y, model, retrieve_func=self.retrieve)
        else:
            loss_mean, num_replays = self.forward_pass(inputs, y, model)

        loss_mean.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()


class Random(ReplayTrainer):
    """
    Random retrieval; can be uniformly random across classes or all samples
    """
    def __init__(self, device, cfg):
        super().__init__(device, cfg)
        self.forward_pass_retrieval = opt_utils.Random(cfg)
        
        # determine the function that will be used to get global retrieval indices on each batch, on device rank 0 
        if cfg.main.retrieval_strat == "RandomSUB":
            self.retr_func = self.replay_buffer.get_SUB_retrieval_indices
        elif cfg.main.retrieval_strat == "RandomUB":
            self.retr_func = self.replay_buffer.get_UB_retrieval_indices
        elif cfg.main.retrieval_strat in ["GRASP", "LWUB"]:
            self.retr_func = functools.partial(self.replay_buffer.get_UB_retrieval_indices, prob_type=cfg.main.retrieval_strat)
        elif cfg.main.retrieval_strat == "Random":
            self.ubal_SV = cfg.main.ubal_SV
            if self.ubal_SV and dist.get_rank() == 0:
                self.init_ubal_SV_probs(cfg.main.ubal_SV_coeff)
            else:
                self.probs = None
            self.retr_func = functools.partial(self.replay_buffer.get_random_retrieval_indices, probs=self.probs)
        else:
            raise Exception

        self.replay_buffer.clear_features()


    def init_post_epoch_data(self):
        pass


    @torch.no_grad()
    def retrieve(self, batch_len:int):
        """
        Get replay batch; adaptivity (also present here) is based on pre-SGD losses of current batch
        """
        total_replays = self.cfg.main.n_replays * batch_len * dist.get_world_size()
        if dist.get_rank() == 0:
            global_indices = self.retr_func(total_replays)
        else:
            global_indices = torch.zeros((total_replays,), dtype=torch.long, device=self.device)
        
        dist.broadcast(global_indices, src=0)

        indices_per_device = int(total_replays / dist.get_world_size())
        start_idx = dist.get_rank() * indices_per_device
        end_idx = (dist.get_rank() + 1) * indices_per_device
        local_indices = global_indices[start_idx:end_idx]
        local_replay_batch = self.get_batch(local_indices, global_indices, append_indices=True)

        return prepare_batch(local_replay_batch, self.cfg.main.prompt_type, device=self.device)


    def init_ubal_SV_probs(self, ubal_SV_coeff:float):
        pc_SVs = self.replay_buffer.load_pc_SVs(
            False, self.cfg.main.o365_thresh, self.cfg.main.VG_thresh, self.cfg.main.coreset_N).to(self.device)
        self.probs = torch.nn.functional.softmax(pc_SVs * ubal_SV_coeff, dim=0).cpu().numpy()
        print_rank_0(["Initialized probs", np.sum(self.probs), np.min(self.probs), np.max(self.probs), np.mean(self.probs)])


class SWIL(ReplayTrainer):
    """
    Similarity-weighted interleaved learning (https://www.pnas.org/doi/10.1073/pnas.2115229119)
    """
    def __init__(self, device, cfg):
        super().__init__(device, cfg)
        assert cfg.main.cls_embeds == True
        self.forward_pass_retrieval = opt_utils.Features(cfg, cfg.main.cls_embeds)
        self.prob_type = cfg.main.bal_prob_type
        self.n_tok = cfg.main.n_tok

        self.adaptivity = cfg.main.swil_ada
        assert self.adaptivity in ["none", "ds", "img"]

        self.ada_ent = cfg.main.swil_ada_ent
        if self.adaptivity == "img":
            assert self.ada_ent is not None
        self.SWIL_active = True


    def init_post_epoch_data(self):
        pass


    @torch.no_grad()
    def retrieve(self, local_class_embeds, local_query_embeds, local_logits):
        """
        Get replay batch; adaptivity (also present here) is based on pre-SGD losses of current batch
        """
        total_replays = self.cfg.main.n_replays * local_class_embeds.size(0) * dist.get_world_size()

        top_k_feature_idxes, _ = trainer_utils.get_top_k_idxes(local_logits, exclusive_classes=True, n_features=self.n_tok) # (B, n_tok), (B, n_tok)
        local_class_embeds = local_class_embeds.gather(1, top_k_feature_idxes.unsqueeze(-1).expand(-1, -1, local_class_embeds.size(-1)))

        # first gather all class embeddings
        if dist.get_rank() == 0:
            all_class_embeds = [torch.empty_like(local_class_embeds) for _ in range(dist.get_world_size())]
            dist.gather(local_class_embeds, all_class_embeds, dst=0)
            all_class_embeds = torch.cat(all_class_embeds)
            assert list(all_class_embeds.size()) == [local_class_embeds.size(0) * dist.get_world_size(), local_class_embeds.size(1), local_class_embeds.size(-1)], str(all_class_embeds.size())
        else:
            dist.gather(local_class_embeds, gather_list=None, dst=0)        

        # get global indices
        if dist.get_rank() == 0:
            if self.SWIL_active:
                global_indices = self.replay_buffer.get_SWIL_retrieval_indices(
                    total_replays, all_class_embeds, self.prob_type, self.adaptivity, ent_thresh=self.ada_ent)
            else:
                global_indices = self.replay_buffer.get_UB_retrieval_indices(total_replays, prob_type="GRASP")
        else:
            global_indices = torch.zeros((total_replays,), dtype=torch.long, device=self.device)
        
        dist.broadcast(global_indices, src=0)

        indices_per_device = int(total_replays / dist.get_world_size())
        start_idx = dist.get_rank() * indices_per_device
        end_idx = (dist.get_rank() + 1) * indices_per_device
        local_indices = global_indices[start_idx:end_idx]
        local_replay_batch = self.get_batch(local_indices, global_indices, append_indices=True)

        return prepare_batch(local_replay_batch, self.cfg.main.prompt_type, device=self.device)


class C_ASER(ReplayTrainer):

    """
    Variants of ASER (https://arxiv.org/pdf/2009.00093.pdf) with continuous classes etc etc
    """
    def __init__(self, device, cfg):
        super().__init__(device, cfg)
        self.forward_pass_retrieval = opt_utils.Features(cfg, cfg.main.cls_embeds)
        self.device = torch.device(dist.get_rank())

        self.n_cand = cfg.retrieval.caser.n_cand
        assert self.n_cand % dist.get_world_size() == 0

        self.knn_SV_K = cfg.retrieval.caser.knn_SV_K
        self.n_tok = cfg.main.n_tok
        self.feat_chunks = cfg.retrieval.caser.feat_chunks
        self.query_chunks = cfg.retrieval.caser.query_chunks
        self.no_buff = self.cfg.retrieval.caser.no_buff
        self.chunked = self.cfg.retrieval.caser.chunked
        self.SV_ratio = self.cfg.retrieval.caser.SV_ratio
        self.cls_embeds = self.cfg.main.cls_embeds
        self.prob_type = self.cfg.main.bal_prob_type
        self.swil = self.cfg.retrieval.caser.swil
        self.balanced = self.cfg.retrieval.caser.balanced
        self.batch_idx_subsets = None

        if self.swil:
            assert cfg.main.cls_embeds, "Must implement C-ASER with image features and SWIL with class embeds combo"
        else:
            assert not self.balanced, "Have not implemented balance for non-swil candidate sets"

        self.SV_type = self.cfg.retrieval.caser.SV_type
        assert self.SV_type in ["KNN", "TKNN"]
        if self.SV_type == "TKNN":
            self.precomputed_A2s = torch.load(f"torch_replay_utils/precomputed_A2s_{len(self.replay_buffer)}.pth").to(self.device).to(torch.float32)

        self.l2_thresh = self.cfg.retrieval.caser.l2_thresh

        self.cand_set_type = self.cfg.retrieval.caser.cand_set_type
        assert self.cand_set_type in ["D", "FD", "U", "KNN"]

        self.pc = self.cfg.retrieval.caser.pc
        if self.pc:
            self.pc_SVs = self.replay_buffer.load_pc_SVs(
                self.cls_embeds, cfg.main.o365_thresh, cfg.main.VG_thresh, cfg.main.coreset_N).to(self.device)

        self.log_stats = self.cfg.retrieval.caser.log_stats
        if self.log_stats:
            self.bins = {
                "L2": np.concatenate([
                    np.array([0.0, 1.5]),
                    np.arange(2.5, 7.0, 0.25),
                    np.arange(7.0, 9.0, 0.5),
                    np.arange(9.0, 25.0, 1.5),
                ]),
                "Qsim": np.arange(-1.0, 1.0, 0.05),
                "ranks": np.array(list(range(self.n_cand + 1))),
                "batch_SV": np.arange(-0.1, 0.1, 0.0001),
                "ASV": np.arange(-0.1, 0.1, 0.0001)
            }
            self.accum_hist = {k: np.zeros(len(v) - 1) for k,v in self.bins.items()}
            self.log_count = 0

            self.stats_id = time.time() + random.randint(0,100000)

            self.hist_dump_pth = f"tmp/stats/SVh_{self.stats_id}_{dist.get_rank()}.pkl"
            print_rank_0(f"Dumping stats at {self.hist_dump_pth}, {self.log_count}")

        self.initial_start = torch.cuda.Event(enable_timing=True)
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)
        self.final_end = torch.cuda.Event(enable_timing=True)


    def init_post_epoch_data(self):
        pass


    @torch.no_grad()
    def retrieve(self, image_features, query_embeds, logits):
        """
        Get replay batch; adaptivity (also present here) is based on pre-SGD losses of current batch
        """
        top_k_feature_idxes, top_k_query_idxes = trainer_utils.get_top_k_idxes(logits, exclusive_classes=True, n_features=self.n_tok) # (B, n_tok), (B, n_tok)
        image_features = image_features.gather(1, top_k_feature_idxes.unsqueeze(-1).expand(-1, -1, image_features.size(-1)))
        query_embeds = query_embeds.gather(1, top_k_query_idxes.unsqueeze(-1).expand(-1, -1, query_embeds.size(-1)))

        original_coreset_indices = self.get_local_coreset_indices(
            image_features.size(0), batch_features=image_features if (self.cand_set_type == "KNN" or self.swil) else None)

        if len(original_coreset_indices.size()) == 1:
            coreset_features, coreset_query_embeds = self.replay_buffer.get_features(original_coreset_indices, self.cls_embeds)
            coreset_features = coreset_features.unsqueeze(0).expand(image_features.size(0), -1, -1, -1)
            coreset_query_embeds = coreset_query_embeds.unsqueeze(0).expand(image_features.size(0), -1, -1, -1)
            original_coreset_indices = original_coreset_indices.unsqueeze(0).expand(image_features.size(0), -1)
        elif len(original_coreset_indices.size()) == 2:
            coreset_features, coreset_query_embeds = self.replay_buffer.get_features(original_coreset_indices.view(-1), self.cls_embeds)
            coreset_features = coreset_features.view(image_features.size(0), original_coreset_indices.size(-1), self.n_tok, coreset_features.size(-1))
            coreset_query_embeds = coreset_query_embeds.view(image_features.size(0), original_coreset_indices.size(-1), self.n_tok, coreset_query_embeds.size(-1))
        else:
            raise Exception(str(original_coreset_indices.size()))

        assert len(original_coreset_indices.size()) == 2, str(original_coreset_indices.size())
        assert list(coreset_features.size()) == [image_features.size(0), original_coreset_indices.size(-1), self.n_tok, coreset_features.size(-1)]
        assert list(coreset_query_embeds.size()) == [image_features.size(0), original_coreset_indices.size(-1), self.n_tok, coreset_query_embeds.size(-1)]

        # self.start.record()
        batch_SVs, coreset_indices1 = self.compute_shapley_values(
            ev_features=image_features, 
            ev_queries=query_embeds, 
            cand_features=coreset_features, 
            cand_queries=coreset_query_embeds, 
            original_indices=original_coreset_indices,
            chunked=False, 
            SV_type="batch"
        )

        if self.log_stats:
            tmp_svs = batch_SVs.min(dim=0)[0]
            log_rank_0({
                "pct_nonzero": torch.count_nonzero(tmp_svs) / tmp_svs.numel(), # (C,)
                "SV_min": tmp_svs.min(),
                "SV_max": tmp_svs.max(),
                "SV_mean": tmp_svs.mean(),
            })
        
        if not self.no_buff:
            #start = time.time()
            if not self.pc:
                assert self.cand_set_type == "U", "On-demand buff SVs would require a complete rework and/or massive cand set reduction for D/FD"

                local_coreset_features, local_coreset_query_embeds = trainer_utils.get_local_cand_set(
                    coreset_features, coreset_query_embeds, dist.get_rank())

                n_local_cands = local_coreset_features.size(0)

                # need to expand the first dimension of the coreset features, queries, and indices to 
                # match the local slice of the candidate set, as opposed to the local batch
                resized_coreset_features = coreset_features[0].unsqueeze(0).expand(n_local_cands, -1, -1, -1)
                resized_coreset_query_embeds = coreset_query_embeds[0].unsqueeze(0).expand(n_local_cands, -1, -1, -1)
                resized_coreset_indices = original_coreset_indices[0].unsqueeze(0).expand(n_local_cands, -1)

                buffer_SVs, resized_coreset_indices = self.compute_shapley_values(
                    ev_features=local_coreset_features, 
                    ev_queries=local_coreset_query_embeds, 
                    cand_features=resized_coreset_features, 
                    cand_queries=resized_coreset_query_embeds, 
                    original_indices=resized_coreset_indices,
                    chunked=self.chunked, 
                    SV_type="buffer"
                )

                local_batch_size = image_features.size(0)
                coreset_indices2 = resized_coreset_indices[:local_batch_size, :]
            else:
                buffer_SVs = self.pc_SVs[coreset_indices1.view(-1)].reshape(coreset_indices1.size()) # (B, C)
                coreset_indices2 = coreset_indices1

            #print_rank_0(["buffer SV:", time.time() - start])

            assert torch.sum(coreset_indices1 == coreset_indices2) == coreset_indices1.numel()
        else:
            buffer_SVs = None

        # self.end.record()
        # torch.cuda.synchronize()
        # print_rank_0(["SV took:", self.start.elapsed_time(self.end) / 1000])

        local_replay_indices, global_replay_indices = self.get_local_replay_batch(
            batch_SVs,
            buffer_SVs,
            coreset_indices1,
            image_features.size(0) * dist.get_world_size()
        )

        # self.final_end.record()
        # torch.cuda.synchronize()
        # print_rank_0(["retrieve took:", self.initial_start.elapsed_time(self.final_end) / 1000]) # CUDA will be synchronized with CPU at this point so we can use 'time'

        b = self.get_batch(local_replay_indices, global_replay_indices, append_indices=True)

        #print("------")

        return prepare_batch(b, self.cfg.main.prompt_type, device=self.device)
    

    def get_local_coreset_indices(self, local_batch_size:int, batch_features:torch.Tensor=None):
        """
        Get local coreset indices; will return set of indices of size (n_cand,) or (B, n_cand)
        """
        def get_global_indices(n:int):
            # gather all features if SWIL
            if self.swil:
                if dist.get_rank() == 0:
                    all_batch_features = [torch.empty_like(batch_features) for _ in range(dist.get_world_size())]
                    dist.gather(batch_features, all_batch_features, dst=0)
                    all_batch_features = torch.cat(all_batch_features)
                    assert list(all_batch_features.size()) == [batch_features.size(0) * dist.get_world_size(), batch_features.size(1), batch_features.size(-1)], str(all_batch_features.size())
                else:
                    dist.gather(batch_features, gather_list=None, dst=0)

            # get global indices on device 0 then broadcast
            if dist.get_rank() == 0:
                if self.swil:
                    if self.balanced:
                        global_coreset_indices, batch_idx_subsets = self.replay_buffer.get_SWIL_retrieval_indices(
                            n, all_batch_features, self.prob_type, "none", inverted=True)
                        self.batch_idx_subsets = batch_idx_subsets
                    else:
                        global_coreset_indices = self.replay_buffer.get_SWIL_retrieval_indices(n, all_batch_features, self.prob_type, "none")
                else:
                    global_coreset_indices = self.replay_buffer.get_UB_retrieval_indices(n, self.prob_type) # deduplicated across the epoch

            if dist.get_rank() == 0:
                cand_set_size = torch.tensor([len(global_coreset_indices)], dtype=torch.long, device=torch.device(0))
            else:
                cand_set_size = torch.tensor([-1], dtype=torch.long, device=torch.device(dist.get_rank()))
            
            dist.broadcast(cand_set_size, src=0)

            if dist.get_rank() != 0:
                global_coreset_indices = torch.zeros((int(cand_set_size.cpu().numpy()),), dtype=torch.long, device=self.device)
            dist.broadcast(global_coreset_indices, src=0)
            
            return global_coreset_indices

        if self.cand_set_type == "U":
            # unified candidate set across devices; get candidate set on rank 0 then broadcast
            global_indices = get_global_indices(self.n_cand)
            return global_indices
        elif self.cand_set_type == "D":
            # distributed; each device gets a different candidate set
            global_indices = get_global_indices(self.n_cand * dist.get_world_size())

            assert global_indices.size(0) == self.n_cand * dist.get_world_size(), "Must re-implement this for variable cand set sizes"

            local_indices_start = self.n_cand * dist.get_rank()
            local_indices_end = self.n_cand * (dist.get_rank() + 1)

            return global_indices[local_indices_start:local_indices_end]
        elif self.cand_set_type == "FD":
            # fully distributed; each sample on each device gets a different candidate set
            global_indices = get_global_indices(self.n_cand * dist.get_world_size() * local_batch_size)

            assert global_indices.size(0) == self.n_cand * dist.get_world_size(), "Must re-implement this for variable cand set sizes"

            local_indices_start = self.n_cand * local_batch_size * dist.get_rank()
            local_indices_end = self.n_cand * local_batch_size * (dist.get_rank() + 1)

            return global_indices[local_indices_start:local_indices_end].view(local_batch_size, self.n_cand)
        elif self.cand_set_type == "KNN":
            # K nearest neighbors of each evaluation point are retrieved and combined 
            # into a unified candidate set, with deduplication (hence the k * world_size step)
            k = self.n_cand // (dist.get_world_size() * batch_features.size(0)) + 1 # number of cands per batch item (assuming no duplicates)
            _, local_coreset_indices, __ = self.replay_buffer.search_index(
                batch_features.view(-1, batch_features.size(-1)), k * dist.get_world_size())
            local_coreset_indices = local_coreset_indices.to(torch.long).to(self.device)
            if dist.get_rank() == 0:
                all_coreset_indices = [torch.zeros_like(local_coreset_indices) for _ in range(dist.get_world_size())]
                dist.gather(local_coreset_indices, all_coreset_indices, dst=0)
            else:
                dist.gather(local_coreset_indices, gather_list=None, dst=0)

            # now broadcast unified candidate set indices to all ranks
            if dist.get_rank() == 0:
                global_coreset_indices = trainer_utils.get_unique_topk_indices(all_coreset_indices, k, self.n_cand)
            else:
                global_coreset_indices = torch.zeros(
                    (self.n_cand,), dtype=torch.long, device=self.device)
            dist.broadcast(global_coreset_indices, src=0)

            raise Exception("Ensure KNN implementation is compatible with variable n_cand")

            return global_coreset_indices
        else:
            raise Exception


    def get_local_replay_batch(
            self, 
            local_batch_SVs:torch.Tensor, 
            local_buff_SVs:torch.Tensor,
            local_indices:torch.Tensor,
            global_batch_size:int
        ):
        """
        Rank 0 will process the shapley values for the entire batch
        """
        assert self.no_buff or local_buff_SVs is not None

        if self.no_buff:
            batch_SVs, buffer_SVs, all_indices = self.gather_shapley_values(local_batch_SVs, local_indices)
        else:
            batch_SVs, buffer_SVs, all_indices = self.gather_shapley_values(local_batch_SVs, local_indices, local_buff_SVs)

        if dist.get_rank() == 0:
            ASVs = self.aggregate_shapley_values(batch_SVs, all_indices, buffer_SVs)

            assert len(ASVs.size()) == 1 and ASVs.size(0) == len(self.replay_buffer)

            # now take the top-k ASVs in a balanced or imbalanced manner
            if self.balanced:
                global_replay_indices = torch.tensor([], dtype=torch.long, device=ASVs.device)
                for idx_subset in self.batch_idx_subsets:
                    idx_subset_tens = torch.tensor(idx_subset, device=ASVs.device)
                    batch_item_indices = ASVs[idx_subset_tens].topk(k=self.cfg.main.n_replays)[1]
                    global_replay_indices = torch.cat((global_replay_indices, idx_subset_tens[batch_item_indices]))
                self.batch_idx_subsets = None # ensure this set of subsets is not used again
            else:
                global_replay_indices = ASVs.topk(k=global_batch_size * self.cfg.main.n_replays)[1]
            
            assert global_replay_indices.size(0) == global_batch_size * self.cfg.main.n_replays and len(global_replay_indices.size()) == 1
        else:
            global_replay_indices = torch.zeros(
                (global_batch_size  * self.cfg.main.n_replays,),
                dtype=torch.long,
                device=self.device
            )

        dist.broadcast(global_replay_indices, src=0)

        indices_per_device = int(global_replay_indices.size(0) / dist.get_world_size())
        start_idx = dist.get_rank() * indices_per_device
        end_idx = (dist.get_rank() + 1) * indices_per_device

        local_replay_indices = global_replay_indices[start_idx:end_idx]

        return local_replay_indices, global_replay_indices


    def gather_shapley_values(
            self, 
            local_batch_SVs, # (B, C)
            local_indices, # (B, C)
            local_buffer_SVs=None
        ):
        """
        Gather and concatenate shapley values on device rank 0
        """
        # get batch SVs
        if dist.get_rank() == 0:
            all_batch_SVs = [torch.empty_like(local_batch_SVs) for _ in range(dist.get_world_size())]
            dist.gather(local_batch_SVs, all_batch_SVs, dst=0)
        else:
            dist.gather(local_batch_SVs, gather_list=None, dst=0)

        # get indices
        if dist.get_rank() == 0:
            all_indices = [torch.empty_like(local_indices) for _ in range(dist.get_world_size())]
            dist.gather(local_indices, all_indices, dst=0)
        else:
            dist.gather(local_indices, gather_list=None, dst=0)

        # get buffer SVs
        if local_buffer_SVs is not None:
            assert local_batch_SVs.size(-1) == local_buffer_SVs.size(-1)

            if dist.get_rank() == 0:
                all_buffer_SVs = [torch.empty_like(local_buffer_SVs) for _ in range(dist.get_world_size())]
                dist.gather(local_buffer_SVs, all_buffer_SVs, dst=0)
            else:
                dist.gather(local_buffer_SVs, gather_list=None, dst=0)
        
        if dist.get_rank() == 0:
            batch_SVs = torch.cat(all_batch_SVs)
            all_indices = torch.cat(all_indices)

            if local_buffer_SVs is not None:
                assert len(local_buffer_SVs.shape) > 1
                buffer_SVs = torch.cat(all_buffer_SVs)

                if self.pc:
                    assert list(buffer_SVs.size()) == list(batch_SVs.size())
                else:
                    if self.balanced:
                        raise Exception("Must check the below assertion")
                    assert list(buffer_SVs.size()) == [self.n_cand, self.n_cand]

                return batch_SVs, buffer_SVs, all_indices
            else:
                return batch_SVs, None, all_indices
        else:
            return None, None, None
        

    def aggregate_batch_SVs(self, batch_SVs, coreset_indices):
        # first, get a (B * world size, coreset len) tensor of batch SVs
        coreset_batch_SVs = torch.full(
            (batch_SVs.size(0), len(self.replay_buffer)), 
            float('inf'), 
            device=self.device
        ) # (B * world size, coreset size)
        assert list(coreset_batch_SVs.size()) == [batch_SVs.size(0), len(self.replay_buffer)], f"{list(coreset_batch_SVs.size())} {[batch_SVs.size(0), len(self.replay_buffer)]}"
        dim0_idxes = torch.arange(batch_SVs.size(0)).unsqueeze(1).to(self.device)
        coreset_batch_SVs[dim0_idxes, coreset_indices] = batch_SVs

        # then min along dim 0
        coreset_batch_SVs, _ = coreset_batch_SVs.min(dim=0) # default to min across batch

        # finally, replace infs with zeros
        coreset_batch_SVs[coreset_batch_SVs == float('inf')] = 0.0

        return coreset_batch_SVs
    

    def aggregate_pc_buffer_SVs(self, buffer_SVs, coreset_indices):
        # first, get a (B * world size, coreset len) tensor of buffer SVs
        coreset_buffer_SVs = torch.zeros(
            (buffer_SVs.size(0), len(self.replay_buffer)), 
            device=self.device
        ) # (B * world size, coreset size)
        dim0_idxes = torch.arange(buffer_SVs.size(0)).unsqueeze(1).to(self.device)
        coreset_buffer_SVs[dim0_idxes, coreset_indices] = buffer_SVs

        # now need to average while ignoring zeros
        nonzero_mask = coreset_buffer_SVs != 0
        nonzero_per_column = nonzero_mask.sum(dim=0) # (coreset_len,)
        nonzero_per_column[nonzero_per_column == 0] = 1.0 # avoid division by 0
        columnwise_sums = coreset_buffer_SVs.sum(dim=0)

        return columnwise_sums / nonzero_per_column
    

    def aggregate_shapley_values(
            self, 
            batch_SVs, # (B * world size, C)
            coreset_indices, # (B * world size, C)
            buffer_SVs=None # (B * world size, C) or (C, C)
        ):
        """
        Get a global ASV for each candidate set item + log metrics
        """
        # arrange batch and buffer SVs into (coreset_len,) tensors
        coreset_batch_SVs = self.aggregate_batch_SVs(batch_SVs, coreset_indices)

        if buffer_SVs is not None:
            if self.pc:
                # need to account for coreset_indices which are heterogenous across dim 0 here; FD or D case
                # same as aggregating batch SVs but averaging instead of minning across dim 0
                coreset_buff_SVs = self.aggregate_pc_buffer_SVs(buffer_SVs, coreset_indices)
            else:
                # in the case of a unified candidate set, just avg buffer SVs across candidates
                # then create (coreset size,) tensor of buffer SVs directly as opposed to (C, coreset size) -> mean(dim=0)
                assert self.cand_set_type == "U"
                assert buffer_SVs.size(0) == buffer_SVs.size(1), str(buffer_SVs.size())
                assert (coreset_indices.to(torch.float).mean(dim=0).to(torch.long) == coreset_indices[0]).sum() == coreset_indices[0].numel()
                coreset_buff_SVs = torch.zeros((len(self.replay_buffer),), device=self.device) # (coreset size,)
                coreset_buff_SVs[coreset_indices[0]] = buffer_SVs.mean(dim=0) # default to avg across batch

            assert list(coreset_batch_SVs.size()) == list(coreset_buff_SVs.size())

            if self.SV_ratio is not None:
                buff_SV_abs_max = torch.abs(coreset_buff_SVs.max()).to(self.device)
                batch_SV_abs_min = torch.abs(coreset_batch_SVs.min()).to(self.device)

                coreset_buff_SVs *= self.SV_ratio * (batch_SV_abs_min / buff_SV_abs_max)

            ASVs = coreset_buff_SVs - coreset_batch_SVs


            log_rank_0({
                "pct_buffer_driven":  np.setdiff1d(
                    ASVs.topk(k=batch_SVs.size(0))[1].cpu().numpy(), 
                    (-coreset_batch_SVs).topk(k=batch_SVs.size(0))[1].cpu().numpy()
                ).size / batch_SVs.size(0)
            })
        else:
            assert self.no_buff
            ASVs = -coreset_batch_SVs

        assert list(ASVs.size()) == [len(self.replay_buffer)], f"{list(ASVs.size())} {[len(self.replay_buffer)]}"

        if self.log_stats:
            self.accum_hist["batch_SV"] += np.histogram(batch_SVs.to(torch.float).cpu().numpy(), bins=self.bins["batch_SV"])[0]
            self.accum_hist["ASV"] += np.histogram(ASVs.to(torch.float).cpu().numpy(), bins=self.bins["batch_SV"])[0]
            self.log_count += 1

            if self.log_count % 10 == 0:
                with open(self.hist_dump_pth, "wb") as f:
                    pickle.dump({"bins": self.bins, "hists": self.accum_hist, "count": self.log_count}, f)

        return ASVs


    def knn_SV(
            self, 
            query_sims:torch.Tensor # (B, N, C), sorted
        ):
        current_SVs = torch.zeros(query_sims.size()).to(self.device) # (B, N, C)
        last_SV = torch.zeros_like(current_SVs[:-1]) # (B, N)
        last_query_sim = torch.zeros_like(current_SVs[:-1]) # (B, N)

        recursion_start = torch.cuda.Event(enable_timing=True)
        recursion_end = torch.cuda.Event(enable_timing=True)
        
        n_cand = query_sims.size(-1)

        # recursion_start.record()
        for SV_i in range(query_sims.size(-1)):
            if SV_i == 0:
                current_SVs[:,:,SV_i] = query_sims[:,:,SV_i] / n_cand
            else:
                m = n_cand - SV_i
                current_SVs[:,:,SV_i] = last_SV + (((query_sims[:,:,SV_i] - last_query_sim) / self.knn_SV_K) * (min(self.knn_SV_K, m) / m))

            last_SV = current_SVs[:,:,SV_i]
            last_query_sim = query_sims[:,:,SV_i]
        # recursion_end.record()
        # torch.cuda.synchronize()
        # print_rank_0([f"{current_SVs.size(0)} recursion took:", recursion_start.elapsed_time(recursion_end) / 1000])
    
        return current_SVs


    def tknn_SV(self, query_sims:torch.Tensor, distances:torch.Tensor, C:int=400):
        # apply L2 threshold
        if self.l2_thresh is not None:
            mask = (distances < self.l2_thresh).to(self.device) # (B, N, C)
        else:
            mask = torch.ones(distances.size(), dtype=torch.bool).to(self.device)

        # compute each necessary c term; all are (B, N, C)
        c_xval = torch.sum(mask, dim=-1) # number of sub-threshold nearest neighbors for each ev token
        c_xval_broad = c_xval.unsqueeze(-1).broadcast_to(distances.size())
        c_zval_plus = torch.sum(query_sims[mask], dim=-1).broadcast_to(distances.size()) # sum of query sims for each ev token's neighbors

        # compute A1 and A2
        A1 = (query_sims / c_xval_broad) - (c_zval_plus / (torch.pow(c_xval_broad, 2) - c_xval_broad)) # (B, N, C)
        A2 = self.precomputed_A2s[c_xval.view(-1).to(torch.long)].view(c_xval.size()).unsqueeze(-1).broadcast_to(c_xval_broad.size())

        A_term_coeff = (c_xval_broad > 2).to(torch.float32)

        SVs = (A_term_coeff * A1 * A2) + ((query_sims - 1 / C) / c_xval_broad)
        SVs[~mask] = 0.0 # (B, N, C)

        return SVs # (B, N, C)


    def compute_shapley_values(
            self,
            ev_features:torch.Tensor, # (B, N, F)
            ev_queries:torch.Tensor, # (B, N, Q)
            cand_features:torch.Tensor, # (B, C, M, F)
            cand_queries:torch.Tensor, # (B, C, M, Q)
            original_indices:torch.Tensor, # (B, C)
            SV_type:str, # ["batch", "buffer"]
            chunked:bool
        ):
        if chunked: # controllably less memory-intensive
            assert self.cand_set_type == "U" # chunked functions expect (C, M, F) tensors; in the unified setting the tensors are repeated across dim 0
            feature_l2 = trainer_utils.batched_dist(ev_features, cand_features[0, :, :, :], "L2", self.feat_chunks)
        else: # most memory-intensive but fastest
            feature_l2 = trainer_utils.naive_dist(ev_features, cand_features, "L2") # (B, N, C, M)

        # get the min distance of features for each candidate/image
        feature_l2, indices = feature_l2.min(dim=-1) # (B, N, C), (B, N, C)

        if chunked: # controllably less memory-intensive
            assert self.cand_set_type == "U" # chunked functions expect (C, M, F) tensors; in the unified setting the tensors are repeated across dim 0
            query_sim = trainer_utils.batched_sim_with_indices(ev_queries, cand_queries[0, :, :, :], indices, self.query_chunks)
        else: # most memory-intensive but fastest
            query_sim = trainer_utils.naive_sim_with_indices(ev_queries, cand_queries, indices)

        if self.log_stats:
            self.accum_hist["L2"] += np.histogram(feature_l2.to(torch.float).cpu().numpy(), bins=self.bins["L2"])[0]
            self.accum_hist["Qsim"] += np.histogram(query_sim.to(torch.float).cpu().numpy(), bins=self.bins["Qsim"])[0]

        # for each batch item, sort the images in descending order by feature similarity
        feature_l2, feature_indices = feature_l2.sort(dim=-1, descending=True) # (B, N, C), (B, N, C)

        # must permute coreset indices to maintain their validity
        coreset_indices = torch.broadcast_to(
            original_indices.unsqueeze(1),
            feature_l2.size()
        ).gather(-1, feature_indices)

        # now align query similarities with feature distances
        query_sim = query_sim.gather(-1, feature_indices) # (B, N, C)

        assert (list(feature_l2.size()) == [ev_features.size(0), ev_features.size(1), cand_features.size(1)])
        assert list(feature_l2.size()) == list(query_sim.size())

        # compute ASV for each token for each image
        if self.SV_type == "KNN":
            current_SVs = self.knn_SV(query_sim)
        elif self.SV_type == "TKNN":
            current_SVs = self.tknn_SV(query_sim, feature_l2)
        else:
            raise Exception(self.SV_type)

        if self.log_stats:
            if SV_type == "buffer":
                raise Exception("Probably do not want this")
                ranks = current_SVs.max(dim=-1)[1]
            else:
                ranks = current_SVs.min(dim=-1)[1]
            self.accum_hist["ranks"] += np.histogram(ranks.to(torch.int).cpu().numpy(), bins=self.bins["ranks"])[0]

        # first need to realign the SVs such that they can be averaged along the token dimension (dim 1)
        aligned_coreset_indices, sort_indices = torch.sort(coreset_indices, dim=-1)
        aligned_SVs = current_SVs.gather(-1, sort_indices)

        # now need to min/max/avg each image's shapley value along the token/feature dimension
        if SV_type == "batch":
            SVs, _ = torch.min(aligned_SVs, dim=1)
        elif SV_type == "buffer":
            SVs, _ = torch.max(aligned_SVs, dim=1)
        else:
            raise Exception(SV_type)

        assert torch.sum(torch.mean(aligned_coreset_indices.to(torch.float), dim=1).to(torch.long) == aligned_coreset_indices[:,0,:]) == aligned_coreset_indices[:,0,:].numel()

        return SVs, aligned_coreset_indices[:,0,:].contiguous()
