import torch
import wandb
import faiss.contrib.torch_utils
import torch.distributed as dist


log_rank_0 = lambda d: wandb.log(d) if dist.get_rank() == 0 else None
print_rank_0 = lambda d: print(d) if dist.get_rank() == 0 else None


def get_top_k_idxes(logits:torch.Tensor, exclusive_classes:bool, n_features:int):
    """
    Finds the top k scores within an image and returns prompt indices and ViT token indices

    Args:
    scores: [num_instances, num_classes] array of scores (i.e. logits or
    probabilities) to sort by.
    k: Number of instances to return.
    exclusive_classes: If True, the top class for each box is returned. If
    False, classes are considered to be non-exclusive (multi-label setting),
    and the top-k computations happens globally across all scores, not just
    the maximum logit for each output token.

    Returns:
    Score, label, and box arrays of shape [top_k, ...] for the selected
    instances.
    """
    assert len(logits.shape) == 3

    k = n_features

    if exclusive_classes:
        k = min(k, logits.shape[1]) # k cannot be greater than the number of ViT tokens
        instance_top_scores, instance_class_ind = torch.max(logits, dim=2) # (B, num ViT tokens), (B, num ViT tokens)
        top_scores, instance_ind = torch.topk(instance_top_scores, k, dim=1) # (B, k), (B, k)
        class_ind = instance_class_ind.gather(-1, instance_ind) # (B, k)
    else:
        raise Exception("Have not yet implemented get_top_k_idxes for non-exclusive classes")
        k = min(k, logits.numel()) # k cannot be greater than the total number of class predictions
        top_scores, top_indices = torch.topk(logits.view(-1), k) # (k), (k)
        instance_ind, class_ind = unravel_index_2D(top_indices, logits.shape) # (k), (k)

    return instance_ind, class_ind


def batched_dist(
        ev_features, # (B, N, F)
        cand_features, # (C, M, F)
        metric: str, 
        chunk_sizes:list
    ):
    """
    Get distance/sim between all possible pairs of feature vectors in batches.
    """
    ev_batch_size, cand_batch_size = chunk_sizes

    # Initialize an empty tensor to store the distances
    distances = torch.empty((ev_features.size(0), ev_features.size(1), cand_features.size(0), cand_features.size(1)), device=ev_features.device)

    # Iterate over ev_features in batches
    for i in range(0, ev_features.size(0), ev_batch_size):
        ev_start = i
        ev_end = i + ev_batch_size if (i + ev_batch_size) < ev_features.size(0) else ev_features.size(0)
        ev_batch = ev_features[ev_start:ev_end]

        # Iterate over cand_features in batches
        for j in range(0, cand_features.size(0), cand_batch_size):
            cand_start = j
            cand_end = j + cand_batch_size if (j + cand_batch_size) < cand_features.size(0) else cand_features.size(0)
            cand_batch = cand_features[cand_start:cand_end]

            # Broadcast the batches for pairwise distance calculation
            ev_b = ev_batch.unsqueeze(2).unsqueeze(3).expand(-1, -1, cand_batch.size(0), cand_batch.size(1), -1)
            cand_b = cand_batch.unsqueeze(0).unsqueeze(1).expand(ev_b.size())

            # Calculate distances
            if metric == "cos":
                batch_dist = torch.nn.functional.cosine_similarity(ev_b, cand_b, dim=-1)
            elif metric == "L2":
                batch_dist = torch.norm(ev_b - cand_b, p=2, dim=-1)
            else:
                raise Exception(metric)

            # Store the calculated distances
            distances[ev_start:ev_end, :, cand_start:cand_end, :] = batch_dist

    return distances


def naive_dist(
        ev_features, # (B, N, F)
        cand_features, # (B, C, M, F) 
        metric:str
    ):
    """
    Get distance/sim between all possible pairs of feature vectors, with memory-intensive broadcasting
    """
    ev_features = torch.broadcast_to(
        ev_features.unsqueeze(2).unsqueeze(3),
        (ev_features.size(0), ev_features.size(1), cand_features.size(1), cand_features.size(2), -1)
    )

    cand_features = torch.broadcast_to(
        cand_features.unsqueeze(1),
        ev_features.size()
    )

    # memory-intensive due to the subtraction
    if metric == "L2":
        feature_dist = torch.norm(ev_features - cand_features, p=2, dim=-1) # (B, N, C, M)
    elif metric == "cos":
        feature_dist = torch.nn.functional.cosine_similarity(ev_features, cand_features, dim=-1) # (B, N, C, M)
    else:
        raise Exception(metric)

    return feature_dist


def batched_sim_with_indices(ev_queries, cand_queries, indices, chunk_sizes:list):
    """
    Get cosine sim of all matched queries, with *controllably* memory-intensive broadcasting
    """
    B, N, _ = ev_queries.shape
    C, _, __ = cand_queries.shape
    all_sims = torch.zeros((B, N, C), device=ev_queries.device)

    ev_batch_size, cand_batch_size = chunk_sizes

    for i in range(0, ev_queries.size(0), ev_batch_size):
        ev_start = i
        ev_end = i + ev_batch_size if (i + ev_batch_size) < ev_queries.size(0) else ev_queries.size(0)
        ev_batch = ev_queries[ev_start:ev_end]
        ev_batch_indices = indices[ev_start:ev_end] # (chunk[0], N, C)

        for j in range(0, cand_queries.size(0), cand_batch_size):
            cand_start = j
            cand_end = j + cand_batch_size if (j + cand_batch_size) < cand_queries.size(0) else cand_queries.size(0)

            cand_batch = cand_queries[cand_start:cand_end]
            indices_batch = ev_batch_indices[:, :, cand_start:cand_end] # (chunk[0], N, chunk[1])

            indices_batch_expanded = indices_batch.unsqueeze(-1).unsqueeze(-1).expand(
                -1, -1, -1, -1, cand_batch.size(-1)) # (chunk[0], N, chunk[1], 1, Q)

            cand_batch_expanded = cand_batch.unsqueeze(0).unsqueeze(0).expand(
                ev_batch_size, ev_queries.size(1), -1, -1, -1) # (chunk[0], N, chunk[1], M, Q)

            cand_batch_expanded = cand_batch_expanded.gather(3, indices_batch_expanded).squeeze(-2) # (chunk[0], N, chunk[1], Q)
            ev_batch_expanded = ev_batch.unsqueeze(-2).expand(-1, -1, cand_batch_expanded.size(-2), -1) # (chunk[0], N, chunk[1], Q)

            all_sims[ev_start:ev_end, :, cand_start:cand_end] = torch.nn.functional.cosine_similarity(ev_batch_expanded, cand_batch_expanded, dim=-1)

    return all_sims


def naive_sim_with_indices(
        ev_queries, # (B, N, Q)
        cand_queries, # (B, C, M, Q)
        indices:torch.Tensor # (B, N, C)
    ):
    """
    Get cosine sim of all matched queries, with memory-intensive broadcasting
    """
    # stack feature and matching query's similarity along the final dimension
    indices = indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, -1, cand_queries.size(-1)) # (B, N, C, 1, Q)
    cand_queries = cand_queries.unsqueeze(1).expand(-1, ev_queries.size(1), -1, -1, -1) # (B, N, C, M, Q)
    cand_queries = cand_queries.gather(3, indices).squeeze(-2) # (B, N, C, Q)
    ev_queries = ev_queries.unsqueeze(-2).expand(-1, -1, cand_queries.size(-2), -1) # (B, N, C, Q)
    query_sim = torch.nn.functional.cosine_similarity(ev_queries, cand_queries, dim=-1) # (B, N, C)

    return query_sim


def get_local_cand_set(
        features, # (B, C, N, F) 
        queries, # (B, C, N, Q)
        rank:int
    ):
    n_cand = features.size(1)
    cand_per_device = int(n_cand / dist.get_world_size())
    start_idx = rank * cand_per_device
    end_idx = (rank+1) * cand_per_device

    local_features = features[0, start_idx:end_idx]

    assert list(local_features.size()) == [cand_per_device, features.size(2), features.size(3)]

    if queries is None:
        return local_features # (C / world size, N, F)
    else:
        local_queries = queries[0, start_idx:end_idx]
        return local_features, local_queries


def precompute_A2(max_k:int, coreset_len:int):
    import numpy as np
    
    a2s = np.zeros((max_k + 1,))
    
    for cxval in np.arange(0, max_k + 1):
        reusable_sum = 0
        stable_ratio = 1
        for k in range(coreset_len):
            stable_ratio *= (coreset_len - k - cxval) / (coreset_len - k)
            reusable_sum += (1 / (k+1)) * (1 - stable_ratio)
        a2s[cxval] = reusable_sum - 1.0
        print(a2s[cxval])

    torch.save(torch.tensor(a2s), f"precomputed_A2s_{coreset_len}.pth")


def get_unique_topk_indices(all_indices:list, k:int, n_cand:int):
    """
    Given a [(B, k * world size) for _ in range(world size)] list of ordered top-k indices for each device, 
    return a (B * world size * k,) tensor containing the top-k indices for each B in the list such that they are
    all unique and maintain their order
    """
    assert len(all_indices[0].size()) == 2, str(all_indices[0].size())
    assert all_indices[0].size(-1) == dist.get_world_size() * k, str(all_indices[0].size())

    current_topk = torch.cat([device_indices[:, :k] for device_indices in all_indices], dim=0).flatten()
    for i in range(k, k * dist.get_world_size()):
        if current_topk.unique().numel() >= n_cand:
            break
        new_indices = torch.cat([device_indices[:, i] for device_indices in all_indices], dim=0)
        current_topk = torch.cat([current_topk, new_indices])
    
    final_topk = current_topk.unique()[:n_cand]

    assert final_topk.numel() == n_cand, str(final_topk.size())
    assert len(final_topk.size()) == 1, str(final_topk.size())

    return final_topk.to(torch.long)
